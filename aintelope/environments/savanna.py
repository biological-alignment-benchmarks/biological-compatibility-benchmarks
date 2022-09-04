import functools
import typing as typ

import numpy as np
import pygame
from gym.spaces import Box, Discrete
from gym.utils import seeding
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector, wrappers, parallel_to_aec

# typing aliases
PositionFloat = np.float32
Action = int

# environment constants
NUM_ITERS = 500  # duration of the game
MAP_MIN, MAP_MAX = 0, 100
AMOUNT_AGENTS = 1  # for now only one agent
AMOUNT_GRASS_PATCHES = 2
ACTION_MAP = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=PositionFloat)


class RenderSettings:
    def __init__(self, metadata):
        prefix = "render_"
        settings = {
            (k.lstrip(prefix), v)
            for k, v in metadata.items()
            if k.startswith(prefix)
        }
        self.__dict__.update(settings)


class RenderState:
    def __init__(self, settings):
        canvas = pygame.Surface((settings.window_size, settings.window_size))
        self.canvas = canvas
        self.settings = settings

    def render(self, agents_state, grass):
        window_size = self.settings.window_size
        canvas = self.canvas

        canvas.fill((255, 255, 255))
        scale = window_size / MAP_MAX

        screen_m = np.identity(2, dtype=PositionFloat) * scale

        def project(p):
            return np.matmul(p, screen_m).astype(np.int32)

        for gr in grass.reshape((2, -1)):
            p = project(gr)
            pygame.draw.circle(
                canvas,
                self.settings.grass_color,
                p,
                scale * self.settings.grass_radius,
            )

        for agent, agent_pos in agents_state.items():
            assert len(agent_pos) == 2, agent_pos
            # TODO: render agent name as text
            p = project(agent_pos)
            pygame.draw.circle(
                canvas,
                self.settings.agent_color,
                p,
                scale * self.settings.agent_radius,
            )


class HumanRenderState:
    def __init__(self, settings):

        self.fps = settings.fps

        window_size = settings.window_size

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()

    def render(self, render_state):
        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(render_state.canvas, render_state.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.fps)


def vec_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> np.float64:
    return np.linalg.norm(np.subtract(vec_a, vec_b))


def reward_agent(
    agent_pos: np.ndarray, grass_patches: np.ndarray
) -> np.float64:
    if len(grass_patches.shape) == 1:
        grass_patches = np.expand_dims(grass_patches, 0)
    assert (
        grass_patches.shape[1] == 2
    ), f"{grass_patches.shape} -- x/y index with axis=1"

    grass_patch_closest = grass_patches[
        np.argmin(
            np.linalg.norm(np.subtract(grass_patches, agent_pos), axis=1)
        )
    ]

    return 1 / (1 + vec_distance(grass_patch_closest, agent_pos))


def move_agent(agent_pos: np.ndarray, action: Action) -> np.ndarray:
    assert agent_pos.dtype == PositionFloat, agent_pos.dtype
    move = ACTION_MAP[action]
    agent_pos = agent_pos + move
    agent_pos = np.clip(agent_pos, MAP_MIN, MAP_MAX)
    return agent_pos


class RawEnv(ParallelEnv):

    metadata = {
        "name": "savanna_v1",
        "render_fps": 15,
        "render_agent_radius": 5,
        "render_agent_color": (200, 50, 0),
        "render_grass_radius": 5,
        "render_grass_color": (20, 200, 0),
        "render_modes": ("human", "offline"),
        "render_window_size": 512,
    }

    def __init__(self):
        self.possible_agents = [f"agent_{r}" for r in range(AMOUNT_AGENTS)]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(AMOUNT_AGENTS)))
        )

        self._action_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }  # agents can walk in 4 directions
        self._observation_spaces = {
            agent: Box(
                MAP_MIN,
                MAP_MAX,
                shape=(2 * (AMOUNT_AGENTS + AMOUNT_GRASS_PATCHES),),
            )
            for agent in self.possible_agents
        }

        render_settings = RenderSettings(self.metadata)
        self.render_state = RenderState(render_settings)
        self.human_render_state = None
        self.seed()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def seed(self, seed: typ.Optional[int] = None) -> None:
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent: str):
        """Return observation of given agent."""
        return np.concatenate(
            [self.agent_states[agent], self.grass_patches.reshape(-1)]
        )

    def render(self, mode="human"):
        """Render the environment."""

        self.render_state.render(self.agent_states, self.grass_patches)

        if mode == "human":
            if not self.human_render_state:
                self.human_render_state = HumanRenderState(
                    self.render_state.settings
                )
            self.human_render_state.render(self.render_state)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.render_state.canvas)),
                axes=(1, 0, 2),
            )

    def close(self):
        """Release any graphical display, subprocesses, network connections
        or any other environment data which should not be kept around after
        the user is no longer using the environment.
        """
        raise NotImplementedError

    def reset(self, seed: typ.Optional[int] = None):
        """Reset needs to initialize the following attributes:
            - agents
            - rewards
            - _cumulative_rewards
            - dones
            - infos
            - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        """
        if seed is not None:
            self.seed(seed)

        self.agents = self.possible_agents[:]
        # self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.dones = {agent: False for agent in self.agents}
        # self.infos = {agent: {} for agent in self.agents}
        self.grass_patches = self.np_random.integers(
            MAP_MIN, MAP_MAX, size=(AMOUNT_GRASS_PATCHES, 2)
        ).astype(PositionFloat)
        self.agent_states = {
            agent: self.np_random.integers(MAP_MIN, MAP_MAX, 2).astype(
                PositionFloat
            )
            for agent in self.agents
        }
        self.num_moves = 0

        # cycle through the agents; needed for wrapper
        # self._agent_selector = agent_selector(self.agents)
        # self.agent_selection = self._agent_selector.next()

        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations

    def step(self, actions: typ.Dict[str, Action]):
        """step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - info
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """  # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        if self.agents == []:
            raise ValueError("No agents found; NUM_ITERS reached?")

        for agent in self.agents:
            self.agent_states[agent] = move_agent(
                self.agent_states[agent], actions[agent]
            )
        rewards = {
            agent: reward_agent(self.agent_states[agent], self.grass_patches)
            for agent in self.agents
        }

        self.num_moves += 1
        env_done = self.num_moves >= NUM_ITERS
        dones = {agent: env_done for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, dones, infos


def raw_env():
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = RawEnv()
    env = parallel_to_aec(env)
    return env


def env():
    """Add PettingZoo wrappers to environment class."""
    env = RawEnv()
    # BaseWrapper class need agent_selection attribute
    # env = wrappers.AssertOutOfBoundsWrapper(env)
    # env = wrappers.OrderEnforcingWrapper(env)
    return env


if __name__ == "__main__":
    e = env()
    ret = e.reset()
    print(ret)
    ret = e.step({"agent_0": 1})
    print(ret)
