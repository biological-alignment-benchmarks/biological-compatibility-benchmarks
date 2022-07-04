import functools
import random
import typing as typ
from pprint import pprint

import numpy as np
import pygame
from gym.spaces import Box, Discrete
from gym.utils import seeding
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers

# typing aliases
PositionFloat = np.float32
Action = int

# environment constants
NUM_ITERS = 500  # duration of the game
MAP_MIN, MAP_MAX = 0, 100
AMOUNT_AGENTS = 1  # for now only one agent
AMOUNT_GRASS_PATCHES = 2
ACTION_MAP = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=PositionFloat)

# numerical constants
EPS = 0.0001
INF = 9999999999


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


class RawEnv(AECEnv):

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
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
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

        # cycle through the agents
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

    def step(self, action: Action):
        """Take in an action for the current agent (specified by
        agent_selection) and needs to update:
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        if self.dones[self.agent_selection]:
            """
            handles stepping an agent which is already done
            accepts a None action for the one agent, and moves the
            agent_selection to the next done agent, or if there are no more
            done agents, to the next live agent
            """
            # FIXME: why is it REQUIRED to call step() on a done agent?!
            return self._was_done_step(action)

        agent = self.agent_selection
        """
        the agent which stepped last had its _cumulative_rewards accounted for
        (because it was returned by last()), so the _cumulative_rewards for
        this agent should start again at 0
        """
        self._cumulative_rewards[agent] = 0

        self.agent_states[agent] = move_agent(self.agent_states[agent], action)

        # collect reward if it is the last agent to act
        if self._agent_selector.is_last():
            for iagent, agent in enumerate(self.agents):
                self.rewards[agent] = reward_agent(
                    self.agent_states[agent], self.grass_patches
                )

            self.num_moves += 1
            # The dones dictionary must be updated for all players.
            self.dones = {
                agent: self.num_moves >= NUM_ITERS for agent in self.agents
            }

        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()


def env():
    """Add PettingZoo wrappers to environment class."""
    env = RawEnv()
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class RandomWalkAgent:
    def __call__(self, action_space, observation, reward, info):
        return action_space.sample()


class OneStepPerfectPredictionAgent:
    def __call__(self, action_space, observation, reward, info):
        # FIXME: are you fucking kidding me?!
        agent_pos, grass = observation[:2], observation[2:].reshape(2, -1)
        bestreward = -INF
        ibestaction = 0
        for iaction in range(action_space.n):
            p = move_agent(agent_pos, iaction)
            reward = reward_agent(p, grass)
            if reward > bestreward:
                bestreward = reward
                ibestaction = iaction
        print(observation)
        print(reward, iaction)
        return ibestaction


class IterativeWeightOptimizationAgent:
    def __init__(self):
        self.is_initialized = False

    def __call__(self, action_space, observation, reward, info):
        MIN_WEIGHT = 0.05
        learning_rate = 0.01
        learning_randomness = 0.00

        LAST_ACTION_KEY = "last_action"
        LAST_REWARD_KEY = "last_reward"
        ACTIONS_WEIGHTS = "actions_weights"

        if not self.is_initialized:
            info[ACTIONS_WEIGHTS] = np.repeat([1.0], action_space.n)
            self.is_initialized = True

        print("step:", reward, observation)
        last_action = info.get(LAST_ACTION_KEY)
        last_reward = info.get(LAST_REWARD_KEY, 0)
        action_weights = info[ACTIONS_WEIGHTS]
        # avoid big weight change on the first valid step
        if last_action is not None and last_reward > EPS:
            last_action_reward_delta = reward - last_reward
            last_action_weight = action_weights[last_action]
            print(
                "dreward",
                last_action_reward_delta,
                last_action,
                ACTION_MAP[last_action],
            )
            last_action_weight += last_action_reward_delta * learning_rate
            last_action_weight = max(MIN_WEIGHT, last_action_weight)
            action_weights[last_action] = last_action_weight
            print("action_weights", action_weights)

            weight_sum = np.sum(action_weights)
            action_weights /= weight_sum

        def cdf(ds):
            res = {}
            x = 0
            for k, v in ds:
                x += v
                res[k] = x
            for k in res:
                res[k] /= x
            return res

        def choose(cdf):
            assert cdf
            x = random.uniform(0, 1 - EPS)
            k = None
            for k, v in cdf.items():
                if x >= v:
                    return k
            return k

        action_weights_cdf = cdf(enumerate(action_weights))
        print(
            "cdf",
            ", ".join(
                [
                    f"{ACTION_MAP[iaction]}: {w}"
                    for iaction, w in action_weights_cdf.items()
                ]
            ),
        )

        pprint(action_weights_cdf)
        action = choose(action_weights_cdf)
        if random.uniform(0, 1) < learning_randomness:
            action = action_space.sample()
        info[LAST_ACTION_KEY] = action
        info[LAST_REWARD_KEY] = reward
        print("chose action", action, ACTION_MAP[action])
        return action


def main(env: RawEnv):
    mode = "human"
    policy = IterativeWeightOptimizationAgent()
    policy = OneStepPerfectPredictionAgent()
    env.reset()
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()

        action_space: Discrete = env.action_space(agent)
        if not done:
            action = policy(action_space, observation, reward, info)
            assert action in action_space
        else:
            action = None
        env.step(action)
        env.render(mode)
    wait = input("Close?")


if __name__ == "__main__":
    e = env()
    main(e)

# Local Variables:
# compile-command: "poetry run python savanna.py"
# End:
