from typing import Dict, List, Optional, NamedTuple
from typing import Tuple, Dict
import logging
from collections import namedtuple

import numpy as np
import numpy.typing as npt
import pygame

try:
    from gymnasium.spaces import Box, Discrete
    from gymnasium.utils import seeding

    gym_v26 = True
except:
    from gym.spaces import Box, Discrete
    from gym.utils import seeding

    gym_v26 = False

from aintelope.environments.env_utils.render_ascii import AsciiRenderState
from aintelope.environments.env_utils.distance import distance_to_closest_item

logger = logging.getLogger("aintelope.environments.savanna")

# typing aliases
ObservationFloat = np.float32
PositionFloat = np.float32
Action = int
AgentId = str
AgentStates = Dict[AgentId, np.ndarray]

Observation = np.ndarray
Reward = float
Info = dict

Step = Tuple[
    Dict[AgentId, Observation],
    Dict[AgentId, Reward],
    Dict[AgentId, bool],
    Dict[AgentId, bool],
    Dict[AgentId, Info],
]

# environment constants
ACTION_MAP = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=PositionFloat)


class RenderSettings:
    def __init__(self, metadata):
        prefix = "render_"
        settings = {
            (k.lstrip(prefix), v) for k, v in metadata.items() if k.startswith(prefix)
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
        scale = window_size / self.settings.map_max

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
            assert len(agent_pos) == 2, ("malformed agent_pos", agent_pos)
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


def reward_agent(min_grass_distance):
    return 1 / (1 + min_grass_distance)


def move_agent(
    agent_pos: np.ndarray, action: Action, map_min=0, map_max=100
) -> np.ndarray:
    move = ACTION_MAP[action]
    agent_pos = agent_pos + move
    agent_pos = np.clip(agent_pos, map_min, map_max)
    return agent_pos


def get_agent_pos_from_state(agent_state) -> List[PositionFloat]:
    return [agent_state[0], agent_state[1]]


class SavannaEnv:
    # @zoo-api
    metadata = {
        "name": "savanna-v2",
        "render_fps": 3,
        "render_agent_radius": 5,
        "render_agent_color": (200, 50, 0),
        "render_grass_radius": 5,
        "render_grass_color": (20, 200, 0),
        "render_modes": ("human", "ascii", "offline"),
        "render_window_size": 512,
        "amount_agents": 1,
        "map_min": 0,
        "map_max": 10,
        "amount_grass_patches": 2,
        "amount_water_holes": 0,
        "num_iters": 1,
    }

    def __init__(self, env_params={}):
        self.metadata.update(env_params)
        logger.info(f"initializing savanna env with params: {self.metadata}")

        # @zoo-api
        self.possible_agents = [
            f"agent_{r}" for r in range(self.metadata["amount_agents"])
        ]
        # FIXME: needed?
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.metadata["amount_agents"])))
        )

        # for @zoo-api
        self._action_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }  # agents can walk in 4 directions

        # for @zoo-api
        self._observation_spaces = {
            agent: Box(
                low=self.metadata["map_min"],
                high=self.metadata["map_max"],
                shape=(
                    2
                    * (
                        self.metadata["amount_agents"]
                        + self.metadata["amount_grass_patches"]
                        + self.metadata["amount_water_holes"]
                    ),
                ),
            )
            for agent in self.possible_agents
        }

        # our own state
        self.agent_states: AgentStates = {}
        self.seed()

        render_settings = RenderSettings(self.metadata)
        self.render_state = RenderState(render_settings)
        self.human_render_state = None
        self.ascii_render_state = None
        self.dones = None

    def seed(self, seed: Optional[int] = None) -> None:
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, seed: Optional[int] = None, options=None):
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
        self.seed(seed)

        self.agents = self.possible_agents[:]
        # self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.dones = {agent: False for agent in self.agents}
        # self.infos = {agent: {} for agent in self.agents}
        self.grass_patches = self.np_random.integers(
            self.metadata["map_min"],
            self.metadata["map_max"],
            size=(self.metadata["amount_grass_patches"], 2),
        ).astype(PositionFloat)
        self.water_holes = self.np_random.integers(
            self.metadata["map_min"],
            self.metadata["map_max"],
            size=(self.metadata["amount_water_holes"], 2),
        ).astype(PositionFloat)
        self.agent_states = {
            agent: self.np_random.integers(
                self.metadata["map_min"], self.metadata["map_max"], 2
            ).astype(PositionFloat)
            for agent in self.agents
        }
        self.num_moves = 0

        # # cycle through the agents; needed for wrapper
        # self._agent_selector = agent_selector(self.agents)
        # self.agent_selection = self._agent_selector.next()
        self.dones = {agent: False for agent in self.agents}
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations

    def step(self, actions: Dict[str, Action]) -> Step:
        """step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - info
        dicts where each dict looks like {agent_1: action_of_agent_1, agent_2: action_of_agent_2}
        or generally {<agent_name>: <agent_action or None if agent is done>}
        """
        logger.debug("debug actions", actions)
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        if self.agents == []:
            raise ValueError("No agents found; num_iters reached?")

        rewards = {}
        for agent in self.agents:
            action = actions.get(agent)
            if isinstance(action, dict):
                action = action.get(agent)
            if action is None:
                continue

            logger.debug("debug action", action)
            self.agent_states[agent] = move_agent(
                self.agent_states[agent],
                action,
                map_min=self.metadata["map_min"],
                map_max=self.metadata["map_max"],
            )
            min_grass_distance = distance_to_closest_item(
                self.agent_states[agent], self.grass_patches
            )
            rewards[agent] = reward_agent(min_grass_distance)

        self.num_moves += 1
        env_done = (self.num_moves >= self.metadata["num_iters"]) or all(
            [actions.get(agent) is None for agent in self.agents]
        )
        self.dones = {
            agent: env_done or (actions.get(agent) is None) for agent in self.agents
        }

        observations = {agent: self.observe(agent) for agent in self.agents}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos: Dict[AgentId, dict] = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []
        logger.debug("debug return", observations, rewards, self.dones, infos)
        return observations, rewards, self.dones, infos

    def observe(self, agent: str) -> npt.NDArray[ObservationFloat]:
        """Return observation of given agent."""

        def stack(*args) -> npt.NDArray[ObservationFloat]:
            return np.hstack(args, dtype=ObservationFloat)

        observations = stack(self.agent_states[agent])
        for x in self.grass_patches:
            observations = stack(observations, x)
        for x in self.water_holes:
            observations = stack(observations, x)
        # just put all positions into one row
        res = observations.reshape(-1)
        assert (
            res.shape == next(iter(self._observation_spaces.values())).shape
        ), "observation / observation space shape mismatch"
        return res

    def set_agent_position(self, agent: str, loc: npt.NDArray[ObservationFloat]):
        """ Move the agent to a location. Tests and inference"""
        self.agent_states[agent] = loc
    
    def state_to_namedtuple(self, state: npt.NDArray[ObservationFloat]) -> NamedTuple:
        """Method to convert a state array into a named tuple."""
        agent_coords = {"agent_coords": state[:2]}
        grass_patches_coords = {}
        gp_offset = 2
        water_holes_coords = {}
        wh_offset = 2 + self.metadata["amount_grass_patches"] * 2
        for i in range(self.metadata["amount_grass_patches"]):
            grass_patches_coords[f"grass_patch_{i}"] = state[
                gp_offset + i : gp_offset + i + 2
            ]
        for i in range(self.metadata["amount_water_holes"]):
            water_holes_coords[f"water_hole_{i}"] = state[
                wh_offset + i : wh_offset + i + 2
            ]

        keys = (
            list(agent_coords) + list(grass_patches_coords) + list(water_holes_coords)
        )
        StateTuple = namedtuple("StateTuple", {k: np.ndarray for k in keys})
        return StateTuple(**agent_coords, **grass_patches_coords, **water_holes_coords)

    def render(self, mode="human"):
        """Render the environment."""

        self.render_state.render(self.agent_states, self.grass_patches)

        if mode == "human":
            if not self.human_render_state:
                self.human_render_state = HumanRenderState(self.render_state.settings)
            self.human_render_state.render(self.render_state)
        elif mode == "ascii":
            if not self.ascii_render_state:
                self.ascii_render_state = AsciiRenderState(
                    self.agent_states,
                    self.grass_patches,
                    self.render_state.settings,
                )
            self.ascii_render_state.render(self.agent_states, self.grass_patches)
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
