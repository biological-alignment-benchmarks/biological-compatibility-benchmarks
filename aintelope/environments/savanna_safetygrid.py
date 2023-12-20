from typing import Dict, List, Optional, Tuple, NamedTuple, Union
import logging
from collections import namedtuple, OrderedDict

import numpy as np
import numpy.typing as npt

from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv, ParallelEnv

from aintelope.environments.env_utils.distance import distance_to_closest_item

# from ai_safety_gridworlds.environments.shared.mo_reward import mo_reward
from ai_safety_gridworlds.helpers.gridworld_zoo_aec_env import GridworldZooAecEnv
from ai_safety_gridworlds.helpers.gridworld_zoo_parallel_env import (
    GridworldZooParallelEnv,
    Actions,
    INFO_OBSERVATION_COORDINATES,
    INFO_OBSERVATION_LAYERS_DICT,
    INFO_OBSERVATION_LAYERS_CUBE,
    INFO_AGENT_OBSERVATIONS,
    INFO_AGENT_OBSERVATION_COORDINATES,
    INFO_AGENT_OBSERVATION_LAYERS_DICT,
    INFO_AGENT_OBSERVATION_LAYERS_CUBE,
)

from ai_safety_gridworlds.environments.aintelope.aintelope_smell import (
    # TODO: import agent char map from env object instead?
    GAME_ART,
    AGENT_CHR1,
    AGENT_CHR2,
    AGENT_CHR3,
    DRINK_CHR,
    FOOD_CHR,
)

from aintelope.environments.typing import (
    ObservationFloat,
    # PositionFloat,
    # Action,
    AgentId,
    # AgentStates,
    Observation,
    Reward,  #  TODO: use np.ndarray or mo_reward
    Info,
)


logger = logging.getLogger("aintelope.environments.savanna_safetygrid")

# typing aliases
Action = Actions  # int

Step = Tuple[
    Dict[AgentId, Observation],
    Dict[AgentId, Reward],
    Dict[AgentId, bool],
    Dict[AgentId, bool],
    Dict[AgentId, Info],
]


class GridworldZooBaseEnv:
    metadata = {
        "seed": None,  # This seed is used only for environment map randomisation. Later the test calls .seed() method on the wrapper and this will determine the random action sampling and other random events during the game play.
        # "name": "savanna-safetygrid-v1",
        # "render_fps": 3,
        "render_agent_radius": 5,
        # "render_agent_color": (200, 50, 0),
        # "render_grass_radius": 5,
        # "render_grass_color": (20, 200, 0),
        # "render_modes": ("human", "ascii", "offline"),
        # "render_window_size": 512,
        "amount_agents": 1,
        # "map_min": 0,
        # "map_max": 10,   # TODO
        "amount_grass_patches": 2,
        "amount_water_holes": 0,
        "num_iters": 1,
        "observation_direction_mode": 0,  # TODO: Joel wanted to use relative direction, so need to use mode 1 or 2 in this case  # 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions.
        "action_direction_mode": 0,  # TODO: Joel wanted to use relative direction, so need to use mode 1 or 2 in this case    # 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions.
        "map_randomization_frequency": 1,  # TODO   # 0 - off, 1 - once per experiment run, 2 - once per trial (a trial is a sequence of training episodes separated by env.reset call, but using a same model instance), 3 - once per training episode.
        "remove_unused_tile_types_from_layers": True,  # Whether to remove tile types not present on initial map from observation layers. - set to False when same agent brain is trained over multiple environments
        "observe_bitmap_layers": False,  # Alternate observation format to current vector of absolute coordinates. Bitmap representation enables representing objects which might be outside of agent's observation zone for time being.
        "override_infos": False,  # Needed for tests. Zoo is unable to compare infos unless they have simple structure.
    }

    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}
        self.metadata.update(env_params)
        logger.info(f"initializing savanna env with params: {self.metadata}")

        self.super_initargs = {
            "env_name": "aintelope.aintelope_smell",
            "seed": self.metadata[
                "seed"
            ],  # This seed is used only for environment map randomisation. Later the test calls .seed() method on the wrapper and this will determine the random action sampling and other random events during the game play.
            "max_iterations": self.metadata["num_iters"],
            "amount_food_patches": self.metadata["amount_grass_patches"],
            "amount_drink_holes": self.metadata["amount_water_holes"],
            "amount_agents": self.metadata["amount_agents"],
            "observation_radius": self.metadata[
                "render_agent_radius"
            ],  # TODO: is render_agent_radius meant as diameter actually?
            "observation_direction_mode": self.metadata[
                "observation_direction_mode"
            ],  # 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions.
            "action_direction_mode": self.metadata[
                "action_direction_mode"
            ],  # 0 - fixed, 1 - relative, depending on last move, 2 - relative, controlled by separate turning actions.
            "map_randomization_frequency": self.metadata[
                "map_randomization_frequency"
            ],  # 0 - off, 1 - once per experiment run, 2 - once per trial (a trial is a sequence of training episodes separated by env.reset call, but using a same model instance), 3 - once per training episode.
            "remove_unused_tile_types_from_layers": self.metadata[
                "remove_unused_tile_types_from_layers"
            ],  # Whether to remove tile types not present on initial map from observation layers. - set to False when same agent brain is trained over multiple environments
        }

        self._observe_bitmap_layers = self.metadata["observe_bitmap_layers"]
        self._override_infos = self.metadata["override_infos"]

    def init_observation_spaces(self):
        # for @zoo-api
        self.transformed_observation_spaces = {
            agent: Box(
                low=0,
                high=len(
                    GAME_ART[0][0]
                ),  # TODO: consider height as well and read it from env object
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

        qqq = True  # for debugging

    # this method has no side effects
    def transform_observation(self, agent: str, info) -> npt.NDArray[ObservationFloat]:
        if self._observe_bitmap_layers:
            return info[INFO_OBSERVATION_LAYERS_CUBE]

        else:
            # NB! So far the savanna code has been using absolute coordinates, not relative coordinates.
            # In case of relative coordinates, sometimes an object might be outside of agent's observation distance.
            # If you want to return object location as agent-centric boolean bitmap, then it is easy to set all cells to False. But with coordinates you need either special values or an additional boolean dimension which indicates whether the coordinate is available or not.

            # TODO: import agent char map from env instead
            agent_observations = []
            for agent_name, agent_chr in self.agent_name_mapping.items():
                agent_observations += list(
                    info[INFO_OBSERVATION_COORDINATES][agent_chr][0]
                )  # convert tuple to list
            for x in info[INFO_OBSERVATION_COORDINATES].get(FOOD_CHR, []):
                agent_observations += list(x)  # convert tuple to list
            for x in info[INFO_OBSERVATION_COORDINATES].get(DRINK_CHR, []):
                agent_observations += list(x)  # convert tuple to list

            agent_observations = np.array(
                agent_observations, np.float32
            )  # NB! Q-agent expects float32 observation type

            assert (
                agent_observations.shape == self.observation_space(agent).shape
            ), "observation / observation space shape mismatch"

            return agent_observations

    @property
    def grass_patches(self):
        any_agent = self._last_infos[
            "agent_0"
        ]  # any agent is good here since we are using global coordinates here
        coordinates = any_agent[INFO_OBSERVATION_COORDINATES].get(FOOD_CHR, [])
        if len(coordinates) > 0:
            grass_patches = np.array(coordinates)
        else:
            grass_patches = np.zeros([0, 2])
        return grass_patches

    @property
    def water_holes(self):
        any_agent = self._last_infos[
            "agent_0"
        ]  # any agent is good here since we are using global coordinates here
        coordinates = any_agent[INFO_OBSERVATION_COORDINATES].get(DRINK_CHR, [])
        if len(coordinates) > 0:
            water_holes = np.array(coordinates)
        else:
            water_holes = np.zeros([0, 2])
        return water_holes

    def observe_from_location(
        self, agents_coordinates: Dict, agents_directions: Dict = None
    ):
        """This method is read-only (does not change the actual state of the environment nor the actual state of agents).
        Each given agent observes itself and the environment as if the agent was in the given location.
        """
        infos = super().observe_infos_from_location(
            agents_coordinates, agents_directions
        )
        # transform observations
        observations2 = {}
        for agent in infos.keys():
            observations2[agent] = self.transform_observation(agent, infos[agent])
        return observations2

    def calc_min_grass_distance(self, agent, info):
        agent_chr = self.agent_name_mapping[agent]
        agent_location = np.array(info[INFO_OBSERVATION_COORDINATES][agent_chr][0])
        grass_patches = np.array(info[INFO_OBSERVATION_COORDINATES][FOOD_CHR])
        min_grass_distance = distance_to_closest_item(agent_location, grass_patches)
        return min_grass_distance

    def reward_agent(self, min_grass_distance):
        # For now measure if agent can eat. Was #1 / (1 + min_grass_distance), moving to instincts
        if min_grass_distance > 1:
            return np.float64(0.0)
        else:
            return np.float64(1.0)

    def observation_space(self, agent):
        return self.transformed_observation_spaces[agent]

    # called by DQNLightning
    def state_to_namedtuple(self, state: npt.NDArray[ObservationFloat]) -> NamedTuple:
        """Method to convert a state array into a named tuple."""
        agent_coords = {
            "agent_coords": state[:2]
        }  # TODO: make it dependant on number of agents
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

    # This API is intended primarily as input for the neural network.
    # if observe_bitmap_layers == True then observe() method returns same value as observe_relative_bitmaps()
    # Relative observation bitmap is agent centric and considers the agent's observation radius. Environments with different sizes will have same-shaped relative observation bitmaps as long as the agent's observation radius is same.
    # if observe_bitmap_layers == False then the agent currently returns vector of coordinates compatible with the old Savanna agent implementation.
    def observe(self, agent=None) -> Union[Dict[AgentId, Observation], Observation]:
        if agent is None:
            return self.observations2
        else:
            return self.observations2[agent]

    # This API is intended primarily as input for the neural network. Relative observation bitmap is agent centric and considers the agent's observation radius. Environments with different sizes will have same-shaped relative observation bitmaps as long as the agent's observation radius is same.
    # if observe_bitmap_layers == True then observe() method returns same value as observe_relative_bitmaps()
    def observe_relative_bitmaps(
        self, agent=None
    ) -> Union[Dict[AgentId, Observation], Observation]:
        if agent is None:
            return {
                agent: self._last_infos[agent][INFO_AGENT_OBSERVATION_LAYERS_CUBE]
                for agent in self._last_infos.keys()
            }
        else:
            return self._last_infos[agent][INFO_AGENT_OBSERVATION_LAYERS_CUBE]

    # This API is intended primarily as alternate observation format input for the neural network. But please consider that absolute bitmaps are less flexible because different environments may have absolute bitmaps with different sizes. Also, absolute bitmaps are less convincing from the agent embodyment perspective.
    def observe_absolute_bitmaps(
        self, agent=None
    ) -> Union[Dict[AgentId, Observation], Observation]:
        if agent is None:
            return {
                agent: self._last_infos[agent][INFO_OBSERVATION_LAYERS_CUBE]
                for agent in self._last_infos.keys()
            }
        else:
            return self._last_infos[agent][INFO_OBSERVATION_LAYERS_CUBE]

    # This API might be useful as input to instincts. For instincts it has more convenient data format than bitmap.
    def observe_relative_coordinates(
        self, agent=None
    ) -> Union[Dict[AgentId, Observation], Observation]:
        if agent is None:
            return {
                agent: self._last_infos[agent][INFO_AGENT_OBSERVATION_COORDINATES]
                for agent in self._last_infos.keys()
            }
        else:
            return self._last_infos[agent][INFO_AGENT_OBSERVATION_COORDINATES]

    # This API might be useful as input to instincts. For instincts it has more convenient data format than bitmap.
    def observe_absolute_coordinates(
        self, agent=None
    ) -> Union[Dict[AgentId, Observation], Observation]:
        if agent is None:
            return {
                agent: self._last_infos[agent][INFO_OBSERVATION_COORDINATES]
                for agent in self._last_infos.keys()
            }
        else:
            return self._last_infos[agent][INFO_OBSERVATION_COORDINATES]

    # @functools.lru_cache(maxsize=None)
    # def observation_space(self, agent: str):
    #    return self._observation_spaces[agent]

    # @functools.lru_cache(maxsize=None)
    # def action_space(self, agent: str):
    #    return self._action_spaces[agent]


class SavannaGridworldParallelEnv(GridworldZooBaseEnv, GridworldZooParallelEnv):
    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}
        GridworldZooBaseEnv.__init__(self, env_params)
        GridworldZooParallelEnv.__init__(self, **self.super_initargs)
        self.init_observation_spaces()
        self._last_infos = {}
        self.observations2 = {}

    # def observe_from_location(self, agents_coordinates: Dict):
    #    """This method is read-only (does not change the actual state of the environment nor the actual state of agents).
    #    Each given agent observes the environment as well as itself as if it was in the given location.
    #    The return values format is similar to reset() method."""
    #    observations, infos = super().observe_from_location(agents_coordinates)
    #    # self._last_infos = infos
    #    # transform observations
    #    observations2 = {}
    #    for agent in infos.keys():
    #        observations2[agent] = self.transform_observation(agent, infos[agent])
    #    return observations2

    def reset(
        self, seed: Optional[int] = None, options=None, *args, **kwargs
    ) -> Tuple[Dict[AgentId, Observation], Dict[AgentId, Info]]:
        observations, infos = GridworldZooParallelEnv.reset(
            self, seed=seed, options=options, *args, **kwargs
        )
        self._last_infos = infos
        # transform observations
        for agent in infos.keys():
            self.observations2[agent] = self.transform_observation(agent, infos[agent])

        if self._override_infos:
            infos = {agent: {} for agent in infos.keys()}

        return self.observations2, infos

    def step(self, actions: Dict[str, Action]) -> Step:
        """step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - truncateds
        - info
        dicts where each dict looks like {agent_1: action_of_agent_1, agent_2: action_of_agent_2}
        or generally {<agent_name>: <agent_action or None if agent is done>}
        """
        logger.debug("debug actions", actions)
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            return {}, {}, {}, {}, {}

        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = GridworldZooParallelEnv.step(
            self,
            OrderedDict({agent: {"step": action} for agent, action in actions.items()}),
        )
        self._last_infos = infos

        rewards2 = {}

        # transform observations and rewards
        for agent in infos.keys():
            self.observations2[agent] = self.transform_observation(agent, infos[agent])

            min_grass_distance = self.calc_min_grass_distance(agent, infos[agent])
            rewards2[agent] = self.reward_agent(min_grass_distance)

        if self._override_infos:
            infos = {agent: {} for agent in infos.keys()}

        logger.debug(
            "debug return", self.observations2, rewards, terminateds, truncateds, infos
        )
        return self.observations2, rewards2, terminateds, truncateds, infos


class SavannaGridworldSequentialEnv(GridworldZooBaseEnv, GridworldZooAecEnv):
    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}
        self.observe_immediately_after_agent_action = env_params.get(
            "observe_immediately_after_agent_action", False
        )  # TODO: configure

        GridworldZooBaseEnv.__init__(self, env_params)
        GridworldZooAecEnv.__init__(self, **self.super_initargs)
        self.init_observation_spaces()
        self._last_infos = {}
        self._last_rewards2 = {
            agent: None for agent in self.possible_agents
        }  # Rewards should be only updated after step, not on each observation before step. Rewards should be initialised to None before any reset() is called so that .rewards property returns dictionary with existing agents.
        self._cumulative_rewards2 = {agent: 0.0 for agent in self.possible_agents}
        self.observations2 = {}

    @property
    def rewards(self):  # Needed for tests
        # NB! rewards should be only updated after step, not on each observation before step
        return {
            agent: reward
            for agent, reward in self._last_rewards2.items()
            if not self.terminations[agent] and not self.truncations[agent]
        }  # terminated agents are not allowed in .rewards, but we still need to store them in self._last_rewards2 so that the reward can be read via .last() method

    @property
    def infos(
        self,
    ):  # Needed for tests. Zoo is unable to compare infos unless they have simple structure.
        infos = GridworldZooAecEnv.infos.fget(
            self
        )  # property needs .fget() to become callable

        if self._override_infos:
            return {agent: {} for agent in infos.keys()}
        else:
            return infos

    # def observe_from_location(self, agents_coordinates: Dict):
    #    """This method is read-only (does not change the actual state of the environment nor the actual state of agents).
    #    Each given agent observes the environment as well as itself as if it was in the given location.
    #    The return values format is similar to reset() method."""

    #    # observe observations, transform observations
    #    observations2 = {}
    #    infos = {}
    #    for agent, coordinate in agents_coordinates.items():
    #        info = self.observe_from_location(agent, coordinate)
    #        infos[agent] = info
    #        observations2[agent] = self.transform_observation(agent, info)

    #    # self._last_infos = infos
    #    return observations2

    def reset(
        self, seed: Optional[int] = None, options=None, *args, **kwargs
    ) -> Tuple[Dict[AgentId, Observation], Dict[AgentId, Info]]:
        GridworldZooAecEnv.reset(self, seed=seed, options=options, *args, **kwargs)

        # observe observations, transform observations
        infos = {}
        for agent in self.possible_agents:
            info = self.observe_info(agent)
            infos[agent] = info
            self.observations2[agent] = self.transform_observation(agent, info)

        self._last_infos = infos
        self._last_rewards2 = {
            agent: np.float64(0) for agent in self.possible_agents
        }  # Rewards should be initialised to 0.0 before any step is taken so that .rewards property returns dictionary with existing agents. NB! not calculating actual rewards yet, rewards should be only updated after step, not on each observation before step.

        if self._override_infos:
            infos = {agent: {} for agent in infos.keys()}

        return self.observations2, infos

    def last(self, observe=True):
        """Returns observation, cumulative reward, terminated, truncated, info for the current agent (specified by self.agent_selection).

        If observe flag is True then current board state is observed. If observe flag is False then the observation that was made after current agent's latest move is returned.
        """

        observation, reward, terminated, truncated, info = GridworldZooAecEnv.last(
            self, observe=observe
        )

        agent = self.agent_selection

        if observe:
            self._last_infos[
                agent
            ] = info  # TODO: is this correct to update infos only when observe=True?
            observation2 = self.transform_observation(agent, info)
            self.observations2[agent] = observation2
        else:
            observation2 = None  # that's how Zoo api_test.py requires it

        reward2 = self._cumulative_rewards2[
            agent
        ]  # rewards should be only updated after step, not on each observation before step

        if self._override_infos:
            info = {}

        return observation2, reward2, terminated, truncated, info

    def step(self, action: Action) -> None:
        logger.debug("debug action", action)

        agent = self.agent_selection

        self._cumulative_rewards2[agent] = 0

        GridworldZooAecEnv.step(self, {"step": action})

        info = self.observe_info(agent)
        min_grass_distance = self.calc_min_grass_distance(agent, info)
        reward2 = self.reward_agent(min_grass_distance)
        self._last_rewards2[agent] = reward2
        self._cumulative_rewards2[agent] += reward2

    def step_single_agent(self, action: Action):
        """step(action) takes in an action for each agent and should return the
        - observation
        - reward
        - done
        - truncated
        - info
        """
        logger.debug("debug action", action)

        agent = self.agent_selection
        self._cumulative_rewards2[agent] = 0
        if action is None:
            action = (
                Actions.NOOP
            )  # all agents need to take a step, the stepping order cannot be modified
        GridworldZooAecEnv.step(self, {"step": action})

        # observe observations, transform observations and rewards
        info = self.observe_info(agent)
        # self.observe_from_location({agent: [1, 1]})    # for debugging
        self._last_infos[agent] = info
        observation2 = self.transform_observation(agent, info)
        self.observations2[agent] = observation2

        min_grass_distance = self.calc_min_grass_distance(agent, info)
        reward2 = self.reward_agent(min_grass_distance)
        self._last_rewards2[agent] = reward2
        self._cumulative_rewards2[agent] += reward2

        terminated = self.terminations[agent]
        truncated = self.truncations[agent]

        if self._override_infos:
            info = {}

        logger.debug("debug return", observation2, reward2, terminated, truncated, info)
        return observation2, reward2, terminated, truncated, info

    def step_multiple_agents(self, actions: Dict[str, Action]) -> Step:
        """step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - truncateds
        - info
        dicts where each dict looks like {agent_1: action_of_agent_1, agent_2: action_of_agent_2}
        or generally {<agent_name>: <agent_action or None if agent is done>}
        """
        logger.debug("debug actions", actions)
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            return {}, {}, {}, {}, {}

        alive_agents = []
        rewards2 = {}
        infos = {}

        # loop over all agents in ENV NOT IN ACTIONS DICT
        for index in range(
            0, self.num_agents
        ):  # do one iteration over all ALIVE agents
            agent = self.agent_selection  # this returns only alive agents
            alive_agents.append(agent)
            action = actions.get(agent, None)
            if action is None:
                action = (
                    Actions.NOOP
                )  # all agents need to take a step, the stepping order cannot be modified
            GridworldZooAecEnv.step(self, {"step": action})

            if (
                self.observe_immediately_after_agent_action
            ):  # observe BEFORE next agent takes its step?
                # observe observations, transform observations and rewards
                info = self.observe_info(agent)
                # self.observe_from_location({agent: [1, 1]})    # for debugging
                infos[agent] = info
                self._last_infos[agent] = info
                self.observations2[agent] = self.transform_observation(agent, info)

                min_grass_distance = self.calc_min_grass_distance(agent, info)
                reward2 = self.reward_agent(min_grass_distance)
                rewards2[agent] = reward2
                self._last_rewards2[agent] = reward2

        # / for index in range(0, self.num_agents)

        if (
            not self.observe_immediately_after_agent_action
        ):  # observe only after ALL agents are done stepping?
            # observe observations, transform observations and rewards
            for agent in alive_agents:
                info = self.observe_info(agent)
                # self.observe_from_location({agent: [1, 1]})    # for debugging
                infos[agent] = info
                self._last_infos[agent] = info
                self.observations2[agent] = self.transform_observation(agent, info)

                min_grass_distance = self.calc_min_grass_distance(agent, info)
                reward2 = self.reward_agent(min_grass_distance)
                rewards2[agent] = reward2
                self._last_rewards2[agent] = reward2

        terminateds = self.terminations
        truncateds = self.truncations

        if self._override_infos:
            infos = {agent: {} for agent in infos.keys()}

        logger.debug(
            "debug return", self.observations2, rewards2, terminateds, truncateds, infos
        )
        return self.observations2, rewards2, terminateds, truncateds, infos
