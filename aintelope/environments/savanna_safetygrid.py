import sys
import logging
from collections import OrderedDict, namedtuple
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

import gymnasium.spaces  # cannot import gymnasium.spaces.Tuple directly since it is already used by typing
from aintelope.environments.ai_safety_gridworlds.aintelope_savanna import (  # TODO: import agent char map from env object instead?; AGENT_CHR3,
    AGENT_CHR1,
    AGENT_CHR2,
    DRINK_CHR,
    FOOD_CHR,
    DRINK_CHR,
    GOLD_CHR,
    SILVER_CHR,
    DANGER_TILE_CHR,
    PREDATOR_NPC_CHR,
    WALL_CHR,
    GAME_ART,
)

from ai_safety_gridworlds.helpers.gridworld_zoo_aec_env import GridworldZooAecEnv
from ai_safety_gridworlds.helpers.gridworld_zoo_parallel_env import (
    INFO_AGENT_OBSERVATION_COORDINATES,
    INFO_AGENT_OBSERVATION_LAYERS_CUBE,
    INFO_AGENT_OBSERVATION_LAYERS_DICT,
    INFO_AGENT_OBSERVATION_LAYERS_ORDER,
    INFO_AGENT_OBSERVATIONS,
    INFO_OBSERVATION_COORDINATES,
    INFO_OBSERVATION_LAYERS_CUBE,
    INFO_OBSERVATION_LAYERS_DICT,
    INFO_OBSERVATION_LAYERS_ORDER,
    INFO_REWARD_DICT,  # TODO: rename to INFO_SCORE_DICT
    INFO_CUMULATIVE_REWARD_DICT,  # TODO: rename to INFO_CUMULATIVE_SCORE_DICT
    Actions,
    GridworldZooParallelEnv,
)

from aintelope.aintelope_typing import Reward  # TODO: use np.ndarray or mo_reward
from aintelope.aintelope_typing import (
    AgentId,
    Info,
    Observation,
    ObservationFloat,
)
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv, ParallelEnv

INFO_AGENT_INTEROCEPTION_ORDER = "info_agent_interoception_order"
INFO_AGENT_INTEROCEPTION_VECTOR = "info_agent_interoception_vector"
INTEROCEPTION_FOOD = "food_satiation"
INTEROCEPTION_DRINK = "drink_satiation"
ACTION_RELATIVE_COORDINATE_MAP = (
    "action_relative_coordinate_map"  # TODO: move to Gridworld environment
)
ALL_AGENTS_LAYER = "all_agents"

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
        "level": 0,  # selects the map template
        # This seed is used mainly for environment map randomisation.
        # Later the test calls .seed() method on the wrapper and this will determine
        # the random action sampling and other random events during the game play.
        "seed": None,
        # "name": "savanna-safetygrid-v1",
        # "render_fps": 3,
        "render_agent_radius": 10,
        # "render_modes": ("human", "ascii", "offline"),
        # "map_min": 0,
        # "map_max": 10,   # TODO
        #
        "amount_agents": 1,
        "amount_grass_patches": 2,
        "amount_water_holes": 0,
        "amount_danger_tiles": 0,
        "amount_predators": 0,
        "enable_homeostasis": False,
        "sustainability_challenge": False,
        "amount_gold_deposits": 0,
        "amount_silver_deposits": 0,
        #
        "num_iters": 1,
        # 0 - fixed, 1 - relative, depending on last move, 2 - relative,
        # controlled by separate turning actions.
        "observation_direction_mode": 1,
        # 0 - fixed, 1 - relative, depending on last move, 2 - relative,
        # controlled by separate turning actions.
        "action_direction_mode": 1,
        # 0 - off (do not use this setting), 1 - once per experiment run, 2 - once per trial
        # (a trial is a sequence of training episodes separated by env.reset call,
        # but using a same model instance), 3 - once per training episode.
        "map_randomization_frequency": 1,
        # Whether to remove tile types not present on initial map from observation
        # layers. - set to False when same agent brain is trained over multiple
        # environments
        "remove_unused_tile_types_from_layers": False,
        # Alternate observation format to current vector of absolute coordinates.
        # Bitmap representation enables representing objects which might be outside of
        # agent's observation zone for time being.
        # Needed for tests. Zoo is unable to compare infos
        # unless they have simple structure.
        "override_infos": False,
        "test_death": False,
        "test_death_probability": 0.33,
        "scalarize_rewards": False,  # needs to be set True for Zoo sequential API unit tests and for OpenAI baselines learning
        "flatten_observations": False,  # this will not work with current code
        "combine_interoception_and_vision": False,  # needs to be set to True for OpenAI baselines learning algorithms
    }

    def __init__(self, env_params: Optional[Dict] = None, **kwargs):
        if env_params is None:
            env_params = {}
        env_params = dict(env_params)  # NB! make a copy before updating with kwargs
        env_params.update(kwargs)

        self.render_mode = None  # Some libraries require this field to be present. The actual value seems to be unimportant.

        # NB! Need to clone in order to not modify the default dict.
        # Similar problem to mutable default arguments.
        self.metadata = dict(self.metadata)
        scores = env_params.get("scores", None)
        self.metadata.update(env_params)
        if scores is not None:  # tests do not have scores config
            self.metadata.update(
                scores
            )  # move scores to same metadata level with other parameters
        logger.info(f"initializing savanna env with params: {self.metadata}")

        # TODO: get rid of this override and just ignore truncation flag from the environment?
        if self.metadata["num_iters"] > 50:  # some unit tests use num_iters <= 50
            self.metadata[
                "num_iters"
            ] = (
                sys.maxsize
            )  # allow learning from last step when the agent does not die on its own

        metadata_to_super_initargs_dict = {
            "level": "level",
            "map_width": "map_width",
            "map_height": "map_height",
            # This seed is used mainly for environment map randomisation.
            # Later the test calls .seed() method on the wrapper and this will
            # determine the random action sampling and other random events
            # during the game play.
            "seed": "seed",
            "max_iterations": "num_iters",
            #
            "amount_agents": "amount_agents",
            "amount_food_patches": "amount_grass_patches",
            "amount_drink_holes": "amount_water_holes",
            "amount_water_tiles": "amount_danger_tiles",
            "amount_predators": "amount_predators",
            "penalise_oversatiation": "enable_homeostasis",
            "sustainability_challenge": "sustainability_challenge",
            "amount_gold_deposits": "amount_gold_deposits",
            "amount_silver_deposits": "amount_silver_deposits",
            #
            "DANGER_TILE_SCORE": "DANGER_TILE_SCORE",
            "PREDATOR_NPC_SCORE": "PREDATOR_NPC_SCORE",
            "PREDATOR_MOVEMENT_PROBABILITY": "PREDATOR_MOVEMENT_PROBABILITY",
            "MOVEMENT_SCORE": "MOVEMENT_SCORE",
            "COOPERATION_SCORE": "COOPERATION_SCORE",
            #
            "GOLD_VISITS_LOG_BASE": "GOLD_VISITS_LOG_BASE",
            "SILVER_VISITS_LOG_BASE": "SILVER_VISITS_LOG_BASE",
            #
            "FOOD_SCORE": "FOOD_SCORE",
            "FOOD_DEFICIENCY_SCORE": "FOOD_DEFICIENCY_SCORE",
            "FOOD_OVERSATIATION_SCORE": "FOOD_OVERSATIATION_SCORE",
            #
            "FOOD_DEFICIENCY_INITIAL": "FOOD_DEFICIENCY_INITIAL",
            "FOOD_EXTRACTION_RATE": "FOOD_EXTRACTION_RATE",
            "FOOD_DEFICIENCY_RATE": "FOOD_DEFICIENCY_RATE",
            "FOOD_OVERSATIATION_LIMIT": "FOOD_OVERSATIATION_LIMIT",
            "FOOD_OVERSATIATION_THRESHOLD": "FOOD_OVERSATIATION_THRESHOLD",
            "FOOD_DEFICIENCY_THRESHOLD": "FOOD_DEFICIENCY_THRESHOLD",
            #
            "FOOD_GROWTH_LIMIT": "FOOD_GROWTH_LIMIT",
            "FOOD_REGROWTH_EXPONENT": "FOOD_REGROWTH_EXPONENT",
            #
            "DRINK_SCORE": "DRINK_SCORE",
            "DRINK_DEFICIENCY_SCORE": "DRINK_DEFICIENCY_SCORE",
            "DRINK_OVERSATIATION_SCORE": "DRINK_OVERSATIATION_SCORE",
            #
            "DRINK_DEFICIENCY_INITIAL": "DRINK_DEFICIENCY_INITIAL",
            "DRINK_EXTRACTION_RATE": "DRINK_EXTRACTION_RATE",
            "DRINK_DEFICIENCY_RATE": "DRINK_DEFICIENCY_RATE",
            "DRINK_OVERSATIATION_LIMIT": "DRINK_OVERSATIATION_LIMIT",
            "DRINK_OVERSATIATION_THRESHOLD": "DRINK_OVERSATIATION_THRESHOLD",
            "DRINK_DEFICIENCY_THRESHOLD": "DRINK_DEFICIENCY_THRESHOLD",
            #
            "DRINK_GROWTH_LIMIT": "DRINK_GROWTH_LIMIT",
            "DRINK_REGROWTH_EXPONENT": "DRINK_REGROWTH_EXPONENT",
            #
            # TODO: is render_agent_radius meant as diameter actually?
            "observation_radius": "render_agent_radius",
            # 0 - fixed, 1 - relative, depending on last move, 2 - relative,
            # controlled by separate turning actions.
            "observation_direction_mode": "observation_direction_mode",
            # 0 - fixed, 1 - relative, depending on last move, 2 - relative,
            # controlled by separate turning actions.
            "action_direction_mode": "action_direction_mode",
            # 0 - off (do not use this setting), 1 - once per experiment run, 2 - once per trial (a trial is a
            # sequence of training episodes separated by env.reset call,
            # but using a same model instance), 3 - once per training episode.
            "map_randomization_frequency": "map_randomization_frequency",
            # Whether to remove tile types not present on initial map from observation
            # layers. - set to False when same agent brain is trained over multiple
            # environments
            "remove_unused_tile_types_from_layers": "remove_unused_tile_types_from_layers",
            "test_death": "test_death",
            "test_death_probability": "test_death_probability",
            "scalarize_rewards": "scalarize_rewards",
            "amount_agents": "amount_agents",
            "flatten_observations": "flatten_observations",
            # "scalarise": "scalarize_rewards",     # NB! not passing scalarise/scalarize_rewards to the environment. Instead, if needed, we do our own scalarization in this wrapper here.
            "max_iterations": "max_iterations",
        }

        self.super_initargs = {"env_name": "ai_safety_gridworlds.aintelope_savanna"}

        for super_initargs_key, metadata_key in metadata_to_super_initargs_dict.items():
            if self.metadata.get(metadata_key, None) is not None:
                self.super_initargs[super_initargs_key] = self.metadata[metadata_key]

        self._override_infos = self.metadata["override_infos"]
        self._scalarize_rewards = self.metadata["scalarize_rewards"]
        self._combine_interoception_and_vision = self.metadata[
            "combine_interoception_and_vision"
        ]

    def init_observation_spaces(self, parent_observation_spaces, infos):
        # for @zoo-api
        if self._combine_interoception_and_vision:
            self.transformed_observation_spaces = {
                agent: Box(
                    low=0,  # this is a boolean bitmap
                    high=1,  # this is a boolean bitmap
                    shape=(
                        len(infos[agent][INFO_AGENT_OBSERVATION_LAYERS_ORDER])
                        + 2,  # this already includes all_agents layer + 2 for interoception
                        parent_observation_spaces[agent].shape[1],
                        parent_observation_spaces[agent].shape[2],
                    ),
                )
                for agent in self.possible_agents
            }
        else:
            self.transformed_observation_spaces = {
                agent: gymnasium.spaces.Tuple(
                    [
                        Box(
                            low=0,  # this is a boolean bitmap
                            high=1,  # this is a boolean bitmap
                            shape=(
                                len(
                                    infos[agent][INFO_AGENT_OBSERVATION_LAYERS_ORDER]
                                ),  # this already includes all_agents layer,
                                parent_observation_spaces[agent].shape[1],
                                parent_observation_spaces[agent].shape[2],
                            ),
                        ),
                        Box(
                            low=-np.inf, high=np.inf, shape=(2,)
                        ),  # interoception vector
                    ]
                )
                for agent in self.possible_agents
            }

        qqq = True  # for debugging

    # this method has no side effects
    def transform_observation(
        self, agent: str, info: dict
    ) -> npt.NDArray[ObservationFloat]:
        if agent is None:
            return info[INFO_OBSERVATION_LAYERS_CUBE].astype(np.float32)
        else:  # the info is already agent-specific, so no need to find agent subkey here
            # agent_interoception_vector = [
            #    info["metrics_dict"]["FoodSatiation_" + self.agent_name_mapping[agent]],
            #    info["metrics_dict"]["DrinkSatiation_" + self.agent_name_mapping[agent]],
            # ]
            observation = info[INFO_AGENT_OBSERVATION_LAYERS_CUBE]

            all_agents_layer = np.zeros(
                [observation.shape[1], observation.shape[2]], bool
            )
            for agent_name, agent_chr in self.agent_name_mapping.items():
                all_agents_layer |= info[INFO_AGENT_OBSERVATION_LAYERS_DICT][agent_chr]

            if self._combine_interoception_and_vision:
                # TODO: Config for interoception scaling? Or use sigmoid transformation?
                # NB! use +0.5 so that interoception value of 0 is centered between min and max of 0 and 1.
                interoception_values = (
                    info[INFO_AGENT_INTEROCEPTION_VECTOR].astype(np.float32) / 1000
                    + 0.5
                )

                # Add two more layers to the vision observation, representing interoception measures. For both interoception measures, entire layer will have same value.
                interoception_layers = np.expand_dims(
                    np.ones([observation.shape[1], observation.shape[2]], np.float32),
                    axis=0,
                ) * np.expand_dims(interoception_values, axis=[1, 2])

                observation = np.vstack(
                    [
                        observation,
                        np.expand_dims(all_agents_layer, axis=0),
                        interoception_layers,
                    ]
                )  # feature vector is the first dimension

                return observation
            else:
                observation = np.vstack(
                    [observation, np.expand_dims(all_agents_layer, axis=0)]
                )  # feature vector is the first dimension

                return (
                    observation.astype(np.float32),
                    # np.array(agent_interoception_vector).astype(np.float32)
                    info[INFO_AGENT_INTEROCEPTION_VECTOR].astype(np.float32),
                )

    def format_info(self, agent: str, info: dict):
        # keep only necessary fields of infos

        agent_interoception_vector = np.array(
            [
                info["metrics_dict"].get(
                    "FoodSatiation_" + self.agent_name_mapping[agent], 0
                ),
                info["metrics_dict"].get(
                    "DrinkSatiation_" + self.agent_name_mapping[agent], 0
                ),
            ]
        )

        all_agents_coordinates = []
        for agent_name, agent_chr in self.agent_name_mapping.items():
            all_agents_coordinates += info[INFO_AGENT_OBSERVATION_COORDINATES][
                agent_chr
            ]

        coordinates_dict = dict(info[INFO_AGENT_OBSERVATION_COORDINATES])
        coordinates_dict[ALL_AGENTS_LAYER] = all_agents_coordinates

        return {
            INFO_AGENT_OBSERVATION_COORDINATES: coordinates_dict,
            INFO_AGENT_OBSERVATION_LAYERS_ORDER: info[
                INFO_AGENT_OBSERVATION_LAYERS_ORDER
            ]
            + [ALL_AGENTS_LAYER],
            INFO_AGENT_OBSERVATION_LAYERS_CUBE: info[
                INFO_AGENT_OBSERVATION_LAYERS_CUBE
            ],
            INFO_AGENT_OBSERVATION_LAYERS_DICT: info[
                INFO_AGENT_OBSERVATION_LAYERS_DICT
            ],
            INFO_AGENT_INTEROCEPTION_ORDER: [INTEROCEPTION_FOOD, INTEROCEPTION_DRINK],
            INFO_AGENT_INTEROCEPTION_VECTOR: agent_interoception_vector,
            INFO_REWARD_DICT: info[INFO_REWARD_DICT],
            INFO_CUMULATIVE_REWARD_DICT: info[INFO_CUMULATIVE_REWARD_DICT],
            # TODO: in case of relative direction, this mapping will change depending on agent and depending on its direction
            ACTION_RELATIVE_COORDINATE_MAP: {  # TODO: read these constants from the environment?
                Actions.NOOP: (0, 0),
                Actions.LEFT: (-1, 0),
                Actions.RIGHT: (1, 0),
                Actions.UP: (0, -1),
                Actions.DOWN: (0, 1),
            },
        }

    def format_infos(self, infos: dict):
        # keep only necessary fields of infos
        return {
            agent: self.format_info(agent, agent_info)
            for agent, agent_info in infos.items()
        }

    def filter_info(self, agent: str, info: dict):
        # keep only necessary keys of infos

        allowed_keys = [
            INFO_AGENT_OBSERVATION_COORDINATES,
            INFO_AGENT_OBSERVATION_LAYERS_ORDER,
            INFO_AGENT_INTEROCEPTION_VECTOR,  # keeping interoception available in info since in observation it may be either located in its own vector or be part of the vision. That make access to this data cumbersome when writing hardcoded rules. Accessing via info argument is more convenient in such cases.
            INFO_AGENT_INTEROCEPTION_ORDER,
            ACTION_RELATIVE_COORDINATE_MAP,
        ]
        result = {key: value for key, value in info.items() if key in allowed_keys}

        if INFO_AGENT_INTEROCEPTION_VECTOR in result:
            result[INFO_AGENT_INTEROCEPTION_VECTOR] = result[
                INFO_AGENT_INTEROCEPTION_VECTOR
            ].tolist()  # NB! need to convert it to list since Zoo unit tests are not able to compare numpy arrays inside info dict

        return result

    def filter_infos(self, infos: dict):
        # keep only necessary fields of infos
        return {
            agent: self.filter_info(agent, agent_info)
            for agent, agent_info in infos.items()
        }

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
        """This method is read-only (does not change the actual state of the
        environment nor the actual state of agents). Each given agent observes itself
        and the environment as if the agent was in the given location.
        """
        infos = super().observe_infos_from_location(
            agents_coordinates, agents_directions
        )
        # transform observations
        observations2 = {}
        for agent in infos.keys():
            observations2[agent] = self.transform_observation(agent, infos[agent])
        return observations2

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

    """
    This API is intended primarily as input for the neural network.
    Currently observe() method returns same value as observe_relative_bitmaps() though it 
    might depend on configuration in future implementations (this has been the case in the
    past with some currently removed implementations).
    Relative observation bitmap is agent centric and considers the agent's observation
    radius. Environments with different sizes will have same-shaped relative
    observation bitmaps as long as the agent's observation radius is same.
    """

    def observe(self, agent=None) -> Union[Dict[AgentId, Observation], Observation]:
        if agent is None:
            return self.observations2
        else:
            return self.observations2[agent]

    # This API is intended primarily as input for the instincts, documenting the order of layers in observation cube
    def relative_observation_layers_order(
        self, agent=None
    ) -> Union[Dict[AgentId, Observation], Observation]:
        if agent is None:
            return {
                agent: self._last_infos[agent][INFO_AGENT_OBSERVATION_LAYERS_ORDER]
                for agent in self._last_infos.keys()
            }
        else:
            return self._last_infos[agent][INFO_AGENT_OBSERVATION_LAYERS_ORDER]

    # This API is intended primarily as input for the instincts, documenting the order of layers in observation cube
    def absolute_observation_layers_order(
        self, agent=None
    ) -> Union[Dict[AgentId, Observation], Observation]:
        if agent is None:
            return {
                agent: self._last_infos[agent][INFO_OBSERVATION_LAYERS_ORDER]
                for agent in self._last_infos.keys()
            }
        else:
            return self._last_infos[agent][INFO_OBSERVATION_LAYERS_ORDER]

    """
    This API is intended primarily as input for the neural network. Relative
    observation bitmap is agent centric and considers the agent's observation
    radius. Environments with different sizes will have same-shaped relative
    observation bitmaps as long as the agent's observation radius is same.
    Currently, observe() method returns same value as observe_relative_bitmaps() though it 
    might depend on configuration in future implementations (this has been the case in the
    past with some currently removed implementations).
    """

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

    """
    This API is intended primarily as alternate observation format input for the
    neural network. But please consider that absolute bitmaps are less flexible
    because different environments may have absolute bitmaps with different sizes.
    Also, absolute bitmaps are less convincing from the agent embodyment perspective.
    """

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

    # This API might be useful as input to instincts.
    # For instincts it has more convenient data format than bitmap.
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

    # This API might be useful as input to instincts.
    # For instincts it has more convenient data format than bitmap.
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

    # def observation_space(self, agent: str):
    #    return self._observation_spaces[agent]

    # def action_space(self, agent: str):
    #    return self._action_spaces[agent]


class SavannaGridworldParallelEnv(GridworldZooBaseEnv, GridworldZooParallelEnv):
    def __init__(self, env_params: Optional[Dict] = None, **kwargs):
        if env_params is None:
            env_params = {}
        GridworldZooBaseEnv.__init__(self, env_params, **kwargs)
        GridworldZooParallelEnv.__init__(self, **self.super_initargs)
        parent_observation_spaces = GridworldZooParallelEnv.observation_spaces.fget(
            self
        )

        _, infos = GridworldZooParallelEnv.reset(
            self
        )  # need to reset in order to access infos
        infos = self.format_infos(infos)
        self.init_observation_spaces(parent_observation_spaces, infos)

        self._last_infos = {}
        self.observations2 = {}

    # def observe_from_location(self, agents_coordinates: Dict):
    #    """This method is read-only. It does not change the actual state of the
    #    environment nor the actual state of agents).
    #    Each given agent observes the environment as well as itself as if it was in
    #    the given location.
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
        infos = self.format_infos(infos)
        self._last_infos = infos
        # transform observations
        for agent in infos.keys():
            self.observations2[agent] = self.transform_observation(agent, infos[agent])

        if self._override_infos:
            infos = {agent: {} for agent in infos.keys()}

        return self.observations2, self.filter_infos(infos)

    def step(self, actions: Dict[str, Action]) -> Step:
        """step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - truncateds
        - info
        dicts where each dict looks like:
            {agent_1: action_of_agent_1, agent_2: action_of_agent_2}
        or generally:
            {<agent_name>: <agent_action or None if agent is done>}
        """
        logger.debug("debug actions", actions)
        # If a user passes in actions with no agents,
        # then just return empty observations, etc.
        if not actions:
            return {}, {}, {}, {}, {}

        (
            observations,
            rewards,  # TODO: rename to scores
            terminateds,
            truncateds,
            infos,
        ) = GridworldZooParallelEnv.step(
            self,
            OrderedDict({agent: {"step": action} for agent, action in actions.items()}),
        )
        infos = self.format_infos(infos)
        self._last_infos = infos

        rewards2 = {}
        # transform observations and rewards
        for agent in list(infos.keys()):
            rewards2[agent] = infos[agent][INFO_REWARD_DICT]
            if self._scalarize_rewards:
                rewards2[agent] = sum(
                    rewards2[agent].values()
                )  # this is currently used for unit tests and OpenAI baselines so no need for nonlinear utility transformations before summation

        for agent in list(
            self.observations2.keys()
        ):  # previously terminated or truncated agents are not in infos or observations
            if agent in observations:
                self.observations2[agent] = self.transform_observation(
                    agent, infos[agent]
                )
            else:  # dead agent, needs to be removed from observations2
                del self.observations2[agent]

        if self._override_infos:
            infos = {agent: {} for agent in infos.keys()}

        logger.debug(
            "debug return",
            self.observations2,
            rewards,
            terminateds,
            truncateds,
            self.filter_infos(infos),
        )
        return (
            self.observations2,
            rewards2,
            terminateds,
            truncateds,
            self.filter_infos(infos),
        )


class SavannaGridworldSequentialEnv(GridworldZooBaseEnv, GridworldZooAecEnv):
    def __init__(self, env_params: Optional[Dict] = None, **kwargs):
        if env_params is None:
            env_params = {}
        self.observe_immediately_after_agent_action = env_params.get(
            "observe_immediately_after_agent_action", False
        )  # TODO: configure

        GridworldZooBaseEnv.__init__(self, env_params, **kwargs)
        GridworldZooAecEnv.__init__(self, **self.super_initargs)
        parent_observation_spaces = GridworldZooAecEnv.observation_spaces.fget(self)

        GridworldZooAecEnv.reset(self)  # need to reset in order to access infos
        infos = GridworldZooAecEnv.infos.fget(
            self
        )  # property needs .fget() to become callable
        infos = self.format_infos(infos)
        self.init_observation_spaces(parent_observation_spaces, infos)

        self._last_infos = {}
        # Rewards should be only updated after step, not on each observation before
        # step. Rewards should be initialised to None before any reset() is called so
        # that .rewards property returns dictionary with existing agents.
        self._last_rewards2 = {agent: None for agent in self.possible_agents}
        self._cumulative_rewards2 = {
            agent: (0.0 if self._scalarize_rewards else {})
            for agent in self.possible_agents
        }
        self.observations2 = {}

    @property
    def rewards(self):  # Needed for tests
        """
        NB! rewards should be only updated after step,
        not on each observation before step

        terminated agents are not allowed in .rewards, but we still need to store them
        in self._last_rewards2 so that the reward can be read via .last() method
        """
        return {
            agent: reward
            for agent, reward in self._last_rewards2.items()
            if not self.terminations[agent] and not self.truncations[agent]
        }

    @property
    def infos(
        self,
    ):
        """Needed for tests.
        Zoo is unable to compare infos unless they have simple structure.
        """
        infos = GridworldZooAecEnv.infos.fget(
            self
        )  # property needs .fget() to become callable
        infos = self.format_infos(infos)

        if self._override_infos:
            return {agent: {} for agent in infos.keys()}
        else:
            return self.filter_infos(infos)

    def observe_info(self, agent):
        info = GridworldZooAecEnv.observe_info(self, agent)
        info = self.format_info(agent, info)
        return self.filter_info(agent, info)

    # def observe_from_location(self, agents_coordinates: Dict):
    #    """This method is read-only. It does not change the actual state of the
    #    environment nor the actual state of agents).
    #    Each given agent observes the environment as well as itself
    #    as if it was in the given location.
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
            info = GridworldZooAecEnv.observe_info(self, agent)
            info = self.format_info(agent, info)
            infos[agent] = info
            self.observations2[agent] = self.transform_observation(agent, info)

        self._last_infos = infos
        # Rewards should be initialised to 0.0 before any step is taken
        # so that .rewards property returns dictionary with existing agents.
        # NB! not calculating actual rewards yet, rewards should be only updated after
        # step, not on each observation before step.
        if self._scalarize_rewards:
            self._last_rewards2 = {
                agent: np.float64(0) for agent in self.possible_agents
            }
        else:
            self._last_rewards2 = {agent: {} for agent in self.possible_agents}

        if self._override_infos:
            infos = {agent: {} for agent in infos.keys()}

        return self.observations2, self.filter_infos(infos)

    def last(self, observe=True):
        """Returns observation, cumulative reward, terminated, truncated, info for the
        current agent (specified by self.agent_selection).

        If observe flag is True then current board state is observed.
        If observe flag is False then the observation that was made
        after current agent's latest move is returned.
        """

        observation, reward, terminated, truncated, info = GridworldZooAecEnv.last(
            self, observe=observe
        )
        agent = self.agent_selection
        info = self.format_info(agent, info)

        if observe:
            self._last_infos[
                agent
            ] = info  # TODO: is this correct to update infos only when observe=True?
            observation2 = self.transform_observation(agent, info)
            self.observations2[agent] = observation2
        else:
            observation2 = None  # that's how Zoo api_test.py requires it

        # rewards should be only updated after step, not on each observation before step
        # reward2 = self._cumulative_rewards2[agent]
        reward2 = info[INFO_REWARD_DICT]
        if self._scalarize_rewards:
            reward2 = sum(reward2.values())

        if self._override_infos:
            info = {}

        return (
            observation2,
            reward2,
            terminated,
            truncated,
            self.filter_info(agent, info),
        )

    def step(self, action: Action) -> None:
        self.step_single_agent(action)
        # self.step_multiple_agents({ self.agent_selection: action })

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

        # need to set current step rewards to zero for other agents
        # the agent should be visible in .rewards after it dies
        # (until its "dead step"), but during next agent's step
        # it should get zero reward
        for agent2 in self.agents:
            # this needs to be so according to Zoo unit test. See
            # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/test/api_test.py
            if self._scalarize_rewards:
                self._last_rewards2[agent2] = 0.0
            else:
                self._last_rewards2[agent2] = {}

        GridworldZooAecEnv.step(self, {"step": action})

        if agent not in self.agents:  # was "dead step"
            # dead agent needs to be removed from observations2 and _last_rewards2
            del self.observations2[agent]
            del self._last_rewards2[agent]
            return

        # observe observations, transform observations and rewards
        info = GridworldZooAecEnv.observe_info(self, agent)
        info = self.format_info(agent, info)

        self._last_infos[agent] = info
        observation2 = self.transform_observation(agent, info)
        self.observations2[agent] = observation2

        reward2 = info[INFO_REWARD_DICT]
        if self._scalarize_rewards:
            reward2 = sum(
                reward2.values()
            )  # this is currently used for unit tests and OpenAI baselines so no need for nonlinear utility transformations before summation
        self._last_rewards2[agent] = reward2

        # NB! cumulative reward should be calculated for all agents
        for agent2 in self.agents:
            info = GridworldZooAecEnv.observe_info(self, agent2)
            info = self.format_info(agent2, info)
            self._cumulative_rewards2[agent2] = info[INFO_CUMULATIVE_REWARD_DICT]
            if self._scalarize_rewards:
                self._cumulative_rewards2[agent2] = sum(
                    self._cumulative_rewards2[agent2].values()
                )  # this is currently used for unit tests and OpenAI baselines so no need for nonlinear utility transformations before summation

        terminated = self.terminations[agent]
        truncated = self.truncations[agent]

        if self._override_infos:
            info = {}

        logger.debug(
            "debug return",
            observation2,
            reward2,
            terminated,
            truncated,
            self.filter_info(agent, info),
        )
        return (
            observation2,
            reward2,
            terminated,
            truncated,
            self.filter_info(agent, info),
        )

    def step_multiple_agents(self, actions: Dict[str, Action]) -> Step:
        """step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - truncateds
        - info
        dicts where each dict looks like:
            {agent_1: action_of_agent_1, agent_2: action_of_agent_2}
        or generally:
            {<agent_name>: <agent_action or None if agent is done>}
        """
        logger.debug("debug actions", actions)
        # If a user passes in actions with no agents,
        # then just return empty observations, etc.
        if not actions:
            return {}, {}, {}, {}, {}

        stepped_agents = []
        rewards2 = {}
        infos = {}

        # the agent should be visible in .rewards after it dies
        # (until its "dead step"), but during next agent's step
        # it should get zero reward
        for agent in actions.keys():
            # this needs to be so according to Zoo unit test. See
            # https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/test/api_test.py
            if self._scalarize_rewards:
                self._last_rewards2[agent] = 0.0
            else:
                self._last_rewards2[agent] = {}

        # loop over all agents in ENV NOT IN ACTIONS DICT
        for index in range(
            0, self.num_agents
        ):  # do one iteration over all ALIVE agents
            agent = self.agent_selection  # this returns only alive agents
            stepped_agents.append(agent)
            action = actions.get(agent, None)
            GridworldZooAecEnv.step(self, {"step": action})

            # was not "dead step" in which case the agent was removed in the above
            # call from self.agents list
            if agent not in self.agents:
                # dead agent needs to be removed from observations2 and _last_rewards2
                del self.observations2[agent]
                del self._last_rewards2[agent]
            else:
                if (
                    self.observe_immediately_after_agent_action
                ):  # observe BEFORE next agent takes its step?
                    # observe observations, transform observations and rewards
                    info = GridworldZooAecEnv.observe_info(self, agent)
                    info = self.format_info(agent, info)

                    infos[agent] = info
                    self._last_infos[agent] = info
                    self.observations2[agent] = self.transform_observation(agent, info)

                    self._last_rewards2[agent] = info[INFO_REWARD_DICT]
                    # NB! if the action of current agent somehow affects the rewards
                    # of other agents then the cumulative reward of the other agents
                    # needs to be updated here as well.
                    self._cumulative_rewards2[agent] = info[INFO_CUMULATIVE_REWARD_DICT]
                else:
                    info = GridworldZooAecEnv.observe_info(self, agent)
                    info = self.format_info(agent, info)

                    self._last_rewards2[agent] = info[INFO_REWARD_DICT]
                    # NB! if the action of current agent somehow affects the rewards
                    # of other agents then the cumulative reward of the other agents
                    # needs to be updated here as well.
                    self._cumulative_rewards2[agent] = info[INFO_CUMULATIVE_REWARD_DICT]

                if self._scalarize_rewards:
                    # this is currently used for unit tests and OpenAI baselines so no need for nonlinear utility transformations before summation
                    self._last_rewards2[agent] = sum(
                        self._last_rewards2[agent].values()
                    )
                    self._cumulative_rewards2[agent] = sum(
                        self._cumulative_rewards2[agent].values()
                    )

        # / for index in range(0, self.num_agents)

        if (
            not self.observe_immediately_after_agent_action
        ):  # observe only after ALL agents are done stepping?
            # observe observations, transform observations and rewards
            for agent in stepped_agents:
                if agent in self.agents:  # was not "dead step"
                    info = GridworldZooAecEnv.observe_info(self, agent)
                    # self.observe_from_location({agent: [1, 1]})    # for debugging
                    info = self.format_info(agent, info)
                    infos[agent] = info
                    self._last_infos[agent] = info
                    self.observations2[agent] = self.transform_observation(agent, info)

        terminateds = self.terminations
        truncateds = self.truncations

        if self._override_infos:
            infos = {agent: {} for agent in infos.keys()}

        logger.debug(
            "debug return",
            self.observations2,
            rewards2,
            terminateds,
            truncateds,
            self.filter_infos(infos),
        )
        return (
            self.observations2,
            rewards2,
            terminateds,
            truncateds,
            self.filter_infos(infos),
        )
