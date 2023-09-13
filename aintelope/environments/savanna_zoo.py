import typing as typ
import logging
import functools

import numpy as np
import pygame

try:
    from gymnasium.spaces import Box, Discrete
    from gymnasium.utils import seeding

    gym_v26 = True
except:
    from gym.spaces import Box, Discrete
    from gym.utils import seeding

    gym_v26 = False

from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector

from aintelope.environments.env_utils.render_ascii import AsciiRenderState
from aintelope.environments.env_utils.distance import distance_to_closest_item
from aintelope.environments.savanna import (
    SavannaEnv,
    RenderSettings,
    RenderState,
    HumanRenderState,
    move_agent,
    reward_agent,
    PositionFloat,
    Action,
)

logger = logging.getLogger("aintelope.environments.savanna_zoo")


class SavannaZooParallelEnv(SavannaEnv, ParallelEnv):
    def __init__(self, env_params={}):
        SavannaEnv.__init__(self, env_params)
        ParallelEnv.__init__(self)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]


class SavannaZooSequentialEnv(SavannaZooParallelEnv, AECEnv):
    pass
