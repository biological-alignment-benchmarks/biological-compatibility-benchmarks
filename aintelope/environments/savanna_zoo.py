from typing import Optional, Dict
import logging
import functools

from pettingzoo import AECEnv, ParallelEnv

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

    metadata = {"render.modes": ["human", "ansi", "rgb_array"]}

    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}
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
