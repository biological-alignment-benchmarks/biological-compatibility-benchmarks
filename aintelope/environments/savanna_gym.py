from typing import Optional, Dict
import logging

import gymnasium as gym

from aintelope.environments.typing import (
    Observation,
    Reward,
    Info,
)
from aintelope.environments.savanna import SavannaEnv

Step = tuple[Observation, Reward, bool, bool, Info]

logger = logging.getLogger("aintelope.environments.savanna_gym")


class SavannaGymEnv(SavannaEnv, gym.Env):
    """Savanna environment class intended to be used with a single agent"""

    metadata = {
        "name": "savanna-v2",
        "render_fps": 3,
        "render_agent_radius": 5,
        "render_agent_color": (200, 50, 0),
        "render_grass_radius": 5,
        "render_grass_color": (20, 200, 0),
        "render_modes": ("human", "ascii", "offline"),
        "render_window_size": 512,
    }

    def __init__(self, env_params: Optional[Dict] = None):
        if env_params is None:
            env_params = {}

        SavannaEnv.__init__(self, env_params)
        gym.Env.__init__(self)
        assert self.metadata["amount_agents"] == 1, "agents must == 1 for gym env"

    def step(self, action) -> Step:
        actions = {self._agent_id: action}
        # should be: observations, rewards, dones, infos
        # but per agent
        res = SavannaEnv.step(self, actions)

        observations, rewards, terminateds, truncateds, infos = res
        dones = {
            key: terminated or truncateds[key]
            for (key, terminated) in terminateds.items()
        }

        # so just return the first
        i = self._agent_id
        observation = observations[i]
        reward = rewards[i]
        done = dones[i]
        info = infos[i]
        logger.warning(res)

        truncated = False
        return observation, reward, done, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if options is None:
            options = {}

        observations, infos = SavannaEnv.reset(self, seed, options)
        return (observations[self._agent_id], infos[self._agent_id])

    @property
    def _agent_id(self):
        return self.possible_agents[0]

    @property
    def action_space(self):
        return self._action_spaces[self._agent_id]

    @property
    def observation_space(self):
        return self._observation_spaces[self._agent_id]
