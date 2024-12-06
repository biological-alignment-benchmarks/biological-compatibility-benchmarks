import logging
from typing import List, NamedTuple, Optional, Tuple
from gymnasium.spaces import Discrete

import pandas as pd
from omegaconf import DictConfig

from aintelope.utils import RobustProgressBar

import numpy as np
import numpy.typing as npt
import os
import datetime

from aintelope.agents.sb3_base_agent import SB3BaseAgent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from stable_baselines3 import DQN
import supersuit as ss

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


logger = logging.getLogger("aintelope.agents.dqn_agent")


class DQNAgent(SB3BaseAgent):
    """DQNAgent class from stable baselines
    https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html

    """

    def __init__(
        self,
        env: PettingZooEnv = None,
        **kwargs,
    ) -> None:
        super().__init__(env=env, **kwargs)

        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(
            env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
        )  # NB! num_vec_envs=1 is important here so that we can use identity function instead of cloning in vec_env_args

        self.model = DQN("CnnPolicy", env, verbose=1)

    # this method is currently called only in test mode
    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        super().reset(state, info, env_class)

    def get_action(self, **kwargs) -> Optional[int]:
        """Given an observation, ask your net what to do. State is needed to be
        given here as other agents have changed the state!

        Returns:
            action (Optional[int]): index of action
        """
        action = super().get_action(**kwargs)
        return action

    def update(self, **kwargs) -> list:
        """
        Takes observations and updates trainer on perceived experiences.
        Needed here to catch instincts.

        Args:
            env: Environment
            observation: Tuple[ObservationArray, ObservationArray]
            score: Only baseline uses score as a reward
            done: boolean whether run is done
            save_path: str
        Returns:
            agent_id (str): same as elsewhere ("agent_0" among them)
            state (Tuple[npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]]): input for the net
            action (int): index of action
            reward (float): reward signal
            done (bool): if agent is done
            next_state (npt.NDArray[ObservationFloat]): input for the net
        """
        event = super().update(**kwargs)
        return event

    def train(self, steps):
        super().train(steps)

    def save_model(self):
        super().save_model()

    def load_model(self, checkpoint):
        super().load_model(checkpoint)
