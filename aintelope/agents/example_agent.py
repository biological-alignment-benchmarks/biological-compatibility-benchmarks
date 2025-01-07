# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import csv
import logging
from typing import List, Optional, Tuple
from collections import defaultdict
from gymnasium.spaces import Discrete

from omegaconf import DictConfig

import numpy as np
import numpy.typing as npt

from aintelope.environments.savanna_safetygrid import ACTION_RELATIVE_COORDINATE_MAP

from aintelope.agents.abstract_agent import Agent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]

logger = logging.getLogger("aintelope.agents.example_agent")


class ExampleAgent(Agent):
    """Example agent class"""

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
        env: Environment = None,
        cfg: DictConfig = None,
        **kwargs,
    ) -> None:
        self.id = agent_id
        self.trainer = trainer
        self.env = env
        self.cfg = cfg
        self.done = False
        self.last_action = None

    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        self.done = False
        self.last_action = None
        self.state = state
        self.info = info
        self.env_class = env_class

    def get_action(
        self,
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        step: int = 0,
        trial: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
    ) -> Optional[int]:
        """Given an observation, ask your net what to do. State is needed to be
        given here as other agents have changed the state!

        Returns:
            action (Optional[int]): index of action
        """

        if self.done:
            return None

        action_space = self.env.action_spaces[self.id]
        if isinstance(action_space, Discrete):
            min_action = action_space.start
            max_action = action_space.start + action_space.n - 1
        else:
            min_action = action_space.min_action
            max_action = action_space.max_action

        action = action_space.sample()

        # print(f"Action: {action}")
        self.last_action = action
        return action

    def update(
        self,
        env: PettingZooEnv = None,
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        score: float = 0.0,
        done: bool = False,
        test_mode: bool = False,
    ) -> list:
        """
        Takes observations and updates trainer on perceived experiences.

        Args:
            env: Environment
            observation: Tuple[ObservationArray, ObservationArray]
            score: Only baseline uses score as a reward
            done: boolean whether run is done
        Returns:
            agent_id (str): same as elsewhere ("agent_0" among them)
            state (Tuple[npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]]): input for the net
            action (int): index of action
            reward (float): reward signal
            done (bool): if agent is done
            next_state (npt.NDArray[ObservationFloat]): input for the net
        """

        assert self.last_action is not None

        next_state = observation
        next_info = info

        # TODO: implement any learning mechanisms here

        reward = 0  # TODO: This will be sent to the log file
        # print(f"reward: {reward}")

        event = [self.id, self.state, self.last_action, reward, done, next_state]
        self.state = next_state
        self.info = info
        return event

    def init_model(
        self,
        observation_shape,
        action_space,
        unit_test_mode: bool,
        checkpoint: Optional[str] = None,
        *args,
        **kwargs,
    ):
        self.trainer.add_agent(
            self.id,
            observation_shape,
            action_space,
            unit_test_mode,
            checkpoint,
            *args,
            **kwargs,
        )

    def save_model(
        self,
        i_episode,
        path,
        experiment_name,
        use_separate_models_for_each_experiment,
        *args,
        **kwargs,
    ):
        self.trainer.save_model(
            self.id,
            i_episode,
            path,
            experiment_name,
            use_separate_models_for_each_experiment,
            *args,
            **kwargs,
        )
