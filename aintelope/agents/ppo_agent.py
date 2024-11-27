import logging
from typing import List, NamedTuple, Optional, Tuple
from gymnasium.spaces import Discrete

import numpy as np
import numpy.typing as npt
import os
import datetime

from aintelope.agents import Agent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from stable_baselines3 import PPO
import supersuit as ss

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


logger = logging.getLogger("aintelope.agents.q_agent")


class HistoryStep(NamedTuple):
    state: Tuple[npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]]
    action: int
    reward: float
    done: bool
    instinct_events: List[Tuple[str, int]]
    next_state: Tuple[npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]]


class PPOAgent:
    """PPOAgent class from stable baselines
    https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
        env: Environment,
        target_instincts: List[
            str
        ] = [],  # unused, argument present for compatibility with other agents
    ) -> None:
        self.id = agent_id
        self.trainer = trainer
        self.hparams = trainer.hparams
        self.done = False
        self.last_action = None
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        # env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")
        self.model = PPO("MlpPolicy", env, verbose=1)

    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        self.done = False
        self.last_action = None
        self.state = state
        self.info = info
        self.env_class = env_class
        # if isinstance(self.state, tuple):
        #    self.state = self.state[0]

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

        Args:
            net: pytorch Module instance, the model
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action (Optional[int]): index of action
        """
        if self.done:
            return None

        action_space = self.trainer.action_spaces[self.id]

        action, _states = self.model.predict(observation)
        if isinstance(action_space, Discrete):
            min_action = action_space.start
        else:
            min_action = action_space.min_action
        action = action + min_action

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
        save_path: Optional[str] = None,  # TODO: this is unused right now
    ) -> list:
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

        assert self.last_action is not None

        next_state = observation

        if next_state is not None:
            next_s_hist = next_state
        else:
            next_s_hist = None

        event = [self.id, self.state, self.last_action, score, done, next_state]
        self.state = next_state
        self.info = info
        return event

    def train(self, steps):
        self.model.learn(total_timesteps=steps)

    def set_env(self, env):
        self.model.set_env(env)

    def save_model(self):
        dir_out = os.path.normpath(self.params.log_dir)
        checkpoint_dir = os.path.normpath(self.params.checkpoint_dir)
        path = os.path.join(dir_out, checkpoint_dir)
        os.makedirs(path, exist_ok=True)
        checkpoint_filename = self.params.experiment_name + "_" + self.id
        filename = os.path.join(
            path,
            checkpoint_filename
            + "-"
            + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"),
        )

        self.model.save(filename)

    def load_model(self, checkpoint):
        if checkpoint:
            self.model.load(checkpoint)
