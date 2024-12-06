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

from aintelope.agents import Agent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from ai_safety_gridworlds.helpers.gridworld_zoo_parallel_env import (
    INFO_REWARD_DICT,
)

import stable_baselines3
from stable_baselines3 import PPO
import supersuit as ss

from typing import Any, Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


logger = logging.getLogger("aintelope.agents.sb3_agent")


def vec_env_args(env, num_envs):
    assert num_envs == 1

    def env_fn():
        # env_copy = cloudpickle.loads(cloudpickle.dumps(env))
        env_copy = env  # TODO: add an assertion check that verifies that this "cloning" function is called only once per environment
        return env_copy

    return [env_fn] * num_envs, env.observation_space, env.action_space


def is_json_serializable(item: Any) -> bool:
    return False


class SB3BaseAgent(Agent):
    """SB3BaseAgent abstract class for stable baselines 3
    https://pettingzoo.farama.org/tutorials/sb3/waterworld/
    https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
    https://spinningup.openai.com/en/latest/algorithms/ppo.html
    """

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
        env: Environment,
        cfg: DictConfig,
        i_pipeline_cycle: int = 0,
        events: pd.DataFrame = None,
        score_dimensions: list = [],
        progressbar: RobustProgressBar = None,
        **kwargs,
    ) -> None:
        self.id = agent_id
        self.cfg = cfg
        self.env = env
        self.i_pipeline_cycle = i_pipeline_cycle
        self.next_episode_no = 0
        self.total_steps_across_episodes = 0
        self.score_dimensions = score_dimensions
        self.progressbar = progressbar
        self.events = events
        self.done = False
        self.last_action = None
        self.info = None
        self.state = None
        self.infos = {}
        self.states = {}

        ss.vector.vector_constructors.vec_env_args = vec_env_args  # The original function tries to do environment cloning, but absl flags currently do not support it. Since we need only one environment, there is no reason for cloning, so lets replace the cloning function with identity function.
        stable_baselines3.common.save_util.is_json_serializable = is_json_serializable  # The original function throws many "Pythonic" exceptions which make debugging in Visual Studio too noisy since VS does not have capacity to filter out handled exceptions

    # this method is currently called only in test mode
    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        self.done = False
        self.last_action = None

        self.state = state
        self.info = info
        self.states = {self.id: state}  # TODO: multi-agent support
        self.infos = {self.id: info}

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

        # action_space = self.env.action_space(self.id)

        action, _states = self.model.predict(observation, deterministic=False)
        action = np.asarray(
            action
        ).item()  # SB3 sends actions in wrapped into an one-item array for some reason. np.asarray is also able to handle lists.
        # if isinstance(action_space, Discrete):
        #    min_action = action_space.start
        # else:
        #    min_action = action_space.min_action
        # action = action + min_action

        self.state = observation
        self.states[self.id] = observation  # TODO: multi-agent support

        self.last_action = action
        return action

    def env_pre_reset_callback(self, seed, options, *args, **kwargs):
        assert seed is None

        i_episode = (
            self.next_episode_no
        )  # cannot use env.get_next_episode_no() here since its counter is reset for each new trial
        self.next_episode_no += 1  # no need to worry about the first reset happening multiple times in experiments.py since the current callback is activated only before self.model.learn() is called

        trial_no = (
            int(
                i_episode / self.cfg.hparams.trial_length
            )  # TODO ensure different trial no during test when num_actual_train_episodes is not divisible by trial_length
            if self.cfg.hparams.trial_length > 0
            else i_episode  # this ensures that during test episodes, trial_no based map randomization seed is different from training seeds. The environment is re-constructed when testing starts. Without explicitly providing trial_no, the map randomization seed would be automatically reset to trial_no = 0, which would overlap with the training seeds.
        )

        kwargs["trial_no"] = trial_no

        return (True, seed, options, args, kwargs)  # allow reset

    def env_post_reset_callback(self, states, infos, seed, options, *args, **kwargs):
        self.state = states[self.id]
        self.info = infos[self.id]
        self.states = states
        self.infos = infos

    def parallel_env_post_step_callback(
        self,
        actions,
        next_states,
        scores,
        terminateds,
        truncateds,
        infos,
        *args,
        **kwargs,
    ):
        if self.events is None:
            return

        self.total_steps_across_episodes += 1
        if self.progressbar is not None:
            self.progressbar.update(self.total_steps_across_episodes)

        i_pipeline_cycle = self.i_pipeline_cycle
        i_episode = (
            self.next_episode_no - 1
        )  # cannot use env.get_next_episode_no() here since its counter is reset for each new trial
        trial_no = (
            self.env.get_trial_no()
        )  # no need to substract 1 here since trial_no value is overridden in env_pre_reset_callback
        step = (
            self.env.get_step_no() - 1
        )  # get_step_no() returned step indexes start with 1
        test_mode = False

        for agent, next_state in next_states.items():
            state = self.states[agent]
            action = actions[agent]
            action = np.asarray(
                action
            ).item()  # SB3 sends actions in wrapped into an one-item array for some reason. np.asarray is also able to handle lists. Gridworlds is able to handle such wrapped actions ok.
            info = infos[agent]
            score = scores[agent]
            score2 = info[
                INFO_REWARD_DICT
            ]  # do not use scores[agent] in env_step_info since it is scalarised
            done = terminateds[agent] or truncateds[agent]

            agent_step_info = [
                agent,
                state,
                action,
                score,
                done,
                next_state,
            ]  # NB! agent_step_info uses scalarised score

            env_step_info = (
                [score2.get(dimension, 0) for dimension in self.score_dimensions]
                if isinstance(score2, dict)
                else [score2]
            )

            # NB! each agent has their own event
            self.events.loc[len(self.events)] = (
                [
                    self.cfg.experiment_name,
                    i_pipeline_cycle,
                    i_episode,
                    trial_no,
                    step,
                    test_mode,
                ]
                + agent_step_info
                + env_step_info
            )

        # / for agent, next_state in next_states.items():

        self.states = next_states
        self.infos = infos

    def sequential_env_post_step_callback(
        self,
        agent,
        action,
        next_state,
        score,
        terminated,
        truncated,
        info,
        *args,
        **kwargs,
    ):
        if self.events is None:
            return

        self.total_steps_across_episodes += 1
        if self.progressbar is not None:
            self.progressbar.update(self.total_steps_across_episodes)

        action = np.asarray(
            action
        ).item()  # SB3 sends actions in wrapped into an one-item array for some reason. np.asarray is also able to handle lists. Gridworlds is able to handle such wrapped actions ok.
        done = terminated or truncated
        score2 = info[
            INFO_REWARD_DICT
        ]  # do not use score in env_step_info since it is scalarised
        agent_step_info = [
            agent,
            self.state,
            action,
            score,
            done,
            next_state,
        ]  # NB! agent_step_info uses scalarised score

        self.state = next_state
        self.info = info

        env_step_info = (
            [score2.get(dimension, 0) for dimension in self.score_dimensions]
            if isinstance(score2, dict)
            else [score2]
        )

        i_pipeline_cycle = self.i_pipeline_cycle
        i_episode = (
            self.next_episode_no - 1
        )  # cannot use env.get_next_episode_no() here since its counter is reset for each new trial
        trial_no = (
            self.env.get_trial_no()
        )  # no need to substract 1 here since trial_no value is overridden in env_pre_reset_callback
        step = (
            self.env.get_step_no() - 1
        )  # get_step_no() returned step indexes start with 1
        test_mode = False

        self.events.loc[len(self.events)] = (
            [
                self.cfg.experiment_name,
                i_pipeline_cycle,
                i_episode,
                trial_no,
                step,
                test_mode,
            ]
            + agent_step_info
            + env_step_info
        )

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

        # if next_state is not None:
        #    next_s_hist = next_state
        # else:
        #    next_s_hist = None

        event = [self.id, self.state, self.last_action, score, done, next_state]
        self.state = next_state
        self.info = info
        return event

    def train(self, steps):
        self.env._pre_reset_callback2 = (
            self.env_pre_reset_callback
        )  # pre-reset callback is same for both parallel and sequential environment
        self.env._post_reset_callback2 = (
            self.env_post_reset_callback
        )  # post-reset callback is same for both parallel and sequential environment
        if isinstance(self.env, ParallelEnv):
            self.env._post_step_callback2 = self.parallel_env_post_step_callback
        else:
            self.env._post_step_callback2 = self.sequential_env_post_step_callback

        self.model.learn(total_timesteps=steps)

        self.env._pre_reset_callback2 = None
        self.env._post_reset_callback2 = None
        self.env._post_step_callback2 = None

    # def set_env(self, env):
    #    self.model.set_env(env)

    def save_model(self):
        dir_out = os.path.normpath(self.cfg.log_dir)
        checkpoint_dir = os.path.normpath(self.cfg.checkpoint_dir)
        path = os.path.join(dir_out, checkpoint_dir)
        os.makedirs(path, exist_ok=True)
        checkpoint_filename = self.cfg.experiment_name + "_" + self.id
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
