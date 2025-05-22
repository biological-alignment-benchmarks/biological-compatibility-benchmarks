# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
import logging
import traceback
from typing import List, NamedTuple, Optional, Tuple
from gymnasium.spaces import Discrete

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from aintelope.utils import RobustProgressBar

import numpy as np
import numpy.typing as npt
import os
import sys
import datetime

from aintelope.config.config_utils import select_gpu, set_priorities, set_memory_limits

from aintelope.agents import Agent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from aintelope.environments.savanna_safetygrid import (
    INFO_REWARD_DICT,
)
from zoo_to_gym_multiagent_adapter.multiagent_zoo_to_gym_adapter import (
    MultiAgentZooToGymAdapterGymSide,
    MultiAgentZooToGymAdapterZooSide,
)

import stable_baselines3
from stable_baselines3.common.callbacks import CheckpointCallback

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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


# TODO: move to a separate file
class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, num_conv_layers=2):
        super().__init__(observation_space, features_dim)

        # TODO: make this architecture configurable

        # Current observation_space is (15, 9, 9) - channels=15, height=9, width=9
        # Let's build a small CNN with two conv layers:
        #   Conv1: kernel_size=3, stride=1, padding=1 - keeps spatial dims at 9x9 -> 9x9
        #   Conv2: kernel_size=3, stride=2, padding=1 - downsamples from 9x9 -> 5x5
        #   Conv3: kernel_size=3, stride=2, padding=1 - downsamples from 5x5 -> 3x3
        #   Flatten into a linear layer

        print("num_conv_layers: " + str(num_conv_layers))

        if num_conv_layers == 2:
            self.cnn = nn.Sequential(
                nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        elif num_conv_layers == 3:
            self.cnn = nn.Sequential(
                nn.Conv2d(15, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                # TODO: test whether adding this third layer helps. It would make the output shape to 3x3.
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
        else:
            raise ValueError("num_conv_layers")

        # TODO, experiment with:
        # * BatchNorm / LayerNorm: Try adding normalization layers if training stability is an issue.
        # * Residual connections: In case of a deeper network, sometimes adding skip connections (resnets) helps performance, though it's more advanced.

        # Figure out the output shape of self.cnn:
        # 1) With the above, input: (batch_size, 15, 9, 9)
        # 2) After Conv1: (batch_size, 32, 9, 9)
        # 3) After Conv2: (batch_size, 64, 5, 5) => 64*5*5=1600
        # 4) After Conv3: (batch_size, 128, 3, 3) => 128*3*3=1152
        # We feed that into a linear layer to get features_dim=256
        with torch.no_grad():
            sample_input = torch.zeros(1, 15, 9, 9)
            n_flatten = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def sb3_agent_train_thread_entry_point(
    pipe,
    gpu_index,
    num_total_steps,
    model_constructor,
    agent_id,
    checkpoint_filename,
    cfg,
    observation_space,
    action_space,
):
    if (
        sys.gettrace() is None
    ):  # do not set low priority while debugging. Note that unit tests also set sys.gettrace() to not-None
        set_priorities()

    if (
        os.name != "nt"
    ):  # Under Windows, the memory limit is shared with parent process as a job object
        set_memory_limits()

    # activate selected GPU
    select_gpu(gpu_index)

    env_wrapper = MultiAgentZooToGymAdapterGymSide(
        pipe, agent_id, checkpoint_filename, observation_space, action_space
    )
    try:
        filename_timestamp_sufix_str = datetime.datetime.now().strftime(
            "%Y_%m_%d_%H_%M_%S_%f"
        )
        filename_with_timestamp = (
            checkpoint_filename + "-" + filename_timestamp_sufix_str
        )

        # resulting filename looks like checkpointfilename_timestamp_200000_steps.zip next checkpointfilename_timestamp_400000_steps.zip etc
        # note that steps count in the checkpoint filename is multiplied by the number of agents in multi-agent scenarios
        checkpoint_callback = (
            CheckpointCallback(
                save_freq=cfg.hparams.save_frequency,  # save frequency in timesteps
                save_path=os.path.dirname(filename_with_timestamp),
                name_prefix=os.path.basename(filename_with_timestamp),
            )
            if cfg.hparams.save_frequency > 0
            else None
        )

        model = model_constructor(env_wrapper, cfg)
        model.learn(total_timesteps=num_total_steps, callback=checkpoint_callback)
        env_wrapper.save_or_return_model(model, filename_timestamp_sufix_str)
    except (
        Exception
    ) as ex:  # NB! need to catch exception so that the env wrapper can signal the training ended
        info = str(ex) + os.linesep + traceback.format_exc()
        env_wrapper.terminate_with_exception(info)
        print(info)


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
        test_mode: bool = False,
        i_pipeline_cycle: int = 0,
        events: pd.DataFrame = None,  # TODO: this is no longer a DataFrame, but an EventLog
        score_dimensions: list = [],
        progressbar: RobustProgressBar = None,
        **kwargs,
    ) -> None:
        self.id = agent_id
        self.cfg = cfg
        self.env = env
        self.test_mode = test_mode
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
        self.model = None  # for single-model scenario
        self.models = None  # for multi-model scenario
        self.exceptions = None  # for multi-model scenario
        self.model_constructor = None  # for multi-model scenario

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
        observation: Tuple[  # TODO: SB3 observation is NOT a Tuple
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        step: int = 0,
        env_layout_seed: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
    ) -> Optional[int]:
        """Given an observation, ask your model what to do.
        Called during test only, not during training.

        Returns:
            action (Optional[int]): index of action
        """
        if self.done:
            return None

        # action_space = self.env.action_space(self.id)

        action, _states = self.model.predict(
            observation, deterministic=True
        )  # TODO: config setting for "deterministic" parameter
        action = np.asarray(
            action
        ).item()  # SB3 sends actions in wrapped into an one-item array for some reason. np.asarray is also able to handle lists.

        action_space = self.env.action_spaces[self.id]
        if isinstance(action_space, Discrete):
            min_action = action_space.start
        else:
            min_action = action_space.min_action
        assert action >= min_action

        self.state = observation
        self.states[self.id] = observation  # TODO: multi-agent support

        self.last_action = action
        return action

    def env_pre_reset_callback(self, seed, options, *args, **kwargs):
        assert seed is None

        self.events.flush()

        i_episode = (
            self.next_episode_no
        )  # cannot use env.get_next_episode_no() here since its counter is reset for each new env_layout_seed
        self.next_episode_no += 1  # no need to worry about the first reset happening multiple times in experiments.py since the current callback is activated only before self.model.learn() is called

        env_layout_seed = (
            int(
                i_episode / self.cfg.hparams.env_layout_seed_repeat_sequence_length
            )  # TODO ensure different env_layout_seed during test when num_actual_train_episodes is not divisible by env_layout_seed_repeat_sequence_length
            if self.cfg.hparams.env_layout_seed_repeat_sequence_length > 0
            else i_episode  # this ensures that during test episodes, env_layout_seed based map randomization seed is different from training seeds. The environment is re-constructed when testing starts. Without explicitly providing env_layout_seed, the map randomization seed would be automatically reset to env_layout_seed = 0, which would overlap with the training seeds.
        )

        kwargs["env_layout_seed"] = env_layout_seed

        return (True, seed, options, args, kwargs)  # allow reset

    def env_post_reset_callback(self, states, infos, seed, options, *args, **kwargs):
        self.state = states[self.id]
        self.info = infos[self.id]
        self.states = states
        self.infos = infos

    def env_pre_step_callback(self, actions):
        return actions  # you can modify the actions in this method but keep in mind that the calling RL algorithm might not become aware of the modified actions later

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
            self.progressbar.update(
                min(
                    self.total_steps_across_episodes, self.progressbar.max_value
                )  # PPO does extra episodes, which causes the step counter to go beyond max_value of progress bar
            )

        i_pipeline_cycle = self.i_pipeline_cycle
        i_episode = (
            self.next_episode_no - 1
        )  # cannot use env.get_next_episode_no() here since its counter is reset for each new env_layout_seed
        env_layout_seed = (
            self.env.get_env_layout_seed()
        )  # no need to substract 1 here since env_layout_seed value is overridden in env_pre_reset_callback
        step = (
            self.env.get_step_no() - 1
        )  # get_step_no() returned step indexes start with 1
        test_mode = False

        for agent, next_state in next_states.items():
            state = self.states[agent]
            action = actions.get(
                agent, None
            )  # may be None in case of multi-agent scenarios
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

            self.events.log_event(
                [
                    self.cfg.experiment_name,
                    i_pipeline_cycle,
                    i_episode,
                    env_layout_seed,
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
            self.progressbar.update(
                min(
                    self.total_steps_across_episodes, self.progressbar.max_value
                )  # PPO does extra episodes, which causes the step counter to go beyond max_value of progress bar
            )

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
        )  # cannot use env.get_next_episode_no() here since its counter is reset for each new env_layout_seed
        env_layout_seed = (
            self.env.get_env_layout_seed()
        )  # no need to substract 1 here since env_layout_seed value is overridden in env_pre_reset_callback
        step = (
            self.env.get_step_no() - 1
        )  # get_step_no() returned step indexes start with 1
        test_mode = False

        self.events.log_event(
            [
                self.cfg.experiment_name,
                i_pipeline_cycle,
                i_episode,
                env_layout_seed,
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

        event = [self.id, self.state, self.last_action, score, done, next_state]
        self.state = next_state
        self.info = info
        return event

    def train(self, num_total_steps):
        self.env._pre_reset_callback2 = (
            self.env_pre_reset_callback
        )  # pre-reset callback is same for both parallel and sequential environment
        self.env._post_reset_callback2 = (
            self.env_post_reset_callback
        )  # post-reset callback is same for both parallel and sequential environment
        self.env._pre_step_callback2 = self.env_pre_step_callback
        if isinstance(self.env, ParallelEnv):
            self.env._post_step_callback2 = self.parallel_env_post_step_callback
        else:
            self.env._post_step_callback2 = self.sequential_env_post_step_callback

        if self.model is not None:  # single-model scenario
            checkpoint_filenames = self.get_checkpoint_filenames(include_timestamp=True)
            filename_with_timestamp = checkpoint_filenames[self.id]

            # resulting filename looks like checkpointfilename_timestamp_100000_steps.zip next checkpointfilename_timestamp_200000_steps.zip etc
            checkpoint_callback = (
                CheckpointCallback(
                    save_freq=self.cfg.hparams.save_frequency,  # save frequency in timesteps
                    save_path=os.path.dirname(filename_with_timestamp),
                    name_prefix=os.path.basename(filename_with_timestamp),
                )
                if self.cfg.hparams.save_frequency > 0
                else None
            )

            self.model.learn(
                total_timesteps=num_total_steps, callback=checkpoint_callback
            )
        else:
            checkpoint_filenames = self.get_checkpoint_filenames(
                include_timestamp=False
            )

            OmegaConf.resolve(
                self.cfg
            )  # need to resolve the conf before passing to subprocesses since OmegaConf resolvers do not seem to work well in subprocesses (at least under Windows)

            env_wrapper = MultiAgentZooToGymAdapterZooSide(self.env, self.cfg)
            self.models, self.exceptions = env_wrapper.train(
                num_total_steps=num_total_steps,
                agent_thread_entry_point=sb3_agent_train_thread_entry_point,
                model_constructor=self.model_constructor,
                terminate_all_agents_when_one_excepts=True,
                checkpoint_filenames=checkpoint_filenames,
            )

        self.env._pre_reset_callback2 = None
        self.env._post_reset_callback2 = None
        self.env._post_step_callback2 = None

        if self.exceptions:
            raise Exception(str(self.exceptions))

    def get_checkpoint_filenames(self, include_timestamp=True):
        checkpoint_filenames = {}

        experiment_name = self.cfg.experiment_name
        use_separate_models_for_each_experiment = (
            self.cfg.hparams.use_separate_models_for_each_experiment
        )
        # if not use_separate_models_for_each_experiment:
        #    raise NotImplementedError("sharing models over experiments is not implemented yet")

        dir_out = os.path.normpath(self.cfg.log_dir)
        checkpoint_dir = os.path.normpath(self.cfg.checkpoint_dir)
        path = os.path.join(dir_out, checkpoint_dir)
        os.makedirs(path, exist_ok=True)

        for agent_id in self.env.possible_agents:
            checkpoint_filename = agent_id
            if use_separate_models_for_each_experiment:
                checkpoint_filename += "-" + experiment_name

            filename = os.path.join(path, checkpoint_filename)

            if include_timestamp:
                filename += "-" + datetime.datetime.now().strftime(
                    "%Y_%m_%d_%H_%M_%S_%f"
                )

            checkpoint_filenames[agent_id] = filename

        return checkpoint_filenames

    def save_model(self, *args, **kwargs):
        checkpoint_filenames = self.get_checkpoint_filenames(include_timestamp=True)
        models = {self.id: self.model} if self.model is not None else self.models

        for agent_id, model in models.items():
            if not isinstance(
                model, str
            ):  # model can contain a path to an already saved model
                checkpoint_filename = checkpoint_filenames[agent_id]
                model.save(checkpoint_filename)

    def init_model(
        self,
        observation_shape,
        action_space,
        unit_test_mode: bool,
        checkpoint: Optional[str] = None,
    ):
        if checkpoint:
            # NB! torch.cuda.device_count() > 0 is needed here since SB3 does not support CPU-based CUDA device during model load() or set_parameters() for some reason
            use_cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
            device = torch.device("cuda" if use_cuda else "cpu")

            # Warning: load() re-creates the model from scratch, it does not update it in-place! For an in-place load use set_parameters() instead.
            self.model.set_parameters(
                checkpoint, device=device
            )  # device argument in needed in case the model is loaded to CPU. SB3 seems to be buggy in that regard that it will crash during model load() or set_parameters() if Torch-CPU device is not explicitly specified.
