# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

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

from aintelope.agents.sb3_base_agent import (
    SB3BaseAgent,
    CustomCNN,
    PolicyWithConfigFactory,
)
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer
from aintelope.agents.sb3_handwritten_rules_expert import SB3HandWrittenRulesExpert
from zoo_to_gym_multiagent_adapter.singleagent_zoo_to_gym_adapter import (
    SingleAgentZooToGymAdapter,
)

import torch
import torch as th
from stable_baselines3 import A2C
from stable_baselines3.a2c.policies import CnnPolicy, MlpPolicy
import supersuit as ss

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


logger = logging.getLogger("aintelope.agents.a2c_agent")


class ExpertOverrideMixin:  # TODO: merge with code from PPO agent (the code is identical)
    def __init__(self, env_classname, agent_id, cfg, *args, **kwargs):
        self.cfg = cfg
        super().__init__(*args, **kwargs)

        self.expert = SB3HandWrittenRulesExpert(
            env_classname=env_classname,
            agent_id=agent_id,
            cfg=cfg,
            action_space=self.action_space,
            **cfg.hparams.agent_params,
        )

    def set_info(self, info):
        self.info = info

    def my_reset(self, observation, info):
        self.expert.reset()

    # code adapted from
    # https://github.com/DLR-RM/stable-baselines3/blob/dd7f5bfe63631630463f2f9bcb4762e6c040de12/stable_baselines3/common/policies.py#L636C5-L658C41
    @torch.no_grad()
    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """

        # orig_action = super().forward(obs, deterministic)

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # inserted code
        step = self.info["step"]
        episode = self.info["i_episode"]
        pipeline_cycle = self.info["i_pipeline_cycle"]
        test_mode = self.info["test_mode"]

        obs_nps = obs.detach().cpu().numpy()
        obs_np = obs_nps[0, :]

        (override_type, _random) = self.expert.should_override(
            deterministic,
            step,
            episode,
            pipeline_cycle,
            test_mode,
            obs_np,
        )
        if override_type != 0:
            action = self.expert.get_action(
                obs_np,
                self.info,
                step,
                episode,
                pipeline_cycle,
                test_mode,
                override_type,
                deterministic,
                _random,
            )
            # TODO: handle multiple observations and actions (for that we need also multiple infos)
            actions = [action]
            actions = torch.as_tensor(actions, device=obs.device, dtype=torch.long)
        else:
            actions = distribution.get_actions(deterministic=deterministic)

        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob


class CnnPolicyWithExpertOverride(ExpertOverrideMixin, CnnPolicy):
    pass


class MlpPolicyWithExpertOverride(ExpertOverrideMixin, MlpPolicy):
    pass


# need separate function outside of class in order to init multi-model training threads
def dqn_model_constructor(env, env_classname, agent_id, cfg):
    # policy_kwarg:
    # if you want to use CnnPolicy or MultiInputPolicy with image-like observation (3D tensor) that are already normalized, you must pass normalize_images=False
    # see the following links:
    # https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html
    # https://github.com/DLR-RM/stable-baselines3/issues/1863
    # Also: make sure your image is in the channel-first format.

    use_imitation_learning = (
        cfg.hparams.model_params.instinct_bias_epsilon_start > 0
        or cfg.hparams.model_params.instinct_bias_epsilon_end > 0
    )
    if use_imitation_learning:
        policy_override_class = (
            CnnPolicyWithExpertOverride
            if cfg.hparams.model_params.num_conv_layers > 0
            else MlpPolicyWithExpertOverride
        )
        policy = PolicyWithConfigFactory(
            env_classname, agent_id, cfg, policy_override_class
        )
    else:
        policy = (
            "CnnPolicy" if cfg.hparams.model_params.num_conv_layers > 0 else "MlpPolicy"
        )

    return A2C(
        policy,
        env,
        verbose=1,
        policy_kwargs=(
            {
                "normalize_images": False,
                "features_extractor_class": CustomCNN,  # need custom CNN in order to handle observation shape 9x9
                "features_extractor_kwargs": {
                    "features_dim": 256,  # TODO: config parameter. Note this is not related to the number of features in the original observation (15 or 39), this parameter here is model's internal feature dimensionality
                    "num_conv_layers": cfg.hparams.model_params.num_conv_layers,
                },
            }
            if cfg.hparams.model_params.num_conv_layers > 0
            else {"normalize_images": False}
        ),
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),  # Note, CUDA-based CPU performance is much better than Torch-CPU mode.
        tensorboard_log=cfg.tensorboard_dir,
        # optimize_memory_usage=True, # this argument is not supported with this algorithm because it does not have a replay buffer
    )


class A2CAgent(SB3BaseAgent):
    """A2CAgent class from stable baselines
    https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html

    """

    def __init__(
        self,
        env: PettingZooEnv = None,
        cfg: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__(env=env, cfg=cfg, **kwargs)

        self.model_constructor = dqn_model_constructor

        if (
            self.env.num_agents == 1 or self.test_mode
        ):  # during test, each agent has a separate in-process instance with its own model and not using threads/subprocesses
            env = SingleAgentZooToGymAdapter(env, self.id)
            # TODO: turn off GPU for A2C and use parallel computation which is supported by A2C:
            # env = make_vec_env(env, n_envs=8, vec_env_cls=SubprocVecEnv)
            self.model = self.model_constructor(env, self.env_classname, self.id, cfg)
        else:
            pass  # multi-model training will be automatically set up by the base class when self.model is None. These models will be saved to self.models and there will be only one agent instance in the main process. Actual agents will run in threads/subprocesses because SB3 requires Gym interface.
