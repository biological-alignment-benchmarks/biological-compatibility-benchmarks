# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import csv
import logging
import hashlib
from typing import List, Optional, Tuple
from collections import defaultdict
from gymnasium.spaces import Discrete
from gymnasium.utils import seeding

from omegaconf import DictConfig

import numpy as np
import numpy.typing as npt

from aintelope.environments.savanna_safetygrid import (
    GridworldZooBaseEnv,
    ACTION_RELATIVE_COORDINATE_MAP,
)

from aintelope.agents.handwritten_rules.savanna_safetygrid_handwritten_rules import (
    savanna_safetygrid_available_handwritten_rules_dict,
    format_float,
)

from aintelope.agents.q_agent import QAgent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from typing import Union
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]

logger = logging.getLogger("aintelope.agents.sb3_handwritten_rules_expert")


class SB3HandWrittenRulesExpert(object):
    """Handwritten rules for SB3"""

    def __init__(
        self,
        env_classname: str,
        agent_id: str,
        cfg: DictConfig,
        target_handwritten_rules: List[str],
        action_space: spaces.Space,
    ) -> None:
        self.id = agent_id
        self.env_classname = env_classname
        self.cfg = cfg
        self.hparams = cfg.hparams
        # self.last_action = None
        self.action_space = action_space

        self.target_handwritten_rules = target_handwritten_rules
        self.handwritten_rules = {}

    def reset(self) -> None:
        self.init_handwritten_rules()

    def tiebreaking_argmax(
        self,
        arr,
        _random=None,
    ):
        """Avoids the agent from repeatedly taking move-left action when the handwritten rule tells the agent to move away from current cell in any direction. Then the handwritten rule will not provide any q value difference in its q values for the different directions, they would be equal. Naive np.argmax would just return the index of first moving action, which happens to be always move-left action."""

        max_values_bitmap = np.isclose(arr, arr.max())
        max_values_indexes = np.flatnonzero(max_values_bitmap)

        if (
            len(max_values_indexes) == 0
        ):  # Happens when all values are infinities or nans. This would cause np.random.choice to throw.
            # result = np.random.randint(0, len(arr))
            result = int(_random.random() * len(arr))
        else:
            result = _random.choice(
                max_values_indexes
            )  # TODO: seed for this random generator

        return result

    def should_override(
        self,
        deterministic: bool = False,  # This is set only during evaluation, not training and the meaning is that the agent is greedy - it takes the best action. It does NOT mean that the action is always same.
        step: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
        test_mode: bool = False,
        observation=None,
    ) -> int:
        _random = np.random

        if self.hparams.num_episodes == 0:  # there is no training
            # detected pure handwritten rules agent test without training - if that config is detected then test mode will use the handwritten rules
            return 1, _random  # handwritten rules mode
        elif test_mode:
            return 0, None  # SB3 policy mode

        # TODO: warn if last_frame=0/1 or last_env_layout_seed=0/1 or last_episode=0/1 in any of the below values: for disabling the epsilon counting for corresponding variable one should use -1
        epsilon = (
            self.hparams.model_params.eps_start - self.hparams.model_params.eps_end
        )
        if self.hparams.model_params.eps_last_frame > 1:
            epsilon *= max(0, 1 - step / self.hparams.model_params.eps_last_frame)
        if self.hparams.model_params.eps_last_env_layout_seed > 1:
            epsilon *= max(
                0,
                1
                - env_layout_seed / self.hparams.model_params.eps_last_env_layout_seed,
            )
        if self.hparams.model_params.eps_last_episode > 1:
            epsilon *= max(0, 1 - episode / self.hparams.model_params.eps_last_episode)
        if self.hparams.model_params.eps_last_pipeline_cycle > 1:
            epsilon *= max(
                0,
                1 - pipeline_cycle / self.hparams.model_params.eps_last_pipeline_cycle,
            )
        epsilon += self.hparams.model_params.eps_end

        handwritten_rule_epsilon = (
            self.hparams.model_params.handwritten_rule_bias_epsilon_start
            - self.hparams.model_params.handwritten_rule_bias_epsilon_end
        )
        if self.hparams.model_params.eps_last_frame > 1:
            handwritten_rule_epsilon *= max(
                0, 1 - step / self.hparams.model_params.eps_last_frame
            )
        if self.hparams.model_params.eps_last_env_layout_seed > 1:
            handwritten_rule_epsilon *= max(
                0,
                1
                - env_layout_seed / self.hparams.model_params.eps_last_env_layout_seed,
            )
        if self.hparams.model_params.eps_last_episode > 1:
            handwritten_rule_epsilon *= max(
                0, 1 - episode / self.hparams.model_params.eps_last_episode
            )
        if self.hparams.model_params.eps_last_pipeline_cycle > 1:
            handwritten_rule_epsilon *= max(
                0,
                1 - pipeline_cycle / self.hparams.model_params.eps_last_pipeline_cycle,
            )
        handwritten_rule_epsilon += (
            self.hparams.model_params.handwritten_rule_bias_epsilon_end
        )

        apply_handwritten_rule_eps_before_random_eps = (
            self.hparams.model_params.apply_handwritten_rule_eps_before_random_eps
        )

        if (
            not apply_handwritten_rule_eps_before_random_eps
            and epsilon > 0
            and _random.random() < epsilon
        ):
            return 2, _random  # random exploration mode

        elif (
            handwritten_rule_epsilon > 0 and _random.random() < handwritten_rule_epsilon
        ):  # TODO: find a better way to combine epsilon and handwritten_rule_epsilon
            return 1, _random  # handwritten rules mode

        elif (
            apply_handwritten_rule_eps_before_random_eps
            and epsilon > 0
            and _random.random() < epsilon
        ):
            return 2, _random  # random exploration mode

        else:
            return 0, None  # SB3 policy mode

    def get_action(
        self,
        observation=None,
        info: dict = {},
        step: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
        test_mode: bool = False,
        override_type: int = 0,
        deterministic: bool = False,  # This is set only during evaluation, not training and the meaning is that the agent is greedy - it takes the best action. It does NOT mean that the action is always same.
        _random=None,
    ) -> Optional[int]:
        """Given an observation, ask your rules what to do.

        Returns:
            action (Optional[int]): index of action
        """

        action_space = self.action_space
        if isinstance(action_space, Discrete):
            min_action = action_space.start
            max_action = action_space.start + action_space.n - 1
        else:
            min_action = action_space.min_action
            max_action = action_space.max_action

        # calculate action reward predictions using handwritten rules
        predicted_action_rewards = defaultdict(float)

        for (
            handwritten_rule_name,
            handwritten_rule_object,
        ) in self.handwritten_rules.items():
            handwritten_rule_action_rewards = {}
            # predict reward for all available actions
            for action in range(
                min_action, max_action + 1
            ):  # NB! max_action is inclusive max
                agent_coordinate = info[ACTION_RELATIVE_COORDINATE_MAP][action]

                (
                    handwritten_rule_reward,
                    handwritten_rule_event,
                ) = handwritten_rule_object.calc_reward(
                    self,
                    observation,
                    info,
                    agent_coordinate=agent_coordinate,
                    predicting=True,
                )

                handwritten_rule_action_rewards[action] = handwritten_rule_reward
                predicted_action_rewards[
                    action
                ] += handwritten_rule_reward  # TODO: nonlinear aggregation

            # debug helper  # TODO: refactor into a separate method
            # if handwritten_rule_name == "gold":
            #    q_values = np.zeros([max_action - min_action + 1], np.float32)
            #    for action, bias in handwritten_rule_action_rewards.items():
            #        q_values[action - min_action] = bias
            #    print(f"gold q_values: {format_float(q_values)}")

        handwritten_rule_q_values = (
            predicted_action_rewards  # handwritten rules see only one step ahead
        )

        if override_type == 2:
            action = action_space.sample()

        elif override_type == 1:
            q_values = np.zeros([max_action - min_action + 1], np.float32)
            for action, q_value in handwritten_rule_q_values.items():
                q_values[action - min_action] = q_value
            action = (
                self.tiebreaking_argmax(
                    q_values,
                    _random,
                )
                + min_action
            )  # take best action predicted by handwritten rules

        else:
            action = None

        # print(f"Action: {action}")
        # self.last_action = action
        return action

    def init_handwritten_rules(self) -> None:
        if self.env_classname in [
            "aintelope.environments.savanna_safetygrid.GridworldZooBaseEnv"
        ]:  # radically different types of environments may need different handwritten rules
            available_handwritten_rules_dict_local = (
                savanna_safetygrid_available_handwritten_rules_dict
            )

        logger.debug(f"target_handwritten_rules: {self.target_handwritten_rules}")
        for handwritten_rule_name in self.target_handwritten_rules:
            if handwritten_rule_name not in available_handwritten_rules_dict_local:
                logger.warning(
                    f"Warning: could not find {handwritten_rule_name} "
                    "in available_handwritten_rules_dict"
                )
                continue

        self.handwritten_rules = {
            handwritten_rule: available_handwritten_rules_dict_local.get(
                handwritten_rule
            )()
            for handwritten_rule in self.target_handwritten_rules
            if handwritten_rule in available_handwritten_rules_dict_local
        }
        for handwritten_rule in self.handwritten_rules.values():
            handwritten_rule.reset()
