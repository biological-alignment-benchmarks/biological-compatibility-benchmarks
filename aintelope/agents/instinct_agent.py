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

from aintelope.environments.savanna_safetygrid import (
    GridworldZooBaseEnv,
    ACTION_RELATIVE_COORDINATE_MAP,
)

from aintelope.agents.instincts.safetygrid_instincts import (
    available_instincts_dict,
    format_float,
)

from aintelope.agents.q_agent import QAgent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

from typing import Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]

logger = logging.getLogger("aintelope.agents.instinct_agent")


class InstinctAgent(QAgent):
    """Agent class with instincts"""

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
        env: Environment = None,
        cfg: DictConfig = None,
        target_instincts: List[str] = [],
        **kwargs,
    ) -> None:
        self.id = agent_id
        self.trainer = trainer
        self.hparams = trainer.hparams
        self.env = env
        self.cfg = cfg
        self.done = False
        self.last_action = None

        self.target_instincts = target_instincts
        self.instincts = {}
    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        self.done = False
        self.last_action = None
        self.state = state
        self.info = info
        self.env_class = env_class

        self.init_instincts()

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
        """Given an observation, ask your model what to do. State is needed to be
        given here as other agents have changed the state!

        Returns:
            action (Optional[int]): index of action
        """

        if self.done:
            return None

        # TODO: warn if last_frame=0/1 or last_trial=0/1 or last_episode=0/1 in any of the below values: for disabling the epsilon counting for corresponding variable one should use -1
        epsilon = (
            self.hparams.model_params.eps_start - self.hparams.model_params.eps_end
        )
        if self.hparams.model_params.eps_last_frame > 1:
            epsilon *= max(0, 1 - step / self.hparams.model_params.eps_last_frame)
        if self.hparams.model_params.eps_last_trial > 1:
            epsilon *= max(0, 1 - trial / self.hparams.model_params.eps_last_trial)
        if self.hparams.model_params.eps_last_episode > 1:
            epsilon *= max(0, 1 - episode / self.hparams.model_params.eps_last_episode)
        if self.hparams.model_params.eps_last_pipeline_cycle > 1:
            epsilon *= max(
                0,
                1 - pipeline_cycle / self.hparams.model_params.eps_last_pipeline_cycle,
            )
        epsilon += self.hparams.model_params.eps_end

        instinct_epsilon = (
            self.hparams.model_params.instinct_bias_epsilon_start
            - self.hparams.model_params.instinct_bias_epsilon_end
        )
        if self.hparams.model_params.eps_last_frame > 1:
            instinct_epsilon *= max(
                0, 1 - step / self.hparams.model_params.eps_last_frame
            )
        if self.hparams.model_params.eps_last_trial > 1:
            instinct_epsilon *= max(
                0, 1 - trial / self.hparams.model_params.eps_last_trial
            )
        if self.hparams.model_params.eps_last_episode > 1:
            instinct_epsilon *= max(
                0, 1 - episode / self.hparams.model_params.eps_last_episode
            )
        if self.hparams.model_params.eps_last_pipeline_cycle > 1:
            instinct_epsilon *= max(
                0,
                1 - pipeline_cycle / self.hparams.model_params.eps_last_pipeline_cycle,
            )
        instinct_epsilon += self.hparams.model_params.instinct_bias_epsilon_end

        # print(f"Epsilon: {epsilon}")
        # print(f"Instinct bias epsilon: {instinct_epsilon}")

        action_space = self.trainer.action_spaces[self.id]
        if isinstance(action_space, Discrete):
            min_action = action_space.start
            max_action = action_space.start + action_space.n - 1
        else:
            min_action = action_space.min_action
            max_action = action_space.max_action

        if instinct_epsilon != 0:
            # calculate action reward predictions using instincts
            action_rewards = defaultdict(float)

            for instinct_name, instinct_object in self.instincts.items():
                instinct_action_rewards = {}
                # predict reward for all available actions
                for action in range(
                    min_action, max_action + 1
                ):  # NB! max_action is inclusive max
                    agent_coordinate = info[ACTION_RELATIVE_COORDINATE_MAP][action]

                    (
                        instinct_reward,
                        instinct_event,
                    ) = instinct_object.calc_reward(
                        self,
                        observation,
                        info,
                        agent_coordinate=agent_coordinate,
                        predicting=True,
                    )

                    instinct_action_rewards[action] = instinct_reward
                    action_rewards[
                        action
                    ] += instinct_reward  # TODO: nonlinear aggregation

                # debug helper  # TODO: refactor into a separate method
                # if instinct_name == "gold":
                #    q_values = np.zeros([max_action - min_action + 1], np.float32)
                #    for action, bias in instinct_action_rewards.items():
                #        q_values[action - min_action] = bias
                #    print(f"gold q_values: {format_float(q_values)}")

            instinct_q_values = action_rewards  # instincts see only one step ahead

        else:
            instinct_q_values = None

        # action = super().get_action(observation=observation, info=info, step=step, trial=trial, episode=episode, pipeline_cycle=pipeline_cycle, q_values=instinct_q_values)

        apply_instinct_eps_before_random_eps = (
            self.hparams.model_params.apply_instinct_eps_before_random_eps
        )

        if (
            not apply_instinct_eps_before_random_eps
            and epsilon > 0
            and np.random.random() < epsilon
        ):
            action = action_space.sample()

        elif (
            instinct_q_values is not None
            and instinct_epsilon > 0
            and np.random.random() < instinct_epsilon
        ):  # TODO: find a better way to combine epsilon and instinct_epsilon
            q_values = np.zeros([max_action - min_action + 1], np.float32)
            for action, bias in instinct_q_values.items():
                q_values[action - min_action] = bias
            action = (
                self.trainer.tiebreaking_argmax(q_values) + min_action
            )  # take best action predicted by instincts

        elif (
            apply_instinct_eps_before_random_eps
            and epsilon > 0
            and np.random.random() < epsilon
        ):
            action = action_space.sample()

        else:
            q_values = self.trainer.get_action(
                self.id, observation, self.info, step, trial, episode, pipeline_cycle
            )

            action = (
                self.trainer.tiebreaking_argmax(q_values) + min_action
            )  # when no axis is provided, argmax returns index into flattened array

            # q_values = self.policy_nets[agent_id](observation)
            # _, action = torch.max(q_values, dim=1)
            # action = int(action.item()) + min_action

        # NB! not calling q_agent.get_action code here at all

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
        Needed here to calculate instinctual rewards.

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
        # For future: add state (interoception) handling here when needed

        # calculate instinctual rewards
        if len(self.instincts) == 0:
            # use env reward if no instincts available
            instinct_events = []
            reward = score
        else:
            # interpret new_state and score to compute actual reward
            reward = 0
            instinct_events = []
            if next_state is not None:  # temporary, until we solve final states
                for instinct_name, instinct_object in self.instincts.items():
                    (
                        instinct_reward,
                        instinct_event,
                    ) = instinct_object.calc_reward(self, next_state, next_info)
                    reward += instinct_reward  # TODO: nonlinear aggregation
                    logger.debug(
                        f"Reward of {instinct_name}: {instinct_reward}; "
                        f"total reward: {reward}"
                    )
                    if instinct_event != 0:
                        instinct_events.append((instinct_name, instinct_event))

            # print(f"reward: {reward}")

        event = [self.id, self.state, self.last_action, reward, done, next_state]
        if not test_mode:  # TODO: do we need to update replay memories during test?
            self.trainer.update_memory(*event)
        self.state = next_state
        self.info = info
        return event

    def init_instincts(self) -> None:
        if issubclass(
            self.env_class, GridworldZooBaseEnv
        ):  # radically different types of environments may need different instincts
            available_instincts_dict_local = available_instincts_dict

        logger.debug(f"target_instincts: {self.target_instincts}")
        for instinct_name in self.target_instincts:
            if instinct_name not in available_instincts_dict_local:
                logger.warning(
                    f"Warning: could not find {instinct_name} "
                    "in available_instincts_dict"
                )
                continue

        self.instincts = {
            instinct: available_instincts_dict_local.get(instinct)()
            for instinct in self.target_instincts
            if instinct in available_instincts_dict_local
        }
        for instinct in self.instincts.values():
            instinct.reset()

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
        pass
