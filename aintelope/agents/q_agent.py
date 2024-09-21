import logging
from typing import List, NamedTuple, Optional, Tuple
from gymnasium.spaces import Discrete

import numpy as np
import numpy.typing as npt

from aintelope.agents import Agent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

logger = logging.getLogger("aintelope.agents.q_agent")


class QAgent(Agent):
    """QAgent class, functioning as a base class for agents"""

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
    ) -> None:
        self.id = agent_id
        self.trainer = trainer
        self.hparams = trainer.hparams
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

        Args:
            net: pytorch Module instance, the model
            epsilon: value to determine likelihood of taking a random action
            device: current device

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

        # print(f"Epsilon: {epsilon}")

        action_space = self.trainer.action_spaces[self.id]

        if np.random.random() < epsilon:
            action = action_space.sample()
        else:
            q_values = self.trainer.get_action(
                self.id, observation, self.info, step, trial, episode, pipeline_cycle
            )

            if isinstance(action_space, Discrete):
                min_action = action_space.start
            else:
                min_action = action_space.min_action
            action = (
                self.trainer.tiebreaking_argmax(q_values) + min_action
            )  # when no axis is provided, argmax returns index into flattened array

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
        # TODO

        event = [self.id, self.state, self.last_action, score, done, next_state]
        if not test_mode:  # TODO: do we need to update replay memories during test?
            self.trainer.update_memory(*event)
        self.state = next_state
        self.info = info
        return event
