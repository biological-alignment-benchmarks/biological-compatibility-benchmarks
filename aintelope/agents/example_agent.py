import csv
import logging
from typing import List, Optional, Tuple
from collections import defaultdict
from gymnasium.spaces import Discrete

import numpy as np
import numpy.typing as npt

from aintelope.environments.savanna_safetygrid import ACTION_RELATIVE_COORDINATE_MAP

from aintelope.agents.q_agent import QAgent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

logger = logging.getLogger("aintelope.agents.example_agent")


class ExampleAgent(QAgent):
    """Example agent class"""

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            trainer=trainer,
        )

    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        super().reset(state, info, env_class)

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
        next_info = info
        # For future: add state (interoception) handling here when needed

        reward = 0    # TODO
        # print(f"reward: {reward}")

        if next_state is not None:
            next_s_hist = next_state
        else:
            next_s_hist = None

        event = [self.id, self.state, self.last_action, reward, done, next_state]
        self.state = next_state
        self.info = info
        return event

