import csv
import logging
from typing import List, Optional, Tuple
from gymnasium.spaces import Discrete

import numpy.typing as npt

from aintelope.agents.instincts.savanna_instincts_old import available_instincts_dict
from aintelope.agents.q_agent import HistoryStep, QAgent
from aintelope.aintelope_typing import ObservationFloat, PettingZooEnv
from aintelope.training.dqn_training import Trainer

logger = logging.getLogger("aintelope.agents.instinct_agent")


class InstinctAgent(QAgent):
    """Agent class with instincts"""

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
        target_instincts: List[str] = [],
    ) -> None:
        self.target_instincts = target_instincts
        self.instincts = {}

        super().__init__(
            agent_id=agent_id,
            trainer=trainer,
        )

    def reset(self, state, info, env_class) -> None:
        """Resets self and updates the state."""
        super().reset(state, info, env_class)
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

        return super().get_action(
            observation, info, step, trial, episode, pipeline_cycle
        )

    # TODO hack, figure out if state_to_namedtuple can be static somewhere
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
        next_info = info
        # For future: add state (interoception) handling here when needed

        # interrupt to do instinctual learning
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
        # interruption done

        if next_state is not None:
            next_s_hist = next_state
        else:
            next_s_hist = None
        # self.history.append(
        #    HistoryStep(
        #        state=self.state,
        #        action=self.last_action,
        #        reward=reward,
        #        done=done,
        #        instinct_events=instinct_events,
        #        next_state=next_s_hist,
        #    )
        # )

        event = [self.id, self.state, self.last_action, reward, done, next_state]
        if not test_mode:  # TODO: do we need to update replay memories during test?
            self.trainer.update_memory(*event)
        self.state = next_state
        self.info = info
        return event

    def init_instincts(self) -> None:
        logger.debug(f"target_instincts: {self.target_instincts}")
        for instinct_name in self.target_instincts:
            if instinct_name not in available_instincts_dict:
                logger.warning(
                    f"Warning: could not find {instinct_name} "
                    "in available_instincts_dict"
                )
                continue

        self.instincts = {
            instinct: available_instincts_dict.get(instinct)()
            for instinct in self.target_instincts
            if instinct in available_instincts_dict
        }
        for instinct in self.instincts.values():
            instinct.reset()
