import csv
import logging
from typing import List, Optional

import numpy.typing as npt

from aintelope.agents import PettingZooEnv, register_agent_class
from aintelope.agents.instincts.savanna_instincts import available_instincts_dict
from aintelope.agents.q_agent import HistoryStep, QAgent
from aintelope.environments.typing import ObservationFloat
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

    def reset(self, state) -> None:
        """Resets self and updates the state."""
        super().reset(state)
        self.init_instincts()

    def get_action(
        self,
        observation: npt.NDArray[ObservationFloat] = None,
        step: int = 0,  # net: nn.Module, epsilon: float, device: str
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
        return super().get_action(observation, step)

    # TODO hack, figure out if state_to_namedtuple can be static somewhere
    def update(
        self,
        env: PettingZooEnv = None,
        observation: npt.NDArray[ObservationFloat] = None,
        score: float = 0.0,
        done: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Takes observations and updates trainer on perceived experiences.
        Needed here to catch instincts.

        Args:
            observation: ObservationArray
            score: Only baseline uses score as a reward
            done: boolean whether run is done

        Returns:
            Reward: float
        """
        next_state = observation
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
                    instinct_reward, instinct_event = instinct_object.calc_reward(
                        next_state
                    )
                    reward += instinct_reward
                    logger.debug(
                        f"Reward of {instinct_name}: {instinct_reward}; "
                        f"total reward: {reward}"
                    )
                    if instinct_event != 0:
                        instinct_events.append((instinct_name, instinct_event))
        # interruption done

        if next_state is not None:
            next_s_hist = env.state_to_namedtuple(next_state.tolist())
        else:
            next_s_hist = None
        self.history.append(
            HistoryStep(
                state=env.state_to_namedtuple(self.state.tolist()),
                action=self.last_action,
                reward=reward,
                done=done,
                instinct_events=instinct_events,
                next_state=next_s_hist,
            )
        )

        if save_path is not None:
            with open(save_path, "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [
                        self.state.tolist(),
                        self.last_action,
                        score,
                        done,
                        instinct_events,
                        next_state,
                    ]
                )

        self.trainer.update_memory(
            self.id, self.state, self.last_action, score, done, next_state
        )
        self.state = next_state
        return reward

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


register_agent_class("instinct_agent", InstinctAgent)
