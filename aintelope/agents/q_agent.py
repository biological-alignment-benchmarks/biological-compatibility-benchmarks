import csv
import logging
from typing import List, NamedTuple, Optional, Tuple
import numpy.typing as npt

from aintelope.agents import Agent, register_agent_class
from aintelope.environments.savanna_gym import SavannaGymEnv  # TODO used for hack
from aintelope.environments.typing import ObservationFloat
from aintelope.training.dqn_training import Trainer

logger = logging.getLogger("aintelope.agents.q_agent")


class HistoryStep(NamedTuple):
    state: NamedTuple
    action: int
    reward: float
    done: bool
    instinct_events: List[Tuple[str, int]]
    next_state: NamedTuple


class QAgent(Agent):
    """QAgent class, functioning as a base class for agents"""

    def __init__(
        self,
        agent_id: str,
        trainer: Trainer,
        target_instincts: List[str] = [],
    ) -> None:
        self.id = agent_id
        self.trainer = trainer
        self.history: List[HistoryStep] = []
        self.done = False
        self.last_action = 0

    def reset(self, state) -> None:
        """Resets self and updates the state."""
        self.done = False
        self.state = state
        if isinstance(self.state, tuple):
            self.state = self.state[0]

    def get_action(
        self,
        observation: npt.NDArray[ObservationFloat] = None,
        step: int = 0,  # net: nn.Module, epsilon: float, device: str
    ) -> Optional[int]:
        """Given an observation, ask your net what to do. State is needed to be given
        here as other agents have changed the state!

        Args:
            net: pytorch Module instance, the model
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action (Optional[int]): index of action
        """
        if self.done:
            return None
        else:
            # For future: observation can go to instincts here
            action = self.trainer.get_action(self.id, self.state, step)

        self.last_action = action
        return action

    # TODO hack, figure out if state_to_namedtuple can be static somewhere
    def update(
        self,
        env: SavannaGymEnv = None,
        observation: npt.NDArray[ObservationFloat] = None,
        score: float = 0.0,
        done: bool = False,
        save_path: Optional[str] = None,
    ) -> list:
        """
        Takes observations and updates trainer on perceived experiences.
        Needed here to catch instincts.

        Args:
            env: Environment
            observation: ObservationArray
            score: Only baseline uses score as a reward
            done: boolean whether run is done
            save_path: str
        Returns:
            agent_id (str): same as elsewhere ("agent_0" among them)
            state (npt.NDArray[ObservationFloat]): input for the net
            action (int): index of action
            reward (float): reward signal
            done (bool): if agent is done
            next_state (npt.NDArray[ObservationFloat]): input for the net
        """

        next_state = observation
        # For future: add state (interoception) handling here when needed

        if next_state is not None:
            next_s_hist = env.state_to_namedtuple(next_state.tolist())
        else:
            next_s_hist = None
        self.history.append(
            HistoryStep(
                state=env.state_to_namedtuple(self.state.tolist()),
                action=self.last_action,
                reward=score,
                done=done,
                instinct_events=[],
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
                        [],
                        next_state,
                    ]
                )
        event = [self.id, self.state, self.last_action, score, done, next_state]
        self.trainer.update_memory(*event)
        self.state = next_state

        return event


register_agent_class("q_agent", QAgent)
