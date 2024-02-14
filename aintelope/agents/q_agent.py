import csv
import logging
from typing import List, NamedTuple, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.figure import Figure

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
    ) -> None:
        """
        Takes observations and updates trainer on perceived experiences.
        Needed here to catch instincts.

        Args:
            observation: ObservationArray
            score: Only baseline uses score as a reward
            done: boolean whether run is done

        Returns:
            reward: float
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

        self.trainer.update_memory(
            self.id, self.state, self.last_action, score, done, next_state
        )
        self.state = next_state
        return score

    def get_history(self) -> pd.DataFrame:
        """
        Method to get the history of the agent. Note that warm_start_steps are excluded.
        warm_start_steps not used atm.
        """
        return pd.DataFrame(
            columns=[
                "state",
                "action",
                "reward",
                "done",
                "instinct_events",
                "next_state",
            ],
            data=self.history,  # self.warm_start_steps :],
        )

    @staticmethod
    def process_history(
        history_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Function to convert the agent history dataframe into individual dataframe for
        agent position, grass and water locations. Instinct events are currently not
        processed.
        """
        state_df = pd.DataFrame(history_df.state.to_list())
        agent_df = pd.DataFrame(
            columns=["x", "y"], data=state_df.agent_coords.to_list()
        )
        grass_columns = [c for c in list(state_df) if c.startswith("grass")]
        grass_df = state_df[grass_columns].applymap(lambda x: tuple(x))
        grass_df = pd.DataFrame(
            columns=["x", "y"], data=set(grass_df.stack().to_list())
        )
        water_columns = [c for c in list(state_df) if c.startswith("water")]
        water_df = state_df[water_columns].applymap(lambda x: tuple(x))
        water_df = pd.DataFrame(
            columns=["x", "y"], data=set(water_df.stack().to_list())
        )

        return agent_df, grass_df, water_df

    def plot_history(self, style: str = "thickness", color: str = "viridis") -> Figure:
        history_df = self.get_history()
        agent_df, food_df, water_df = self.process_history(history_df)

        fig, ax = plt.subplots(figsize=(8, 8))

        if style == "thickness":
            ax.plot(agent_df["x"], agent_df["y"], ".r-")
        elif style == "colormap":
            cmap = matplotlib.colormaps[color]

            agent_arr = agent_df.to_numpy()  # coordinates x y
            # coordinates are ordered in x1 y1 x2 y2
            step_pairs = np.concatenate([agent_arr[:-1], agent_arr[1:]], axis=1)
            unique_steps, step_freq = np.unique(step_pairs, axis=0, return_counts=True)

            for line_segment, col in zip(unique_steps, step_freq / step_freq.max()):
                if (line_segment[:2] == line_segment[2:]).all():  # agent did not move
                    im = ax.scatter(
                        line_segment[0],
                        line_segment[1],
                        s=70,
                        marker="o",
                        color=cmap(col),
                    )
                else:
                    ax.plot(line_segment[[0, 2]], line_segment[[1, 3]], color=cmap(col))

            cbar = fig.colorbar(im)
            cbar.set_label("Relative Frequency Agent")
        else:
            raise NotImplementedError(f"{style} is not a valid plot style!")

        ax.plot(food_df["x"], food_df["y"], "xg", markersize=8, label="Food")
        ax.plot(water_df["x"], water_df["y"], "xb", markersize=8, label="Water")
        ax.legend()
        plt.tight_layout()
        return fig


register_agent_class("q_agent", QAgent)
