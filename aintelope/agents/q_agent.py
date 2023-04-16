from typing import Optional, Tuple
import logging

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import torch
from torch import nn

from aintelope.agents import (
    Agent,
    GymEnv,
    PettingZooEnv,
    Environment,
    register_agent_class,
)
from aintelope.agents.memory import Experience, ReplayBuffer


logger = logging.getLogger("aintelope.agents.q_agent")


class QAgent(Agent):
    """QAgent class, functioning as a base class for agents"""

    def __init__(
        self, env: Environment, model: nn.Module, replay_buffer: ReplayBuffer
    ) -> None:
        self.env = env
        if isinstance(env, GymEnv):
            self.action_space = self.env.action_space
        elif isinstance(env, PettingZooEnv):
            self.action_space = self.env.action_space("agent0")
        else:
            raise TypeError(f"{type(env)} is not a valid environment")
        self.model = model
        self.replay_buffer = replay_buffer
        self.history = []
        self.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.done = False
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]

    def get_action(self, epsilon: float, device: str) -> Optional[int]:
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        Args:
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action (Optional[int]): index of action
        """
        if self.done:
            return None
        elif np.random.random() < epsilon:
            action = self.action_space.sample()
        else:
            logger.debug("debug state", type(self.state))
            state = torch.tensor(np.expand_dims(self.state, 0))
            logger.debug("debug state tensor", type(self.state), state.shape)
            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = self.model(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
        save_path: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """
        Only for Gym envs, not PettingZoo envs
        Carries out a single interaction step between the agent and the
        environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        # The 'mind' (model) of the agent decides what to do next
        action = self.get_action(epsilon, device)

        # do step in the environment
        # the environment reports the result of that decision
        new_state, reward, done, info = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)
        self.state = new_state

        # if scenario is complete or agent experiences catastrophic failure, end the agent.
        if done:
            self.reset()

        return reward, done

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "state",
                "action",
                "reward",
                "done",
                "instinct_events",
                "new_state",
            ],
            data=self.history,
        )

    @staticmethod
    def process_history(
        history_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x = []
        y = []
        event_x = []
        event_y = []
        event_type = []
        food_x = []
        food_y = []
        water_x = []
        water_y = []

        for _, row in history_df.iterrows():
            state = row["state"]
            # fmt: off
            i = 1
            x.append(state[i]); i += 1
            y.append(state[i]); i += 1
            i += 1

            food_x.append(state[i]); i += 1
            food_y.append(state[i]); i += 1
            i += 1
            food_x.append(state[i]); i += 1
            food_y.append(state[i]); i += 1
            i += 1

            water_x.append(state[i]); i += 1
            water_y.append(state[i]); i += 1
            i += 1
            water_x.append(state[i]); i += 1
            water_y.append(state[i]); i += 1
            # fmt: on

            if row["instinct_events"] != "[]":
                event_x.append(x[-1])
                event_y.append(y[-1])
                event_type.append(row["instinct_events"])
            assert i == len(state)

        agent_df = pd.DataFrame(data={"x": x, "y": y})
        food_df = pd.DataFrame(data={"x": food_x, "y": food_y})
        water_df = pd.DataFrame(data={"x": water_x, "y": water_y})
        event_df = pd.DataFrame(
            data={"x": event_x, "y": event_y, "event_type": event_type}
        )
        return agent_df, food_df, water_df, event_df

    def plot_history(self, style: str = "thickness", color: str = "viridis") -> Figure:
        history_df = self.get_history()
        agent_df, food_df, water_df, event_df = self.process_history(history_df)

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
