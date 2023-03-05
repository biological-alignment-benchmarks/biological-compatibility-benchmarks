import typing as typ
import logging
import csv

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import gym
import numpy as np
import pandas as pd
import torch
from torch import nn

from aintelope.agents.memory import Experience, ReplayBuffer
from aintelope.agents.instincts.savanna_instincts import available_instincts_dict

logger = logging.getLogger("aintelope.agents.instinct_agent")


class InstinctAgent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(
        self,
        env: gym.Env,
        model,
        replay_buffer: ReplayBuffer,
        target_instincts: list = [],
    ) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.model = model
        self.replay_buffer = replay_buffer
        self.history = []
        self.target_instincts = target_instincts
        self.instincts = {}
        self.done = False
        self.reset()

    def init_instincts(self):
        logger.debug("debug target_instincts", self.target_instincts)
        for instinct_name in self.target_instincts:
            if instinct_name not in available_instincts_dict:
                logger.warning(
                    f"Warning: could not find {instinct_name} in available_instincts_dict"
                )
                continue

        self.instincts = {
            instinct: available_instincts_dict.get(instinct)()
            for instinct in self.target_instincts
            if instinct in available_instincts_dict
        }
        for instinct in self.instincts.values():
            instinct.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.done = False
        # GYM_INTERACTION
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        self.init_instincts()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            # GYM_INTERACTION
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(np.expand_dims(self.state, 0))

            if device not in ["cpu"]:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        epsilon: float = 0.0,
        device: str = "cpu",
        save_path: str = None,
    ) -> typ.Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the
        environment.

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        # The 'mind' of the agent decides what to do next
        action = self.get_action(net, epsilon, device)

        # you could optionally have a filter step here where the body'/'instincts'/'hindbrain'
        # can veto certain actions, for example stepping off a cliff
        # or trying to run fast despite a broken leg
        # this would be like Redwood Research's Harm/Failure Classifier
        body_veto = "stub"

        # do step in the environment
        # the environment reports the result of that decision
        new_state, env_reward, done, info = self.env.step(action)

        # we need a layer of body interpretation of the physical state of the environment
        # to track things like impact which can cause persistent injuries
        # or death (catatrophic failure of episode). Also, what physical inputs rise above
        # sense thresholds. How 'embodied' do we need to make our agent? Not sure. This
        # requires more thought and discussion.

        # the 'body'/'instincts'/'hindbrain' of the agent decides what reward the 'mind' should receive
        # based on the current and historical state reported by the environment
        # and also the 'state' that the agent receives, based on sense thresholds.
        if len(self.instincts) == 0:
            # use env reward as default
            instinct_events = []
            reward = env_reward
        else:
            # interpret new_state and env_reward to compute actual reward

            # state = [0] + [agent_x, agent_y] + [[1, x[0], x[1]] for x in self.grass_patches] + [[2, x[0], x[1]] for x in self.water_holes]
            reward = 0
            instinct_events = []
            for instinct_name, instinct_object in self.instincts.items():
                instinct_reward, instinct_event = instinct_object.calc_reward(
                    self, new_state
                )
                reward += instinct_reward
                if instinct_event != 0:
                    instinct_events.append((instinct_name, instinct_event))

        # the action taken, the environment's response, and the body's reward are all recorded together in memory
        exp = Experience(self.state, action, reward, done, new_state)
        self.history.append(
            (self.state.tolist(), action, reward, done, instinct_events, new_state)
        )

        if save_path is not None:
            with open(save_path, "a+") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(
                    [
                        self.state.tolist(),
                        action,
                        reward,
                        done,
                        instinct_events,
                        new_state,
                    ]
                )

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

    def plot_history(self) -> Figure:
        history_df = self.get_history()

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
            x.append(state[1])
            y.append(state[2])

            food_x.append(state[4])
            food_y.append(state[5])
            food_x.append(state[7])
            food_y.append(state[8])

            water_x.append(state[10])
            water_y.append(state[11])
            water_x.append(state[13])
            water_y.append(state[14])

            if row["instinct_events"] != "[]":
                event_x.append(x[-1])
                event_y.append(y[-1])
                event_type.append(row["instinct_events"])

        agent_df = pd.DataFrame(data={"x": x, "y": y})
        food_df = pd.DataFrame(data={"x": food_x, "y": food_y})
        water_df = pd.DataFrame(data={"x": water_x, "y": water_y})
        event_df = pd.DataFrame(
            data={"x": event_x, "y": event_y, "event_type": event_type}
        )

        fig, ax = plt.subplots()
        ax.plot(agent_df["x"], agent_df["y"], ".r-")
        ax.plot(food_df["x"], food_df["y"], ".g", markersize=15)
        ax.plot(water_df["x"], water_df["y"], ".b", markersize=15)
        plt.tight_layout()
        return fig
