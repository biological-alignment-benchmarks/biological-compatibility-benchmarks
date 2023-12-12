from typing import Optional
import logging
from collections import deque, namedtuple
import random

import numpy.typing as npt
import numpy as np

import torch
from torch import nn
import torch.optim as optim

from aintelope.models.dqn import DQN

from aintelope.environments.typing import ObservationFloat

logger = logging.getLogger("aintelope.training.dqn_training")
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class ReplayMemory(object):
    """
    Replay memory for each agent, saves transitions (from RL literature).
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def load_checkpoint(PATH, obs_size, action_space):
    """
    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    Load a model from a checkpoint. Commented parts optional for later.

    Args:
        path: str
        obs_size: int, input size
        action_space: int, output size

    Returns:
        model: torch.nn.Module
    """

    model = DQN(obs_size, action_space)
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # loss = checkpoint['loss']

    model.eval()
    return model


class Trainer:
    """
    Trainer class, entry point to all things pytorch. Init a single instance for
    handling the models, register agents in for their personal models.
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, params, n_observations, action_space):
        self.policy_nets = {}
        self.target_nets = {}
        self.losses = {}
        self.replay_memories = {}

        self.n_observations = n_observations
        self.action_space = action_space
        self.hparams = params.hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.AdamW(
            DQN(self.n_observations, self.action_space.n).parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
        )

    def add_agent(self, agent_id):
        """
        Register an agent.

        Args:
            agent_id: str, same as elsewhere ("agent_0" among them)

        Returns:
            None
        """
        self.replay_memories[agent_id] = ReplayMemory(self.hparams.replay_size)
        self.policy_nets[agent_id] = DQN(
            self.n_observations, self.action_space(agent_id).n
        ).to(self.device)
        self.target_nets[agent_id] = DQN(
            self.n_observations, self.action_space(agent_id).n
        ).to(self.device)
        self.target_nets[agent_id].load_state_dict(
            self.policy_nets[agent_id].state_dict()
        )

    @torch.no_grad()
    def get_action(
        self,
        agent_id: str = "",
        observation: npt.NDArray[ObservationFloat] = None,
        step: int = 0,
    ) -> Optional[int]:
        """
        Get action from an agent

        Args:
            agent_id: str, same as elsewhere ("agent_0" among them)
            observation: npt.NDArray[ObservationFloat], input for the net
            step: int, used to calculate epsilon

        Returns:
            None
        """
        if step > 0:
            epsilon = max(
                self.hparams.eps_end,
                self.hparams.eps_start - step * 1 / self.hparams.eps_last_frame,
            )
        else:
            epsilon = 0.0

        if np.random.random() < epsilon:
            action = self.action_space(agent_id).sample()
        else:
            logger.debug("debug observation", type(observation))
            observation = torch.tensor(np.expand_dims(observation, 0))
            logger.debug(
                "debug observation tensor", type(observation), observation.shape
            )

            if str(self.device) not in ["cpu"]:
                print(self.device not in ["cpu"])
                observation = observation.cuda(self.device)

            q_values = self.policy_nets[agent_id].net(observation)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    def update_memory(self, agent_id: str, state, action, reward, done, next_state):
        """
        Add transition into agent specific ReplayMemory.

        Args:
            agent_id: str, same as elsewhere ("agent_0" among them)
            state: npt.NDArray[ObservationFloat], input for the net
            action: int, index of action
            reward: float, reward signal
            done: bool, if agent is done
            next_state: npt.NDArray[ObservationFloat], input for the net

        Returns:
            None
        """
        # add experience to torch device if bugged
        if done:
            return
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        action = torch.tensor(action, device=self.device).unsqueeze(0).view(1, 1)
        reward = torch.tensor(
            reward, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        next_state = torch.tensor(
            next_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.replay_memories[agent_id].push(state, action, reward, done, next_state)

    def optimize_models(self, step):
        """
        Optimize personal models based on contents of ReplayMemory of each agent.
        Check: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        Args:
            None

        Returns:
            None
        """
        for agent_id in self.policy_nets.keys():
            if len(self.replay_memories[agent_id]) < self.hparams.batch_size:
                return

            transitions = self.replay_memories[agent_id].sample(self.hparams.batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=self.device,
                dtype=torch.bool,
            )
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            )
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            policy_net = self.policy_nets[agent_id]
            target_net = self.target_nets[agent_id]
            state_action_values = policy_net(state_batch).gather(1, action_batch.long())

            next_state_values = torch.zeros(self.hparams.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(
                    non_final_next_states
                ).max(1)[0]

            expected_state_action_values = (
                next_state_values * self.hparams.gamma
            ) + reward_batch

            criterion = nn.SmoothL1Loss()
            loss = criterion(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )
            self.losses[agent_id] = loss

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            self.optimizer.step()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.hparams.tau + target_net_state_dict[key] * (
                    1 - self.hparams.tau
                )
            target_net.load_state_dict(target_net_state_dict)

    def save_models(self, episode, path):
        """
        Save models to the given directory.

        Args:
            episode: point in training, used by torch
            path: path defined outside

        Returns:
            None
        """
        for agent_id in self.policy_nets.keys():
            model = self.policy_nets[agent_id]
            loss = 1.0
            if agent_id in self.losses:
                loss = self.losses[agent_id]
            torch.save(
                {
                    "epoch": episode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "loss": loss,
                },
                path + agent_id + "_" + str(episode),
            )
