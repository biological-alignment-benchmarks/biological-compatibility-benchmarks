from typing import Dict, List, Optional, NamedTuple, Tuple
import typing as typ
import logging
from pathlib import Path
from collections import OrderedDict
from collections import deque, namedtuple
import random

from omegaconf import DictConfig
import numpy.typing as npt
import numpy as np
import gymnasium as gym

import torch
from torch import Tensor, nn
import torch.optim as optim

from aintelope.models.dqn import DQN

# from aintelope.agents.memory import ReplayBuffer#, Experience

from aintelope.environments.typing import (
    ObservationFloat,
    PositionFloat,
    Action,
    AgentId,
    AgentStates,
    Observation,
    Reward,
    Info,
)

logger = logging.getLogger("aintelope.training.dqn_training")
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Trainer:
    def __init__(self, params, n_observations, action_space):
        self.policy_nets = {}
        self.target_nets = {}
        self.replay_memories = {}

        self.n_observations = n_observations
        self.action_space = action_space  # TODO add these to the config?
        self.hparams = params.hparams
        # tb-logging and device control, check lightning_Trainer for 'AVAIL'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.AdamW(
            DQN(self.n_observations, self.action_space.n).parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
        )  # refactor, making a dummy network now. problem is add_agent inits first real network---v

    def add_agent(self, agent_id):
        self.replay_memories[agent_id] = ReplayMemory(self.hparams.replay_size)
        self.policy_nets[agent_id] = DQN(self.n_observations, self.action_space.n).to(
            self.device
        )
        self.target_nets[agent_id] = DQN(self.n_observations, self.action_space.n).to(
            self.device
        )
        self.target_nets[agent_id].load_state_dict(
            self.policy_nets[agent_id].state_dict()
        )

    @torch.no_grad()  # TODO this might not be in the right place!
    def get_action(
        self,
        agent_id: str = "",
        observation: npt.NDArray[ObservationFloat] = None,
        step: int = 0,
    ) -> Optional[int]:
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - step * 1 / self.hparams.eps_last_frame,
        )
        if np.random.random() < epsilon:
            action = self.action_space.sample()
        else:
            logger.debug(
                "debug state", type(observation)
            )  # TODO figure out when obs becomes state
            observation = torch.tensor(np.expand_dims(observation, 0))
            logger.debug("debug state tensor", type(observation), observation.shape)

            if str(self.device) not in ["cpu"]:
                print(self.device not in ["cpu"])
                observation = observation.cuda(self.device)

            q_values = self.policy_nets[agent_id].net(observation)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    def update_memory(self, agent_id: str, state, action, reward, done, next_state):
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

    # TODO: is optimizer supposed to be done one step at a time?
    def optimize_models(self, step):
        for agent_id in self.policy_nets.keys():
            if len(self.replay_memories[agent_id]) < self.hparams.batch_size:
                return

            # transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            # batch = Transition(*zip(*transitions))
            transitions = self.replay_memories[agent_id].sample(self.hparams.batch_size)
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)
            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=self.device,
                dtype=torch.bool,
            )
            non_final_next_states = torch.cat(
                [s for s in batch.next_state if s is not None]
            )
            state_batch = torch.cat(
                batch.state
            )  # torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0) if bug
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(
                batch.reward
            )  # torch.tensor([reward], device=device) if bug

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            policy_net = self.policy_nets[agent_id]
            target_net = self.target_nets[agent_id]
            state_action_values = policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1)[0].
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(self.hparams.batch_size, device=self.device)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(
                    non_final_next_states
                ).max(1)[0]
            # Compute the expected Q values
            expected_state_action_values = (
                next_state_values * self.hparams.gamma
            ) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            self.optimizer.step()

            # Soft update of the target network's weights
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.hparams.tau + target_net_state_dict[key] * (
                    1 - self.hparams.tau
                )
            target_net.load_state_dict(target_net_state_dict)
