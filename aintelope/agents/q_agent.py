import typing as typ

import gym
import numpy as np
import torch
from torch import nn

from aintelope.agents.memory import Experience, ReplayBuffer


class Agent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env, model: nn.Module, replay_buffer: ReplayBuffer, name='agent_0') -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.name = name
        if isinstance(self.env, gym.Env):
            self.action_space = self.env.action_space
        else:
            self.action_space = self.env.action_space(self.name)
        self.model = model
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.done = False
        # GYM_INTERACTION
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]        

    def get_action(self, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        Args:
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if self.done:
            return None
        elif np.random.random() < epsilon:
            # GYM_INTERACTION
            action = self.action_space.sample()
        else:
            # TODO: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at  ../torch/csrc/utils/tensor_new.cpp:201.)
            # is state already a tensor here?
            # is it already a numpy array?
            print('debug state', type(self.state))
            state = torch.tensor([self.state])
            print('debug state tensor', type(self.state), state.shape)
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
        save_path: str = None
    ) -> typ.Tuple[float, bool]:
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