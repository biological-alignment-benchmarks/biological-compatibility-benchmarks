import typing as typ

import gym
import numpy as np
import torch
from torch import nn
import csv

from aintelope.agents.memory import Experience, ReplayBuffer
from aintelope.agents.shards.savanna_shards import available_shards_dict


class ShardAgent:
    """Base Agent class handeling the interaction with the environment."""

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer, target_shards: list = []) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.target_shards = target_shards
        self.shards = {}
        self.reset()

    def init_shards(self):
        self.shards = {shard : available_shards_dict.get(shard)() for shard in self.target_shards if shard in available_shards_dict}
        print('target_shards', self.target_shards)
        for shard in self.shards.values():
            shard.reset()

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        # GYM_INTERACTION
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]  
        self.init_shards()     

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
            # TODO: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at  ../torch/csrc/utils/tensor_new.cpp:201.)
            
            state = torch.tensor([self.state])

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
        save_path: str = None
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
        body_veto = 'stub'
        
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
        if len(self.shards) == 0:
            # use env reward as default
            shard_events = []
            reward = env_reward
        else:
            # interpret new_state and env_reward to compute actual reward
            
            
            # state = [0] + [agent_x, agent_y] + [[1, x[0], x[1]] for x in self.grass_patches] + [[2, x[0], x[1]] for x in self.water_holes]
            reward = 0
            shard_events = []
            for shard_name, shard_object in self.shards.items():
                shard_reward, shard_event = shard_object.calc_reward(self, new_state)
                reward += shard_reward
                if shard_event != 0:
                    shard_events.append((shard_name, shard_event))



        # the action taken, the environment's response, and the body's reward are all recorded together in memory
        exp = Experience(self.state, action, reward, done, new_state)
        if save_path is not None:
            with open(save_path, 'a+') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([self.state.tolist(),action,reward,done,shard_events,new_state])
                
        self.replay_buffer.append(exp)
        self.state = new_state
        
        
        # if scenario is complete or agent experiences catastrophic failure, end the agent.
        if done:
            self.reset()
        return reward, done
