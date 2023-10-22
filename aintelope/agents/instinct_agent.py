from typing import Optional, List, Tuple
import logging
import csv

import numpy as np
import torch
from torch import nn

from aintelope.agents import Environment, register_agent_class, GymEnv, PettingZooEnv
from aintelope.agents.q_agent import QAgent, HistoryStep
from aintelope.agents.memory import Experience, ReplayBuffer
from aintelope.agents.instincts.savanna_instincts import available_instincts_dict

logger = logging.getLogger("aintelope.agents.instinct_agent")


class InstinctAgent(QAgent):
    """Agent class with instincts"""

    def __init__(
        self,
        env: Environment,
        model: nn.Module,
        replay_buffer: ReplayBuffer,
        warm_start_steps: int,
        target_instincts: List[str] = [],
    ) -> None:
        """
        Args:
            env (Environment): environment instance
            model (nn.Module): neural network instance
            replay_buffer (ReplayBuffer): replay buffer of the agent
            warm_start_steps (int): amount of initial random buffer
            target_instincts (List[str]): names if used instincts
        """
        self.target_instincts = target_instincts
        self.instincts = {}
        self.done = False

        # reset after attribute setup
        super().__init__(
            env=env,
            model=model,
            replay_buffer=replay_buffer,
            warm_start_steps=warm_start_steps,
        )

    def reset(self) -> None:
        """Reset environment and initialize instincts"""
        self.done = False
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        self.init_instincts()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """Using the given network, decide what action to carry out using an
        epsilon-greedy policy.

        Args:
            net (nn.Module): DQN network instance
            epsilon (float): value to determine likelihood of taking a random action
            device (str): current device

        Returns:
            action (int): index of action
        """
        if np.random.random() < epsilon:
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
        save_path: Optional[str] = None,
    ) -> Tuple[float, bool]:
        """Carries out a single interaction step between the agent and the
        environment.

        Args:
            net: DQN network instance
            epsilon: value to determine likelihood of taking a random action
            device: current device
            save_path (typ.Optional[str]): path to save agent history

        Returns:
            reward, done (Tuple[float, bool]): reward value and done state
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
        if isinstance(self.env, GymEnv):
            new_state, env_reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
        elif isinstance(self.env, PettingZooEnv):
            new_state, env_reward, terminateds, truncateds, info = self.env.step(action)
            done = {
                key: terminated or truncateds[key]
                for (key, terminated) in terminateds.items()
            }
        else:
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
            HistoryStep(
                state=self.env.state_to_namedtuple(self.state.tolist()),
                action=action,
                reward=reward,
                done=done,
                instinct_events=instinct_events,
                new_state=self.env.state_to_namedtuple(new_state.tolist()),
            )
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

    def init_instincts(self) -> None:
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


register_agent_class("instinct_agent", InstinctAgent)
