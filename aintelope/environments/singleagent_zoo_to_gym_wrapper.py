# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv


class SingleAgentZooToGymWrapper(gym.Env):
    """
    A wrapper that transforms a PettingZoo environment with exactly one agent
    into a single-agent Gymnasium environment.
    Both Zoo ParallelEnv and Zoo AECEnv (sequential env) are supported.
    """

    def __init__(self, zoo_env):
        super().__init__()

        assert zoo_env.num_agents == 1

        single_agent_name = zoo_env.agents[0]

        self.env = zoo_env
        self.agent_name = single_agent_name

        self.observation_space = self.env.observation_spaces[self.agent_name]
        self.action_space = self.env.action_spaces[self.agent_name]

    def reset(self, seed=None, options=None, *args, **kwargs):
        """
        Reset the environment.
        Return: initial observation and an info dict
        """

        # Normally AEC env reset() method does not provide observations and infos as a return value, but the savanna_safetygrid wrapper adds this capability
        if seed is not None:
            (observations, infos) = self.env.reset(
                seed=seed, options=options, *args, **kwargs
            )
        else:
            (observations, infos) = self.env.reset(options=options, *args, **kwargs)

        observation = observations[self.agent_name]
        info = infos[self.agent_name]

        return observation, info

    def step(self, action):
        """
        Take one step in the environment using the provided action.
        Return: observation, reward, done, truncated, info
        """
        if isinstance(self.env, ParallelEnv):
            actions = {self.agent_name: action}

            observations, rewards, terminations, truncations, infos = self.env.step(
                actions
            )

            observation = observations[self.agent_name]
            reward = rewards[self.agent_name]
            terminated = terminations[self.agent_name]
            truncated = truncations[self.agent_name]
            info = infos[self.agent_name]
        elif isinstance(self.env, AECEnv):
            # Normally AEC env step() method does not provide observations and infos as a return value, but the savanna_safetygrid wrapper adds this capability via step_single_agent() method
            (
                observation,
                reward,
                terminated,
                truncated,
                info,
            ) = self.env.step_single_agent(action)

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """
        Render the environment if supported by the underlying environment.
        """
        return self.env.render(mode=mode)

    def close(self):
        """
        Close the environment.
        """
        self.env.close()
