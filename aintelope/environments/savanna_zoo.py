import functools
import typing as typ

import numpy as np
import pygame
from gym.spaces import Box, Discrete
from gym.utils import seeding
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.test import api_test
from pettingzoo.utils import agent_selector, wrappers, parallel_to_aec
from aintelope.environments.env_utils.render_ascii import AsciiRenderState

from savanna import (
    RenderSettings, 
    RenderState,
    HumanRenderState, 
    move_agent, 
    reward_agent, 
    PositionFloat,
    Action
    )


class RawEnv(ParallelEnv):

    metadata = {
        "name": "savanna_v1",
        "render_fps": 3,
        "render_agent_radius": 5,
        "render_agent_color": (200, 50, 0),
        "render_grass_radius": 5,
        "render_grass_color": (20, 200, 0),
        "render_modes": ("human", "ascii", "offline"),
        "render_window_size": 512,
    }

    def __init__(self, env_params={}):
        self.metadata.update(env_params)
        print(f'initializing savanna env with params: {self.metadata}')
        self.possible_agents = [f"agent_{r}" for r in range(self.metadata['AMOUNT_AGENTS'])]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.metadata['AMOUNT_AGENTS'])))
        )

        self._action_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }  # agents can walk in 4 directions
        self._observation_spaces = {
            agent: Box(
                self.metadata['MAP_MIN'],
                self.metadata['MAP_MAX'],
                shape=(2 * (self.metadata['AMOUNT_AGENTS'] + self.metadata['AMOUNT_GRASS_PATCHES']),),
            )
            for agent in self.possible_agents
        }

        render_settings = RenderSettings(self.metadata)
        self.render_state = RenderState(render_settings)
        self.human_render_state = None
        self.ascii_render_state = None
        self.dones = None
        self.seed()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        return self._action_spaces[agent]

    def seed(self, seed: typ.Optional[int] = None) -> None:
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent: str):
        """Return observation of given agent."""
        return np.concatenate(
            [self.agent_states[agent], self.grass_patches.reshape(-1)]
        )

    def render(self, mode="human"):
        """Render the environment."""

        self.render_state.render(self.agent_states, self.grass_patches)

        if mode == "human":
            if not self.human_render_state:
                self.human_render_state = HumanRenderState(
                    self.render_state.settings
                )
            self.human_render_state.render(self.render_state)
        elif mode == "ascii":
            if not self.ascii_render_state:
                self.ascii_render_state = AsciiRenderState(
                    self.agent_states, 
                    self.grass_patches,
                    self.render_state.settings
                )
            self.ascii_render_state.render(
                    self.agent_states, 
                    self.grass_patches
                    )
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.render_state.canvas)),
                axes=(1, 0, 2),
            )

    def close(self):
        """Release any graphical display, subprocesses, network connections
        or any other environment data which should not be kept around after
        the user is no longer using the environment.
        """
        raise NotImplementedError

    def reset(self, seed: typ.Optional[int] = None):
        """Reset needs to initialize the following attributes:
            - agents
            - rewards
            - _cumulative_rewards
            - dones
            - infos
            - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        """
        if seed is not None:
            self.seed(seed)

        self.agents = self.possible_agents[:]
        # self.rewards = {agent: 0 for agent in self.agents}
        # self._cumulative_rewards = {agent: 0 for agent in self.agents}
        # self.dones = {agent: False for agent in self.agents}
        # self.infos = {agent: {} for agent in self.agents}
        self.grass_patches = self.np_random.integers(
            self.metadata['MAP_MIN'], self.metadata['MAP_MAX'], size=(self.metadata['AMOUNT_GRASS_PATCHES'], 2)
        ).astype(PositionFloat)
        self.agent_states = {
            agent: self.np_random.integers(self.metadata['MAP_MIN'], self.metadata['MAP_MAX'], 2).astype(
                PositionFloat
            )
            for agent in self.agents
        }
        self.num_moves = 0

        # cycle through the agents; needed for wrapper
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.dones = {agent: False for agent in self.agents}
        observations = {agent: self.observe(agent) for agent in self.agents}
        return observations

    def step(self, actions: typ.Dict[str, Action]):
        """step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - dones
        - info
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """  # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        if self.agents == []:
            raise ValueError("No agents found; NUM_ITERS reached?")

        for agent in self.agents:
            self.agent_states[agent] = move_agent(
                self.agent_states[agent], actions[agent],
                map_min=self.metadata['MAP_MIN'], map_max=self.metadata['MAP_MAX']
            )
        rewards = {
            agent: reward_agent(self.agent_states[agent], self.grass_patches)
            for agent in self.agents
        }

        self.num_moves += 1
        env_done = self.num_moves >= self.metadata['NUM_ITERS']
        self.dones = {agent: env_done for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []

        return observations, rewards, self.dones, infos


def env(env_params={}):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env_obj = RawEnv(env_params=env_params)
    env_obj = parallel_to_aec(env_obj)
    return env_obj


# def env(env_params={}):
#     """Add PettingZoo wrappers to environment class."""
#     env = RawEnv(env_params=env_params)
#     # BaseWrapper class need agent_selection attribute
#     # env = wrappers.AssertOutOfBoundsWrapper(env)
#     # env = wrappers.OrderEnforcingWrapper(env)
#     return env


if __name__ == "__main__":
    env_params = {
        'NUM_ITERS':500,  # duration of the game
        'MAP_MIN':0, 
        'MAP_MAX':100,
        'render_map_max':100,
        'AMOUNT_AGENTS':1,  # for now only one agent
        'AMOUNT_GRASS_PATCHES':2
                }
    e = raw_env(env_params=env_params)
    print(type(e))
    print(e)
    print(e.__dict__)
    ret = e.reset()
    print(ret)

    api_test(e, num_cycles=10, verbose_progress=True)
    print(e.last())
