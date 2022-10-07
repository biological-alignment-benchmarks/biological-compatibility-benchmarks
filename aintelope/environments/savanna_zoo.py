import functools
import typing as typ

import numpy as np
import pygame
from gym.spaces import Box, Discrete
from gym.utils import seeding
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.test import api_test
from pettingzoo.utils import agent_selector, wrappers
from aintelope.environments.env_utils.render_ascii import AsciiRenderState
from aintelope.environments.env_utils.distance import distance_to_closest_item

from aintelope.environments.savanna import (
    RenderSettings, 
    RenderState,
    HumanRenderState, 
    move_agent, 
    reward_agent, 
    PositionFloat,
    Action
    )


class SavannaZooParallelEnv(ParallelEnv):

    metadata = {
        "name": "savanna_v1",
        "render_fps": 3,
        "render_agent_radius": 5,
        "render_agent_color": (200, 50, 0),
        "render_grass_radius": 5,
        "render_grass_color": (20, 200, 0),
        "render_modes": ("human", "ascii", "offline"),
        "render_window_size": 512,
        "amount_agents": 1,
        "map_min": 0,
        "map_max": 10,
        "amount_grass_patches": 2,
        "amount_water_holes": 0,
        "num_iters": 1
    }

    def __init__(self, env_params={}):
        self.metadata.update(env_params)
        print(f'initializing savanna env with params: {self.metadata}')
        self.possible_agents = [f"agent_{r}" for r in range(self.metadata['amount_agents'])]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(self.metadata['amount_agents'])))
        )

        self._action_spaces = {
            agent: Discrete(4) for agent in self.possible_agents
        }  # agents can walk in 4 directions
        self._observation_spaces = {
            agent: Box(
                self.metadata['map_min'],
                self.metadata['map_max'],
                shape=(2 * (self.metadata['amount_agents'] + self.metadata['amount_grass_patches']),),
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

    def reset(self, seed: typ.Optional[int] = None, options = None):
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
            self.metadata['map_min'], self.metadata['map_max'], size=(self.metadata['amount_grass_patches'], 2)
        ).astype(PositionFloat)
        self.agent_states = {
            agent: self.np_random.integers(self.metadata['map_min'], self.metadata['map_max'], 2).astype(
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
        dicts where each dict looks like {agent_1: action_of_agent_1, agent_2: action_of_agent_2}
        or generally {<agent_name>: <agent_action or None if agent is done>}
        """  
        print('debug actions', actions)
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}

        if self.agents == []:
            raise ValueError("No agents found; num_iters reached?")
        
        rewards = {}
        for agent in self.agents:
            action = actions.get(agent)    
            if isinstance(action, dict):
                action = action.get(agent)
            if action is None:
                continue
            
            print('debug action', action)
            self.agent_states[agent] = move_agent(
                self.agent_states[agent], action,
                map_min=self.metadata['map_min'], map_max=self.metadata['map_max']
            )
            min_grass_distance = distance_to_closest_item(self.agent_states[agent], self.grass_patches)
            rewards[agent] = reward_agent(min_grass_distance)

        self.num_moves += 1
        env_done = (self.num_moves >= self.metadata['num_iters']) or all([actions.get(agent) is None for agent in self.agents])
        self.dones = {agent: env_done or (actions.get(agent) is None) for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_done:
            self.agents = []
        print('debug return', observations, rewards, self.dones, infos)
        return observations, rewards, self.dones, infos

class SavannaZooSequentialEnv(SavannaZooParallelEnv, AECEnv):
    pass