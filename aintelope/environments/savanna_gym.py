from gym import spaces
import gym
import functools
import typing as typ

import numpy as np
from gym.spaces import Box, Discrete
from gym.utils import seeding
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector, wrappers, parallel_to_aec
from aintelope.environments.env_utils.render_ascii import AsciiRenderState
from aintelope.environments.env_utils.distance import distance_to_closest_item

from aintelope.environments.savanna import (
    RenderSettings, 
    RenderState,
    move_agent, 
    reward_agent,
    PositionFloat,
    Action
    )


class SavannaGymEnv(gym.Env):

    metadata = {
        "name": "savanna-v2",
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
        assert self.metadata['amount_agents'] == 1, print(
            'agents must == 1 for gym env')
        self.action_space = Discrete(4)
        # observation space will be (object_type, pos_x, pos_y)
        self.observation_space = spaces.Box(
            low=self.metadata['map_min'],
            high=self.metadata['map_max'],
            shape=(
                3 * (self.metadata['amount_agents'] + self.metadata['amount_grass_patches'] + self.metadata['amount_water_holes']),)
        )
        self.agent_state = np.ndarray([])  # just the agents position for now
        self._seed()
        render_settings = RenderSettings(self.metadata)
        self.render_state = RenderState(render_settings)
        self.human_render_state = None
        self.ascii_render_state = None

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # costs = np.sum(u**2) + np.sum(self.state**2)
        # self.state = np.clip(
        #     self.state + u, self.observation_space.low, self.observation_space.high)
        self.last_action = action
        self.agent_state = move_agent(
            self.agent_state, action,
            map_min=self.metadata['map_min'], map_max=self.metadata['map_max']
        )
        
        
        min_grass_distance = distance_to_closest_item(self.agent_state, self.grass_patches)
        reward = reward_agent(min_grass_distance)
        if min_grass_distance < 1.0:
            self.grass_patches = self.replace_grass(
                self.agent_state, self.grass_patches)
        self.num_moves += 1
        done = self.num_moves >= self.metadata['num_iters']

        observation = self._get_obs()
        info = {"placeholder":"Placeholder because Nathan is confused here."}
        return observation, reward, done, info

    def reset(self, seed = None, options={}):
        self.agent_state = self.np_random.integers(
            self.metadata['map_min'], self.metadata['map_max'], 2)
        self.grass_patches = self.np_random.integers(
            self.metadata['map_min'], self.metadata['map_max'], size=(
                self.metadata['amount_grass_patches'], 2))
        self.water_holes = self.np_random.integers(
            self.metadata['map_min'], self.metadata['map_max'], size=(
                self.metadata['amount_water_holes'], 2))
        self.last_action = None
        self.num_moves = 0
        info = {"placeholder":"hmmm"}
        return (self._get_obs(), info)

    def _get_obs(self):
        observations = [0] + self.agent_state.tolist() 
        for x in self.grass_patches:
            observations += [1, x[0], x[1]]
        for x in self.water_holes:
            observations += [2, x[0], x[1]]
        return np.array(observations, dtype=np.float32)

    def replace_grass(self,
                      agent_pos: np.ndarray, grass_patches: np.ndarray
                      ) -> np.float32:
        if len(grass_patches.shape) == 1:
            grass_patches = np.expand_dims(grass_patches, 0)

        replacement_grass = self.np_random.integers(
            self.metadata['map_min'], self.metadata['map_max'], size=(2))
        grass_patches[
            np.argmin(
                np.linalg.norm(np.subtract(grass_patches, agent_pos), axis=1)
            )
        ] = replacement_grass

        return grass_patches
