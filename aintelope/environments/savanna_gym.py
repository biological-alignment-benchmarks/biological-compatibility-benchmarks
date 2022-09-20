from gym import spaces
import gym
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
from aintelope.environments.env_utils.distance import distance_to_closest_item

from savanna import (
    RenderSettings, 
    RenderState,
    HumanRenderState, 
    move_agent, 
    calc_grass_reward,
    reward_agent, 
    PositionFloat,
    Action
    )


class SavannaEnv(gym.Env):

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
        assert self.metadata['AMOUNT_AGENTS'] == 1, print(
            'agents must == 1 for gym env')
        self.action_space = Discrete(4)
        # observation space will be (object_type, pos_x, pos_y)
        self.observation_space = spaces.Box(
            low=self.metadata['MAP_MIN'],
            high=self.metadata['MAP_MAX'],
            shape=(
                3 * (self.metadata['AMOUNT_AGENTS'] + self.metadata['AMOUNT_GRASS_PATCHES'] + self.metadata['AMOUNT_WATER_HOLES']),)
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
            map_min=self.metadata['MAP_MIN'], map_max=self.metadata['MAP_MAX']
        )
        min_grass_distance = distance_to_closest_item(self.agent_state, self.grass_patches)
        reward = calc_grass_reward(min_grass_distance)
        if min_grass_distance < 1.0:
            self.grass_patches = self.replace_grass(
                self.agent_state, self.grass_patches)
        self.num_moves += 1
        done = self.num_moves >= self.metadata['NUM_ITERS']

        observation = self._get_obs()
        return observation, reward, done

    def reset(self):
        self.agent_state = self.np_random.integers(
            self.metadata['MAP_MIN'], self.metadata['MAP_MAX'], 2)
        self.grass_patches = self.np_random.integers(
            self.metadata['MAP_MIN'], self.metadata['MAP_MAX'], size=(
                self.metadata['AMOUNT_GRASS_PATCHES'], 2))
        self.water_holes = self.np_random.integers(
            self.metadata['MAP_MIN'], self.metadata['MAP_MAX'], size=(
                self.metadata['AMOUNT_WATER_HOLES'], 2))
        self.last_action = None
        self.num_moves = 0
        return self._get_obs()

    def _get_obs(self):
        observations = [0] + self.agent_state.tolist() 
        for x in self.grass_patches:
            observations += [1, x[0], x[1]]
        for x in self.water_holes:
            observations += [2, x[0], x[1]]
        return np.array(observations, dtype=np.float64)

    def replace_grass(self,
                      agent_pos: np.ndarray, grass_patches: np.ndarray
                      ) -> np.float64:
        if len(grass_patches.shape) == 1:
            grass_patches = np.expand_dims(grass_patches, 0)

        replacement_grass = self.np_random.integers(
            self.metadata['MAP_MIN'], self.metadata['MAP_MAX'], size=(2))
        grass_patches[
            np.argmin(
                np.linalg.norm(np.subtract(grass_patches, agent_pos), axis=1)
            )
        ] = replacement_grass

        return grass_patches


if __name__ == "__main__":
    env_params = {
        'NUM_ITERS': 500,  # duration of the game
        'MAP_MIN': 0,
        'MAP_MAX': 100,
        'render_map_max': 100,
        'AMOUNT_AGENTS': 1,  # for now only one agent
        'AMOUNT_GRASS_PATCHES': 2,
        'AMOUNT_WATER_HOLES': 2,
    }
    # e = raw_env(env_params=env_params)
    # print(type(e))
    # print(e)
    # print(e.__dict__)
    # ret = e.reset()
    # print(ret)

    # api_test(e, num_cycles=10, verbose_progress=True)
    # print(e.last())
