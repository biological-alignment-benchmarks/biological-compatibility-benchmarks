import functools
import typing as typ

import numpy as np
import pygame
from gym import spaces
from gym.spaces import Box, Discrete
from gym.utils import seeding
from aintelope.environments.env_utils.render_ascii import AsciiRenderState
from aintelope.environments.env_utils.distance import distance_to_closest_item

# typing aliases
PositionFloat = np.float32
Action = int

# environment constants
ACTION_MAP = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]], dtype=PositionFloat)


class RenderSettings:
    def __init__(self, metadata):
        prefix = "render_"
        settings = {
            (k.lstrip(prefix), v)
            for k, v in metadata.items()
            if k.startswith(prefix)
        }
        self.__dict__.update(settings)


class RenderState:
    def __init__(self, settings):
        canvas = pygame.Surface((settings.window_size, settings.window_size))
        self.canvas = canvas
        self.settings = settings

    def render(self, agents_state, grass):
        window_size = self.settings.window_size
        canvas = self.canvas

        canvas.fill((255, 255, 255))
        scale = window_size / self.settings.map_max

        screen_m = np.identity(2, dtype=PositionFloat) * scale

        def project(p):
            return np.matmul(p, screen_m).astype(np.int32)

        for gr in grass.reshape((2, -1)):
            p = project(gr)
            pygame.draw.circle(
                canvas,
                self.settings.grass_color,
                p,
                scale * self.settings.grass_radius,
            )

        for agent, agent_pos in agents_state.items():
            assert len(agent_pos) == 2, agent_pos
            # TODO: render agent name as text
            p = project(agent_pos)
            pygame.draw.circle(
                canvas,
                self.settings.agent_color,
                p,
                scale * self.settings.agent_radius,
            )


class HumanRenderState:
    def __init__(self, settings):

        self.fps = settings.fps

        window_size = settings.window_size

        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()

    def render(self, render_state):
        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(render_state.canvas, render_state.canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.fps)


def reward_agent(min_grass_distance):
    return 1 / (1 + min_grass_distance)


def move_agent(agent_pos: np.ndarray, action: Action, map_min=0, map_max=100) -> np.ndarray:
    move = ACTION_MAP[action]
    agent_pos = agent_pos + move
    agent_pos = np.clip(agent_pos, map_min, map_max)
    return agent_pos


def get_agent_pos_from_state(agent_state):
    """be sure to update this if you change
    the organization of the state to put the
    agent's position somewhere else.
    TODO: make unit test for this
    """
    return [agent_state[1], agent_state[2]]
    

class SavannaEnv():

    metadata = {
        "name": "savanna-v2",
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
        return observation, reward, done

    def reset(self):
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
            self.metadata['map_min'], self.metadata['map_max'], size=(2))
        grass_patches[
            np.argmin(
                np.linalg.norm(np.subtract(grass_patches, agent_pos), axis=1)
            )
        ] = replacement_grass

        return grass_patches

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



