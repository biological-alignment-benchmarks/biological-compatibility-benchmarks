import logging


import numpy as np

# Code to enable Visual Studio debugging with non-legacy-debugger.
# If you want to avoid exceptions during code start altogether then an alternative is to use "legacy debugger":
# Visual Studio -> Tools -> Options -> Python -> Debugging -> Enable "Use legacy debugger".
import os
import sys

if os.name == "nt" and sys.gettrace() is not None:  # detect debugging under Windows
    original_get_terminal_size = os.get_terminal_size

    def get_terminal_size(fd=1):  # 1 = STDOUT_FILENO
        try:
            return original_get_terminal_size(fd)
        except OSError:  # "[WinError 6] The handle is invalid"
            return os.terminal_size([80, 24])

    os.get_terminal_size = get_terminal_size

from gemini import Scene, Sprite, txtcolours as tc, sleep

logger = logging.getLogger("aintelope.environments.env_utils.render_ascii")

# typing aliases
PositionFloat = np.float32
Action = int


class AsciiRenderState:
    def __init__(self, agent_states, grass_patches, settings):
        self.window_size = settings.map_max
        self.fps = settings.fps
        self.scene = Scene(
            (self.window_size, self.window_size), is_main_scene=True, clear_char="."
        )

        self.ascii_symbols = {
            "agent": [str(x) for x in range(10)],
            "food": "f",
            "barrier": "#",
            "door": "+",
            "civilian": "abcde".split(),
            "enemy": "x",
        }
        self.agent_sprites = {}
        self.agent_prev_pos = {}
        count = 0
        for agent_name, agent_pos in agent_states.items():
            agent_image = """x""".replace(
                "x",
                self.ascii_symbols["agent"][count % len(self.ascii_symbols["agent"])],
            )
            agent_sprite = Sprite(
                (agent_pos[0], self.window_size - agent_pos[1]), agent_image, layer=1
            )
            self.agent_sprites[agent_name] = agent_sprite
            self.agent_prev_pos[agent_name] = agent_pos
            count += 1

        self.grass_sprites = []
        for grass_pos in grass_patches:  # np.argwhere(grass_patches==1):
            logger.debug("debug init grass", grass_pos, grass_patches)
            grass_image = """x""".replace("x", self.ascii_symbols["food"])
            grass_sprite = Sprite(
                (grass_pos[0], self.window_size - grass_pos[1]),
                grass_image,
                colour=tc.GREEN,
                layer=2,
            )
            self.grass_sprites.append(grass_sprite)
        # [tc.RED, tc.YELLOW, tc.CYAN, tc.GREEN]
        # scene =
        # car = Sprite((5,5), car_image)
        # while True:
        # 	scene.render()
        # 	car.move(1,0)
        # 	sleep(.1)

        self.steps = 0

    def get_ascii_pos(self, position_array):
        x = position_array[0] // self.window_size
        y = position_array[1] // self.window_size
        return (x, y)

    def render(self, agent_states, grass_patches):
        for agent_name, agent_pos in agent_states.items():
            # Move the Entity within the scene. `+x` is right and `+y` is down.
            prev_pos = self.agent_prev_pos[agent_name]
            move_x = agent_pos[0] - prev_pos[0]
            move_y = (self.window_size - agent_pos[1]) - (
                self.window_size - prev_pos[1]
            )
            agent_movement = (move_x, move_y)
            # collisions handled elsewhere, don't have rendering engine do that
            self.agent_sprites[agent_name].move(agent_movement, collide=False)

        for x in self.grass_sprites:
            x = None
        self.grass_sprites = []
        for grass_pos in np.argwhere(grass_patches == 1):
            grass_image = """x""".replace("x", self.ascii_symbols["food"])
            grass_sprite = Sprite(
                (grass_pos[0], grass_pos[1]), grass_image, colour=tc.GREEN, layer=2
            )
            self.grass_sprites.append(grass_sprite)

        self.scene.render()
        self.steps += 1
        sleep(1.0 / self.fps)
