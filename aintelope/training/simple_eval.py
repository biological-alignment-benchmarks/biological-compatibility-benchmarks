import random
from pprint import pprint
import numpy as np
from gym.spaces import Discrete
import gym
from aintelope.agents.q_agent import Agent as Qagent
from aintelope.agents.shard_agent import Agent as ShardAgent
from aintelope.agents.simple_agents import (
    RandomWalkAgent,
    OneStepPerfectPredictionAgent,
    IterativeWeightOptimizationAgent,
)
from aintelope.agents.memory import ReplayBuffer
from aintelope.aintelope.environments.savanna_zoo import (
    env,
    move_agent,
    reward_agent,
)

import os


def main(env):
    mode = "ascii"
    # policy = IterativeWeightOptimizationAgent()
    # policy = OneStepPerfectPredictionAgent()
    # policy = RandomWalkAgent()

    replay_size = 1000
    env.reset()
    replay_buffer = ReplayBuffer(replay_size)
    policy = Q_agent(gym.make(env), replay_buffer)
    for agent in env.agent_iter():
        observation, reward, done, info = env.last()

        action_space: Discrete = env.action_space(agent)
        if not done:
            action = policy(action_space, observation, reward, info)
            assert action in action_space
        else:
            action = None
        env.step(action)
        env.render(mode)

    # wait = input("Close?")


if __name__ == "__main__":
    env_params = {
        "NUM_ITERS": 40,  # duration of the game
        "MAP_MIN": 0,
        "MAP_MAX": 20,
        "render_map_max": 20,
        "AMOUNT_AGENTS": 1,  # for now only one agent
        "AMOUNT_GRASS_PATCHES": 2,
    }
    e = env(env_params=env_params)
    main(e)

# Local Variables:
# compile-command: "poetry run python simple.py"
# End:
