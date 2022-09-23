import random
import os
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
from aintelope.models.dqn import DQN
from aintelope.aintelope.environments.savanna_zoo import SavannaZooEnv


# is there a better way to do this?
# to register a lookup table from hparam name to function?
AGENT_LOOKUP = {
    "q_agent": Qagent,
    "shard_agent": ShardAgent,
    "random_walk_agent": RandomWalkAgent,
    "one_step_perfect_prediction_agent": OneStepPerfectPredictionAgent,
    "iterative_weight_optimization_agent": IterativeWeightOptimizationAgent,
}

ENV_LOOKUP = {"savanna_zoo_env": SavannaZooEnv}

MODEL_LOOKUP = {"dqn": DQN}


def run_episode(hparams: dict = {}):
    env_params = hparams.get("env_params", {})
    agent_params = hparams.get("agent_params", {})
    render_mode = hparams.get("render_mode")
    verbose = hparams.get("verbose", False)

    if env_params.get("env_type") == "zoo":
        env = ENV_LOOKUP[env_params["name"]](env_params=env_params)
        obs_size = env.observation_space.shape[0]
    elif env_params.get("env_type") == "gym":
        # GYM_INTERACTION
        if hparams.get("env_entry_point") is not None:
            gym.envs.register(
                id=env_params["name"],
                entry_point=hparams[
                    "env_entry_point"
                ],  # e.g. 'aintelope.environments.savanna_gym:SavannaGymEnv'
                kwargs={"env_params": env_params},
            )
        env = gym.make(hparams["env"])
        obs_size = env.observation_space.shape[0]
    else:
        print(
            f'env_type {env_params.get("env_type")} not implemented. Choose: [zoo, gym]'
        )

    env.reset()
    n_actions = env.action_space.n

    net = MODEL_LOOKUP[hparams["model"]](obs_size, n_actions)

    buffer = ReplayBuffer(hparams.replay_size)
    agent = AGENT_LOOKUP[hparams["agent"]](env, buffer, **agent_params)
    episode_reward = 0

    done = False
    warm_start_steps = hparams.get("warm_start_steps", 1000)
    for i in range(warm_start_steps):
        reward, done = agent.play_step(net, epsilon=1.0)
        if done:
            if verbose:
                print(
                    f"Uhoh! Your agent terminated during warmup on step {i}/{warm_start_steps}"
                )
            break

    while done is False:
        # step through environment with agent
        epsilon = max(
            hparams["eps_end"],
            hparams["eps_start"]
            - env.num_moves * 1 / hparams["eps_last_frame"],
        )

        reward, done = agent.play_step(
            net, epsilon, hparams.get("device", "cuda")
        )
        episode_reward += reward
        if render_mode is not None:
            env.render(render_mode)

    if verbose:
        print(
            f"Simple Episode Evaluation completed. Final episode reward: {episode_reward}"
        )
