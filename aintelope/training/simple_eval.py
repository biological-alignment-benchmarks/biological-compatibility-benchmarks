import random
import os
from pprint import pprint
import numpy as np
from gym.spaces import Discrete
import gym
from pettingzoo.utils import parallel_to_aec
from aintelope.agents.q_agent import Agent as Qagent
from aintelope.agents.shard_agent import ShardAgent
from aintelope.agents.simple_agents import (
    RandomWalkAgent,
    OneStepPerfectPredictionAgent,
    IterativeWeightOptimizationAgent,
)
from aintelope.agents.memory import ReplayBuffer
from aintelope.models.dqn import DQN
from aintelope.environments.savanna_zoo import SavannaZooEnv


# is there a better way to do this?
# to register a lookup table from hparam name to function?
AGENT_LOOKUP = {
    "q_agent": Qagent,
    "shard_agent": ShardAgent,
    "random_walk_agent": RandomWalkAgent,
    "one_step_perfect_prediction_agent": OneStepPerfectPredictionAgent,
    "iterative_weight_optimization_agent": IterativeWeightOptimizationAgent,
}

ENV_LOOKUP = {"savanna-zoo-v2": SavannaZooEnv}

MODEL_LOOKUP = {"dqn": DQN}


def run_episode(hparams: dict = {}):
    env_params = hparams.get("env_params", {})
    agent_params = hparams.get("agent_params", {})
    render_mode = hparams.get("render_mode")
    verbose = hparams.get("verbose", False)

    if hparams.get("env_type") == "zoo":
        env = ENV_LOOKUP[hparams["env"]](env_params=env_params)
        if hparams.get('sequential_env', False) is True:
            env = parallel_to_aec(env)
        # assumption here: all agents in zoo have same observation space shape
        
        obs_size = list(env.observation_spaces.values())[0].shape[0]
        print(env.action_spaces)
        n_actions = list(env.action_spaces.values())[0].n
    elif hparams.get("env_type") == "gym":
        # GYM_INTERACTION
        if hparams.get("env_entry_point") is not None:
            gym.envs.register(
                id=hparams["env"],
                entry_point=hparams[
                    "env_entry_point"
                ],  # e.g. 'aintelope.environments.savanna_gym:SavannaGymEnv'
                kwargs={"env_params": env_params},
            )
        env = gym.make(hparams["env"])
        obs_size = env.observation_space.shape[0]
        n_actions = env.action_space.n
    else:
        print(
            f'env_type {hparams.get("env_type")} not implemented. Choose: [zoo, gym]'
        )

    env.reset()
    
    buffer = ReplayBuffer(hparams['replay_size'])
    
    agent_spec = hparams["agent"]
    if isinstance(agent_spec, list):
        agents = [AGENT_LOOKUP[agent](env, buffer, **agent_params) for agent in agent_spec]
    else:
        agents = [AGENT_LOOKUP[agent_spec](env, buffer, **agent_params)]
    
    model_spec = hparams["model"]
    if isinstance(model_spec, list):
        nets = [MODEL_LOOKUP[net](obs_size, n_actions) for net in model_spec]
    else:
        nets = [MODEL_LOOKUP[model_spec](obs_size, n_actions)]
    
    
    episode_rewards = [0 for x in agents]
    dones = [False for x in agents]
    warm_start_steps = hparams.get("warm_start_steps", 1000)
    
    for i, agent in enumerate(agents):
        for step in range(warm_start_steps):
            reward, done = agent.play_step(nets[i], epsilon=1.0)
            dones[i] = done
            if done:
                if verbose:
                    print(
                        f"Uhoh! Your agent {agent} terminated during warmup on step {step}/{warm_start_steps}"
                    )
                break

    while not all(dones):
        for i, agent in enumerate(agents):
            # step through environment with agent
            epsilon = max(
                hparams["eps_end"],
                hparams["eps_start"]
                - env.num_moves * 1 / hparams["eps_last_frame"],
            )

            reward, done = agent.play_step(
                nets[i], epsilon, hparams.get("device", "cuda")
            )
            dones[i] = done
            episode_rewards[i] += reward
            if render_mode is not None:
                env.render(render_mode)

    if verbose:
        print(
            f"Simple Episode Evaluation completed. Final episode reward: {episode_rewards}"
        )
