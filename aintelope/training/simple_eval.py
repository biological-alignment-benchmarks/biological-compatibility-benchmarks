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
from aintelope.environments.savanna_zoo import SavannaZooParallelEnv, SavannaZooSequentialEnv


# is there a better way to do this?
# to register a lookup table from hparam name to function?
AGENT_LOOKUP = {
    "q_agent": Qagent,
    "shard_agent": ShardAgent,
    "random_walk_agent": RandomWalkAgent,
    "one_step_perfect_prediction_agent": OneStepPerfectPredictionAgent,
    "iterative_weight_optimization_agent": IterativeWeightOptimizationAgent,
}

ENV_LOOKUP = {"savanna-zoo-parallel-v2": SavannaZooParallelEnv, "savanna-zoo-sequential-v2": SavannaZooSequentialEnv}

MODEL_LOOKUP = {"dqn": DQN}


def run_episode(hparams: dict = {}, **args):
    env_params = hparams.get("env_params", {})
    agent_params = hparams.get("agent_params", {})
    render_mode = hparams.get("render_mode")
    verbose = hparams.get("verbose", False)

    env_type = hparams.get("env_type")
    print('env type', env_type)
    # gym_vec_env_v0(env, num_envs) creates a Gym vector environment with num_envs copies of the environment.
    # https://tristandeleu.github.io/gym/vector/
    # https://github.com/Farama-Foundation/SuperSuit

    # stable_baselines3_vec_env_v0(env, num_envs) creates a stable_baselines vector environment with num_envs copies of the environment. 

    if env_type == "zoo":
        env = ENV_LOOKUP[hparams["env"]](env_params=env_params)
        # if hparams.get('sequential_env', False) is True:
        #     print('converting to sequential from parallel')
        #     env = parallel_to_aec(env)
        # assumption here: all agents in zoo have same observation space shape
        env.reset()
        obs_size = env.observation_space('agent_0').shape[0]
        
        print('obs size', obs_size)
        n_actions = env.action_space('agent_0').n
        print('n actions', n_actions)

    elif env_type == "gym":
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
        # env = gym_vec_env_v0(env, num_envs=1)
    else:
        print(
            f'env_type {hparams.get("env_type")} not implemented. Choose: [zoo, gym]. TODO: add stable_baselines3'
        )

    env.reset(options={})
    
    buffer = ReplayBuffer(hparams['replay_size']) 
    
    model_spec = hparams["model"]
    if isinstance(model_spec, list):
        models = [MODEL_LOOKUP[net](obs_size, n_actions) for net in model_spec]
    else:
        models = [MODEL_LOOKUP[model_spec](obs_size, n_actions)]
    
    agent_spec = hparams["agent"]
    if isinstance(agent_spec, list) or env_params.get('num_agents', 1) > 1:
        if not isinstance(agent_spec, list):
            agent_spec = [agent_spec]
        if len(models) < len(agent_spec):
            models *= len(agent_spec)
        agents = [AGENT_LOOKUP[agent](env, model, buffer, **agent_params) for agent, model in zip(agent_spec, models)]
    else:
        agents = [AGENT_LOOKUP[agent_spec](env, models[0], buffer, **agent_params)]
    
    episode_rewards = [0 for x in agents]
    dones = [False for x in agents]
    warm_start_steps = hparams.get("warm_start_steps", 1000)
    
    for step in range(warm_start_steps):
        epsilon = 1.0 # forces random action for warmup steps
        if env_type == "zoo":
            actions = {}
            for agent in agents:
                # agent doesn't get to play_step, only env can, for multi-agent env compatibility
                # reward, done = agent.play_step(nets[i], epsilon=1.0)
                actions[agent.name] = agent.get_action(epsilon=epsilon, device=hparams.get("device", "cpu"))
            print('debug actions', actions)
            print('debug step')
            print(env.__dict__)
            print(env.step(actions))
            observations, rewards, dones, infos = env.step(actions)
        else:
            # the assumption by non-zoo env will be 1 agent generally I think
            for agent in agents:
                reward, done = agent.play_step(
                agent.model, epsilon, hparams.get("device", "cpu")
                )
                dones = [done]
        if any(dones):
            for agent in agents:
                if agent.done and verbose:
                    print(
                        f"Uhoh! Your agent {agent.name} terminated during warmup on step {step}/{warm_start_steps}"
                    )
        if all(dones):
            break

    while not all(dones):
        epsilon = max(
            hparams["eps_end"],
            hparams["eps_start"]
            - env.num_moves * 1 / hparams["eps_last_frame"],
        )
        if env_type == "zoo":
            actions = {}
            for agent in agents:
                # agent doesn't get to play_step, only env can, for multi-agent env compatibility
                # reward, done = agent.play_step(nets[i], epsilon=1.0)
                actions[agent.name] = agent.get_action(epsilon=1.0, device=hparams.get("device", "cpu"))
            observations, rewards, dones, infos = env.step(actions)
        else:
            # the assumption by non-zoo env will be 1 agent generally I think
            for agent in agents:
                reward, done = agent.play_step(
                agent.model, epsilon, hparams.get("device", "cpu")
                )
                dones = [done]
                rewards = [reward]
        episode_rewards += rewards
        if render_mode is not None:
            env.render(render_mode)

    
    if verbose:
        print(
            f"Simple Episode Evaluation completed. Final episode rewards: {episode_rewards}"
        )
