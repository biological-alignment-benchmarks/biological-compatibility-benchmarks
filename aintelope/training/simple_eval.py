from collections import Counter

import logging

import gymnasium as gym

from omegaconf import OmegaConf, DictConfig
from typing import Dict

from pettingzoo import AECEnv, ParallelEnv

from aintelope.agents.q_agent import QAgent
from aintelope.agents.instinct_agent import InstinctAgent
from aintelope.agents.simple_agents import (
    RandomWalkAgent,
    OneStepPerfectPredictionAgent,
    IterativeWeightOptimizationAgent,
)
from aintelope.models.dqn import DQN
from aintelope.environments.savanna_zoo import (
    SavannaZooParallelEnv,
    SavannaZooSequentialEnv,
)
from aintelope.environments.savanna_safetygrid import (
    SavannaGridworldParallelEnv,
    SavannaGridworldSequentialEnv,
)

from aintelope.training.dqn_training import Trainer


logger = logging.getLogger("aintelope.training.simple_eval")

AGENT_LOOKUP = {
    "q_agent": QAgent,
    "instinct_agent": InstinctAgent,
    "random_walk_agent": RandomWalkAgent,
    "one_step_perfect_prediction_agent": OneStepPerfectPredictionAgent,
    "iterative_weight_optimization_agent": IterativeWeightOptimizationAgent,
}

ENV_LOOKUP = {
    "savanna-zoo-parallel-v2": SavannaZooParallelEnv,
    "savanna-zoo-sequential-v2": SavannaZooSequentialEnv,
    "savanna-safetygrid-parallel-v1": SavannaGridworldParallelEnv,
    "savanna-safetygrid-sequential-v1": SavannaGridworldSequentialEnv,
}

MODEL_LOOKUP = {"dqn": DQN}


def run_episode(full_params: Dict) -> None:
    tparams = full_params.trainer_params
    hparams = full_params.hparams

    env_params = hparams["env_params"]
    agent_params = hparams["agent_params"]
    render_mode = env_params["render_mode"]
    verbose = tparams["verbose"]

    env_type = hparams["env_type"]
    logger.info("env type", env_type)
    # gym_vec_env_v0(env, num_envs) creates a Gym vector environment with num_envs copies of the environment.
    # https://tristandeleu.github.io/gym/vector/
    # https://github.com/Farama-Foundation/SuperSuit

    # stable_baselines3_vec_env_v0(env, num_envs) creates a stable_baselines vector environment with num_envs copies of the environment.

    if env_type == "zoo":
        env = ENV_LOOKUP[hparams["env"]](env_params=env_params)
        # if hparams.get('sequential_env', False) is True:
        #     logger.info('converting to sequential from parallel')
        #     env = parallel_to_aec(env)
        # assumption here: all agents in zoo have same observation space shape
        env.reset()

        # TODO: multi-agent compatibility
        # TODO: support for 3D-observation cube
        obs_size = env.observation_space("agent_0").shape[0]
        logger.info("obs size", obs_size)

        # TODO: multi-agent compatibility
        # TODO: multi-modal action compatibility
        n_actions = env.action_space("agent_0").n
        logger.info("n actions", n_actions)
    else:
        logger.info(
            f"env_type {hparams['env_type']} not implemented."
            "Choose: [zoo, gym]. TODO: add stable_baselines3"
        )

    env.reset(options={})

    action_space = env.action_space

    if isinstance(env, ParallelEnv):
        (
            observations,
            infos,
        ) = env.reset()  # TODO: each agent has their own state, refactor
        # TODO: each agent has their own observation size    # observation_space and action_space require agent argument: https://pettingzoo.farama.org/content/basic_usage/#additional-environment-api
        n_observations = len(  # TODO: support for 3D-observation cube
            observations["agent_0"]
        )
    elif isinstance(env, AECEnv):
        env.reset()
        # TODO: each agent has their own observation size    # observation_space and action_space require agent argument: https://pettingzoo.farama.org/content/basic_usage/#additional-environment-api
        observation = env.observe(
            "agent_0"
        )  # TODO: each agent has their own state, refactor
        n_observations = len(observation)  # TODO: support for 3D-observation cube
    else:
        raise NotImplementedError(f"Unknown environment type {type(env)}")

    # Common trainer for each agent's models
    trainer = Trainer(
        full_params, n_observations, action_space
    ) 

    model_spec = hparams["model"]
    if isinstance(model_spec, list):
        models = [MODEL_LOOKUP[net](obs_size, n_actions) for net in model_spec]
    else:
        models = [MODEL_LOOKUP[model_spec](obs_size, n_actions)]

    agent_spec = hparams["agent_id"]
    if isinstance(agent_spec, list) or env_params["amount_agents"] > 1:
        if not isinstance(agent_spec, list):
            agent_spec = [agent_spec]
        if len(models) < len(agent_spec):
            models *= len(agent_spec)
        agents = [
            AGENT_LOOKUP[agent](
                agent_id="agent_0",  
                trainer=trainer,
                target_instincts = [],
            )
            for agent in agent_spec
        ]
    else:
        agents = [
            AGENT_LOOKUP[agent_spec](
                agent_id="agent_0", 
                trainer=trainer,
                target_instincts = [],
            )
        ]

    # Agents
    for agent in agents:
        observation = env.observe(agent.id)  # TODO parallel env observation handling
        agent.reset(observation)
        trainer.add_agent(agent.id)

    agents_dict = {agent.id: agent for agent in agents}

    episode_rewards = Counter(
        {agent: 0.0 for agent in agents}
    )  # cannot use list since some of the agents may be terminated in the middle of the episode
    dones = {
        agent: False for agent in agents
    }  # cannot use list since some of the agents may be terminated in the middle of the episode
    warm_start_steps = hparams["warm_start_steps"]

    for step in range(warm_start_steps):
        if env_type == "zoo":
            dones = {}
            for agent_id in env.agent_iter(
                max_iter=env.num_agents
            ):  # num_agents returns number of alive (non-done) agents
                agent = agents_dict[agent_id]
                observation = env.observe(agent.id)  # TODO: parallel env support
                action = agent.get_action(
                    observation,
                    step=0,
                )
                logger.debug("debug action", action)
                logger.debug("debug step")
                logger.debug(env.__dict__)

                # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope provide slightly modified Zoo API. Normal Zoo sequential API step() method does not return values and is not allowed to return values else Zoo API tests will fail.
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step_single_agent(
                    action
                )  # TODO: parallel env support
                logger.debug((observation, reward, terminated, truncated, info))
                done = terminated or truncated
                dones[agent.id] = done

        else:
            logger.warning(
                        "Simple_eval: non-zoo env, test not yet implemented!"
                    )
            pass

        if any(dones.values()):
            for agent in agents:
                if dones[agent.id] and verbose:
                    logger.warning(
                        f"Uhoh! Your agent {agent.id} terminated during warmup"
                        "on step {step}/{warm_start_steps}"
                    )
        if all(dones.values()):
            break

    step = -1
    while not all(dones.values()):
        step += 1 # debugging only
        if env_type == "zoo":
            rewards = {}
            for agent_id in env.agent_iter(
                max_iter=env.num_agents
            ):  # num_agents returns number of alive (non-done) agents
                agent = agents_dict[agent_id]
                action = action_space("agent_0").sample()
                action = agent.get_action(
                    observation,
                    step=0,
                )
                logger.debug("debug action", action)
                logger.debug("debug step")
                logger.debug(env.__dict__)

                # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope provide slightly modified Zoo API. Normal Zoo sequential API step() method does not return values and is not allowed to return values else Zoo API tests will fail.
                (
                    observation,
                    reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step_single_agent(
                    action
                )  # TODO: parallel env support
                logger.debug((observation, reward, terminated, truncated, info))
                done = terminated or truncated
                dones[agent.id] = done
                rewards[agent] = reward
        else:
            logger.warning(
                        "Simple_eval: non-zoo env, test not yet implemented!"
                    )
            pass

        episode_rewards += rewards  # Counter class allows addition per dictionary keys
        if render_mode is not None:
            env.render(render_mode)

    if verbose:
        logger.info(
            f"Simple Episode Evaluation completed."
            "Final episode rewards: {episode_rewards}"
        )
