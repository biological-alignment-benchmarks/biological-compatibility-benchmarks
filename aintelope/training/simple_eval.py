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

# TODO: use registration instead?
AGENT_LOOKUP = {
    "q_agent": QAgent,
    "instinct_agent": InstinctAgent,
    "random_walk_agent": RandomWalkAgent,
    "one_step_perfect_prediction_agent": OneStepPerfectPredictionAgent,
    "iterative_weight_optimization_agent": IterativeWeightOptimizationAgent,
}

# TODO: use registration instead?
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
        obs_size = (
            env.observation_space("agent_0")[0].shape,
            env.observation_space("agent_0")[1].shape,
        )
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

    if isinstance(env, ParallelEnv):
        (
            observations,
            infos,
        ) = env.reset()
    elif isinstance(env, AECEnv):
        env.reset()
    else:
        raise NotImplementedError(f"Unknown environment type {type(env)}")

    # Common trainer for each agent's models
    trainer = Trainer(full_params)

    model_spec = hparams["model"]
    unit_test_mode = hparams[
        "unit_test_mode"
    ]  # is set during tests in order to speed up DQN computations

    # TODO: support for different observation shapes in different agents?
    # TODO: support for different action spaces in different agents?
    if isinstance(model_spec, list):
        models = [
            MODEL_LOOKUP[net](obs_size, n_actions, unit_test_mode=unit_test_mode)
            for net in model_spec
        ]
    else:
        models = [
            MODEL_LOOKUP[model_spec](obs_size, n_actions, unit_test_mode=unit_test_mode)
        ]

    agent_spec = hparams["agent_id"]  # TODO: why is this value a list?
    if isinstance(agent_spec, list) and len(agent_spec) == 1:
        # NB! after this step the agent_spec is not a list anymore and the following if condition will be False, so do not try to merge these "if" branches.
        agent_spec = agent_spec[0]

    if isinstance(agent_spec, list):
        if len(models) < len(agent_spec):
            models *= len(
                agent_spec
            )  # TODO: shouldnt it be env_params["amount_agents"] here?
        agents = [  # TODO: this nested list structure probably will not work in below code. What is the intention of using multiple agent_specs?
            [
                AGENT_LOOKUP[agent](
                    agent_id=f"agent_{i}",
                    trainer=trainer,
                    target_instincts=[],
                )
                for agent in agent_spec
            ]
            for i in range(env_params["amount_agents"])
        ]
    else:
        agents = [
            AGENT_LOOKUP[agent_spec](
                agent_id=f"agent_{i}",
                trainer=trainer,
                target_instincts=[],
            )
            for i in range(env_params["amount_agents"])
        ]

    # Agents
    for agent in agents:
        if isinstance(env, ParallelEnv):
            observation = observations[agent.id]
            info = infos[agent.id]
        elif isinstance(env, AECEnv):
            observation = env.observe(agent.id)
            info = env.observe_info(agent.id)

        agent.reset(observation, info)
        trainer.add_agent(
            agent.id,
            (observation[0].shape, observation[1].shape),
            env.action_space,
            unit_test_mode=unit_test_mode,
        )

    agents_dict = {agent.id: agent for agent in agents}

    episode_rewards = Counter(
        {agent.id: 0.0 for agent in agents}
    )  # cannot use list since some of the agents may be terminated in the middle of the episode
    dones = {
        agent.id: False for agent in agents
    }  # cannot use list since some of the agents may be terminated in the middle of the episode
    warm_start_steps = hparams["warm_start_steps"]

    for step in range(warm_start_steps):
        if env_type == "zoo":
            if isinstance(env, ParallelEnv):
                # loop: get observations and collect actions
                actions = {}
                for agent in agents:  # TODO: exclude terminated agents
                    if dones[agent.id]:
                        continue
                    observation = observations[agent.id]
                    info = infos[agent.id]
                    actions[agent.id] = agent.get_action(observation, info, step)

                logger.debug("debug actions", actions)
                logger.debug("debug step")
                logger.debug(env.__dict__)

                # call: send actions and get observations
                observations, rewards, terminateds, truncateds, infos = env.step(
                    actions
                )

                logger.debug((observations, rewards, terminateds, truncateds, infos))
                dones.update(
                    {  # call update since the list of terminateds will become smaller on second step after agents have died
                        key: terminated or truncateds[key]
                        for (key, terminated) in terminateds.items()
                    }
                )

            elif isinstance(env, AECEnv):
                for agent_id in env.agent_iter(
                    max_iter=env.num_agents
                ):  # num_agents returns number of alive (non-done) agents
                    agent = agents_dict[agent_id]
                    observation = env.observe(agent.id)
                    info = env.observe_info(agent.id)
                    # agent doesn't get to play_step, only env can, for multi-agent env compatibility
                    # reward, score, done = agent.play_step(nets[i], epsilon=1.0)
                    # Per Zoo API, a dead agent must call .step(None) once more after becoming dead. Only after that call will this dead agent be removed from various dictionaries and from .agent_iter loop.
                    if env.terminations[agent_id] or env.truncations[agent.id]:
                        action = None
                    else:
                        # action = action_space(agent.id).sample()
                        action = agent.get_action(
                            observation,
                            info,
                            step,  # TODO: there was step=0 but we have step index available in this iteration. So is it okay if I use the step variable here?
                        )

                    logger.debug("debug action", action)
                    logger.debug("debug step")
                    logger.debug(env.__dict__)

                    # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope provide slightly modified Zoo API. Normal Zoo sequential API step() method does not return values and is not allowed to return values else Zoo API tests will fail.
                    result = env.step_single_agent(action)  # TODO: parallel env support

                    if agent.id in env.agents:  # was not "dead step"
                        (
                            observation,
                            reward,  # NB! This is only initial reward upon agent's own step. When other agents take their turns then the reward of the agent may change. If you need to learn an agent's accumulated reward over other agents turns (plus its own step's reward) then use env.last property.
                            terminated,
                            truncated,
                            info,
                        ) = result

                        logger.debug((observation, reward, terminated, truncated, info))

                        # NB! any agent could die at any other agent's step
                        for agent_id in env.agents:
                            dones[agent_id] = (
                                env.terminations[agent_id] or env.truncations[agent.id]
                            )

        else:
            logger.warning("Simple_eval: non-zoo env, test not yet implemented!")
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
        step += 1
        if env_type == "zoo":
            rewards = {}

            if isinstance(env, ParallelEnv):
                # loop: get observations and collect actions
                actions = {}
                for agent in agents:  # TODO: exclude terminated agents
                    if dones[agent.id]:
                        continue
                    observation = observations[agent.id]
                    actions[agent.id] = agent.get_action(observation, step)

                logger.debug("debug actions", actions)
                logger.debug("debug step")
                logger.debug(env.__dict__)

                # call: send actions and get observations
                observations, rewards, terminateds, truncateds, infos = env.step(
                    actions
                )

                logger.debug((observations, rewards, terminateds, truncateds, infos))
                dones.update(
                    {  # call update since the list of terminateds will become smaller on second step after agents have died
                        key: terminated or truncateds[key]
                        for (key, terminated) in terminateds.items()
                    }
                )

                # rewards is already in a proper format here

            elif isinstance(env, AECEnv):
                for agent_id in env.agent_iter(
                    max_iter=env.num_agents
                ):  # num_agents returns number of alive (non-done) agents
                    agent = agents_dict[agent_id]
                    # agent doesn't get to play_step, only env can, for multi-agent env compatibility
                    # reward, score, done = agent.play_step(nets[i], epsilon=1.0)
                    # Per Zoo API, a dead agent must call .step(None) once more after becoming dead. Only after that call will this dead agent be removed from various dictionaries and from .agent_iter loop.
                    if env.terminations[agent_id] or env.truncations[agent.id]:
                        action = None
                    else:
                        # action = action_space(agent.id).sample()
                        action = agent.get_action(
                            observation,
                            step,  # TODO: there was step=0 but we have step index available in this iteration. So is it okay if I use the step variable here?
                        )

                    logger.debug("debug action", action)
                    logger.debug("debug step")
                    logger.debug(env.__dict__)

                    # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope provide slightly modified Zoo API. Normal Zoo sequential API step() method does not return values and is not allowed to return values else Zoo API tests will fail.
                    result = env.step_single_agent(action)  # TODO: parallel env support

                    if agent.id in env.agents:  # was not "dead step"
                        (
                            observation,
                            reward,  # NB! This is only initial reward upon agent's own step. When other agents take their turns then the reward of the agent may change. If you need to learn an agent's accumulated reward over other agents turns (plus its own step's reward) then use env.last property.
                            terminated,
                            truncated,
                            info,
                        ) = result

                        logger.debug((observation, reward, terminated, truncated, info))
                        rewards[agent] = reward

                        # NB! any agent could die at any other agent's step
                        for agent_id in env.agents:
                            dones[agent_id] = (
                                env.terminations[agent_id] or env.truncations[agent.id]
                            )
        else:
            logger.warning("Simple_eval: non-zoo env, test not yet implemented!")
            pass

        episode_rewards += rewards  # Counter class allows addition per dictionary keys
        if render_mode is not None:
            env.render(render_mode)

    if verbose:
        logger.info(
            f"Simple Episode Evaluation completed."
            "Final episode rewards: {episode_rewards}"
        )
