from collections import namedtuple

import logging
from omegaconf import DictConfig

import os
from pathlib import Path

from pettingzoo import AECEnv, ParallelEnv
from aintelope.environments.savanna_zoo import (
    SavannaZooParallelEnv,
    SavannaZooSequentialEnv,
)
from aintelope.environments.savanna_safetygrid import (
    SavannaGridworldParallelEnv,
    SavannaGridworldSequentialEnv,
)
from aintelope.environments.savanna_safetygrid import SavannaGridworldSequentialEnv

from aintelope.models.dqn import DQN
from aintelope.agents import (
    Agent,
    PettingZooEnv,
    Environment,
    register_agent_class,
)
from aintelope.agents.instinct_agent import QAgent  # initialize agent registry
from aintelope.agents import get_agent_class
from aintelope.training.dqn_training import Trainer


def run_experiment(cfg: DictConfig) -> None:
    logger = logging.getLogger("aintelope.experiment")

    # Environment
    hparams = cfg.hparams
    if hparams.env == "savanna-zoo-parallel-v2":
        env = SavannaZooParallelEnv(env_params=hparams.env_params)
    elif hparams.env == "savanna-safetygrid-parallel-v1":
        env = SavannaGridworldParallelEnv(env_params=hparams.env_params)
    elif hparams.env == "savanna-zoo-sequential-v2":
        env = SavannaZooSequentialEnv(env_params=hparams.env_params)
    elif hparams.env == "savanna-safetygrid-sequential-v1":
        env = SavannaGridworldSequentialEnv(env_params=hparams.env_params)
    else:
        raise NotImplementedError()

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
    else:
        env.reset()
        # TODO: each agent has their own observation size    # observation_space and action_space require agent argument: https://pettingzoo.farama.org/content/basic_usage/#additional-environment-api
        observation = env.observe(
            "agent_0"
        )  # TODO: each agent has their own state, refactor
        n_observations = len(observation)  # TODO: support for 3D-observation cube

    # Common trainer for each agent's models
    trainer = Trainer(
        cfg, n_observations, action_space
    )  # TODO: have a section in params for trainer? its trainer and hparams now tho

    # Agents
    agents = []
    dones = {}  # are agents done
    for i in range(cfg.hparams.env_params.amount_agents):
        agent_id = f"agent_{i}"
        agents.append(
            get_agent_class(cfg.hparams.agent_id)(
                agent_id,
                trainer,
                cfg.hparams.warm_start_steps,
                **cfg.hparams.agent_params,
            )
        )
        observation = env.observe(agent_id)
        agents[-1].reset(observation)
        trainer.add_agent(agent_id)

    # Warmup not supported atm, would be here
    # for _ in range(hparams.warm_start_steps):
    #     agents.play_step(self.net, epsilon=1.0)

    # Main loop
    for i_episode in range(cfg.hparams.num_episodes):
        # Reset
        if isinstance(env, ParallelEnv):
            (
                observations,
                infos,
            ) = env.reset()
            for agent in agents:
                agent.reset(observations[agent.id])
                dones[agent.id] = False

        elif isinstance(env, AECEnv):
            env.reset()
            for agent in agents:
                agent.reset(env.observe(agent.id))
                dones[agent.id] = False

        for step in range(cfg.hparams.env_params.num_iters):
            if isinstance(env, ParallelEnv):
                # loop: get observations and collect actions
                actions = {}
                for agent in agents:
                    observation = env.observe(agent.id)
                    actions[agent.id] = agent.get_action(observation, step)

                # call: send actions and get observations
                observations, scores, terminateds, truncateds, _ = env.step(actions)
                dones = {
                    key: terminated or truncateds[key]
                    for (key, terminated) in terminateds.items()
                }

                # loop: update
                for agent in agents:
                    observation = observations[agent.id]
                    score = scores[agent.id]
                    done = dones[agent.id]
                    terminated = terminateds[agent.id]
                    if terminated:
                        observation = None
                    agent.update(
                        env, observation, score, done
                    )  # note that score is used ONLY by baseline

            elif isinstance(env, AECEnv):
                # loop: observe, collect action, send action, get observation, update
                for (
                    agent
                ) in agents:  # TODO: the order of agents needs to be obtained from env
                    observation = env.observe(agent.id)
                    action = agent.get_action(observation, step)

                    # Env step
                    # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope provide slightly modified Zoo API. Normal Zoo sequential API step() method does not return values.
                    result = env.step(action)
                    (
                        observation,
                        score,
                        terminated,
                        truncated,
                        info,
                    ) = result
                    done = terminated or truncated

                    # Agent is updated based on what the env shows. All commented above included ^
                    dones[agent.id] = done
                    if terminated:
                        observation = None
                    agent.update(
                        env, observation, score, done
                    )  # note that score is used ONLY by baseline

            else:
                raise NotImplementedError(f"Unknown environment type {type(env)}")

            # Perform one step of the optimization (on the policy network)
            trainer.optimize_models(step)

            # Break when all agents are done
            if all(dones.values()):
                break

        # Save models
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        dir_out = f"{cfg.experiment_dir}"
        if i_episode % cfg.hparams.every_n_episodes == 0:
            dir_cp = dir_out + "checkpoints/"
            os.makedirs(dir_cp, exist_ok=True)
            trainer.save_models(i_episode, dir_cp)

    record_path = Path(cfg.experiment_dir) / "memory_records.csv"
    logger.info(f"Saving training records to disk at {record_path}")
    record_path.parent.mkdir(exist_ok=True, parents=True)
    for agent in agents:
        agent.get_history().to_csv(record_path, index=False)


# @hydra.main(version_base=None, config_path="config", config_name="config_experiment")
if __name__ == "__main__":
    run_experiment()
