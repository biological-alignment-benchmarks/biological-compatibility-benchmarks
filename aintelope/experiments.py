from collections import namedtuple

import logging
from omegaconf import DictConfig

import os
from pathlib import Path

from aintelope.training.dqn_training import Trainer

from aintelope.agents import (
    Agent,
    PettingZooEnv,
    Environment,
    register_agent_class,
)

# initialize environment registries
from pettingzoo import AECEnv, ParallelEnv
from aintelope.environments.savanna_zoo import (
    SavannaZooParallelEnv,
    SavannaZooSequentialEnv,
)
from aintelope.environments.savanna_safetygrid import (
    SavannaGridworldParallelEnv,
    SavannaGridworldSequentialEnv,
)
from aintelope.environments import get_env_class

# initialize agent registries
from aintelope.agents.instinct_agent import InstinctAgent
from aintelope.agents.q_agent import QAgent
from aintelope.agents import get_agent_class


def run_experiment(cfg: DictConfig) -> None:
    logger = logging.getLogger("aintelope.experiment")

    # Environment
    env = get_env_class(cfg.hparams.env)(env_params=cfg.hparams.env_params)
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
    trainer = Trainer(cfg)

    # Agents
    agents = []
    dones = {}
    for i in range(cfg.hparams.env_params.amount_agents):
        agent_id = f"agent_{i}"
        agents.append(
            get_agent_class(cfg.hparams.agent_id)(
                agent_id,
                trainer,
                **cfg.hparams.agent_params,
            )
        )
		# TODO: it is possible to call env.observation_space(agent_id) directly to get the observation shape. No need to call observe().
		if isinstance(env, ParallelEnv):
        	observation = observations[agent_id]
		elif isinstance(env, AECEnv):
			observation = env.observe(agent_id)
        agents[-1].reset(observation)
        trainer.add_agent(agent_id, observation.shape, env.action_space)

    # Warmup NIY
    # for _ in range(hparams.warm_start_steps):
    #    agents.play_step(self.net, epsilon=1.0)

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

        # Iterations within the episode
        for step in range(cfg.hparams.env_params.num_iters):
            if isinstance(env, ParallelEnv):
                # loop: get observations and collect actions
                actions = {}
                for agent in agents:  # TODO: exclude terminated agents
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
                agents_dict = {agent.id: agent for agent in agents}
                for agent_id in env.agent_iter(
                    max_iter=env.num_agents
                ):  # num_agents returns number of alive (non-done) agents
                    agent = agents_dict[agent_id]

                    # Per Zoo API, a dead agent must call .step(None) once more after becoming dead. Only after that call will this dead agent be removed from various dictionaries and from .agent_iter loop.
                    if env.terminations[agent.id] or env.truncations[agent.id]:
                        action = None
                    else:
                        observation = env.observe(agent.id)
                        action = agent.get_action(observation, step)

                    # Env step
                    # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope provide slightly modified Zoo API. Normal Zoo sequential API step() method does not return values and is not allowed to return values else Zoo API tests will fail.
                    result = env.step_single_agent(action)

                    if agent.id in env.agents:  # was not "dead step"
                        (
                            observation,
                            score,  # NB! This is only initial reward upon agent's own step. When other agents take their turns then the reward of the agent may change. If you need to learn an agent's accumulated reward over other agents turns (plus its own step's reward) then use env.last property.
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
            trainer.optimize_models()

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
