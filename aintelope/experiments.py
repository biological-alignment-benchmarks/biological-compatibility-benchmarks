from collections import namedtuple

import logging
from omegaconf import DictConfig
import hydra

from aintelope.environments.savanna_gym import SavannaGymEnv
from aintelope.models.dqn import DQN
from aintelope.agents import (
    Agent,
    GymEnv,
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
    env = SavannaGymEnv(
        env_params=cfg.hparams.env_params
    )  # TODO: get env from parameters
    action_space = env.action_space
    observation, info = env.reset()  # TODO: each agent has their own state, refactor
    n_observations = len(observation)

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
        agents[-1].reset(env.observe(agent_id))
        trainer.add_agent(agent_id)

    # Warmup not supported atm, would be here
    # for _ in range(hparams.warm_start_steps):
    #     agents.play_step(self.net, epsilon=1.0)

    # Main loop
    for i_episode in range(cfg.hparams.num_episodes):
        # Reset
        _, _ = env.reset()
        for agent in agents:
            agent.reset(env.observe(agent.id))
            dones[agent.id] = False

        for step in range(cfg.hparams.env_params.num_iters):
            for agent in agents:
                observation = env.observe(agent.id)
                action = agent.get_action(observation, step)

                # Env step
                if isinstance(env, GymEnv):
                    observation, score, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                elif isinstance(env, PettingZooEnv):
                    observation, score, terminateds, truncateds, _ = env.step(action)
                    done = {
                        key: terminated or truncateds[key]
                        for (key, terminated) in terminateds.items()
                    }
                else:
                    logger.warning(f"{env} is not of type GymEnv or PettingZooEnv")
                    observation, score, done, _ = env.step(action)
                # observation, reward, terminated, truncated, _ = env.step(action)

                # Agent is updated based on what the env shows. All commented above included ^
                done = terminated or truncated
                dones[agent.id] = done
                if terminated:
                    observation = None
                agent.update(
                    env, observation, score, done
                )  # note that score is used ONLY by baseline

                # Perform one step of the optimization (on the policy network)
                trainer.optimize_models(step)
            
            # Break when all agents are done
            if all(dones.values()):
                break

        # Save models
        # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        dir_out = f"{cfg.experiment_dir}"
        if i_episode % cfg.hparams.every_n_episodes == 0:
            trainer.save_models(i_episode, dir_out + "/")


# @hydra.main(version_base=None, config_path="config", config_name="config_experiment")
if __name__ == "__main__":
    run_experiment()
