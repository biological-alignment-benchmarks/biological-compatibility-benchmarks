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

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def main(cfg: DictConfig) -> None:
    logger = logging.getLogger("aintelope.experiment")
    
    episode_durations = []

    # Environment
    env = SavannaGymEnv(env_params=cfg.hparams.env_params) #TODO: get env from parameters
    action_space = env.action_space
    observation, info = env.reset() #TODO: each agent has their own state, refactor
    # TODO: env doesnt register agents properly, it hallucinates from zooapi and names in its own way
    # figure out how to make this coherent. there's "possible_agents" now
    n_observations = len(observation)
    
    # Common trainer for each agent's models
    trainer = Trainer(cfg, n_observations, action_space) # TODO: have a section in params for trainer? its trainer and hparams now tho
    
    # Agents
    agents = []
    for i in range(cfg.hparams.env_params.amount_agents):
        agent_id = f"agent_{i}"
        agents.append(get_agent_class(cfg.hparams.agent_id)(
            agent_id,
            trainer,
            cfg.hparams.warm_start_steps,
            **cfg.hparams.agent_params,
        ))
        # TODO: savanna_gym interface will reduce {agent_0:obs} to obs... take into account here
        agents[-1].reset(env.observe(agent_id)) 
        trainer.add_agent(agent_id)
    
    # Warmup not supported atm, maybe not needed?
    #for _ in range(hparams.warm_start_steps):
    #     agents.play_step(self.net, epsilon=1.0) # TODO
    steps_done = 0
    
    # Main loop
    for i_episode in range(cfg.hparams.num_episodes):
        # Reset
        _, _ = env.reset()
        for agent in agents:
            agent.reset(env.observe(agent.id))

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
                ### TODO: move to support only pettingzoo?
                #observation, reward, terminated, truncated, _ = env.step(action)

                # Agent is updated based on what the env shows. All commented above included ^
                done = terminated or truncated
                if terminated:
                    observation = None
                agent.update(env, observation, score, done) # note that score is used ONLY by baseline

                # Perform one step of the optimization (on the policy network)
                # TEST: if we call this every time, will it overlearn the initial steps? The buffer
                # is filled only with a batch worth of stuff, and it might overrepresent?
                trainer.optimize_models(step)
                
            # TODO: break when all agents are don
            #if done:
            #    episode_durations.append(step + 1)
            #    break 
        
if __name__ == "__main__":
    main()
