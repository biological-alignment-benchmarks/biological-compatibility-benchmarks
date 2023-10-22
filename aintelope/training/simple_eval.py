import logging

import gymnasium as gym

from omegaconf import DictConfig

from aintelope.agents.q_agent import Agent as Qagent
from aintelope.agents.instinct_agent import InstinctAgent
from aintelope.agents.simple_agents import (
    RandomWalkAgent,
    OneStepPerfectPredictionAgent,
    IterativeWeightOptimizationAgent,
)
from aintelope.agents.memory import ReplayBuffer
from aintelope.models.dqn import DQN
from aintelope.environments.savanna_zoo import (
    SavannaZooParallelEnv,
    SavannaZooSequentialEnv,
)

logger = logging.getLogger("aintelope.training.simple_eval")

# is there a better way to do this?
# to register a lookup table from hparam name to function?
AGENT_LOOKUP = {
    "q_agent": Qagent,
    "instinct_agent": InstinctAgent,
    "random_walk_agent": RandomWalkAgent,
    "one_step_perfect_prediction_agent": OneStepPerfectPredictionAgent,
    "iterative_weight_optimization_agent": IterativeWeightOptimizationAgent,
}

ENV_LOOKUP = {
    "savanna-zoo-parallel-v2": SavannaZooParallelEnv,
    "savanna-zoo-sequential-v2": SavannaZooSequentialEnv,
}

MODEL_LOOKUP = {"dqn": DQN}


def run_episode(tparams: DictConfig, hparams: DictConfig) -> None:
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
        obs_size = env.observation_space("agent_0").shape[0]

        logger.info("obs size", obs_size)
        n_actions = env.action_space("agent_0").n
        logger.info("n actions", n_actions)

    elif env_type == "gym":
        # GYM_INTERACTION
        if hparams["env_entry_point"] is not None:
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
        logger.info(
            f"env_type {hparams['env_type']} not implemented."
            "Choose: [zoo, gym]. TODO: add stable_baselines3"
        )

    env.reset(options={})

    buffer = ReplayBuffer(hparams["replay_size"])

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
                env, model, buffer, hparams["warm_start_size"], **agent_params
            )
            for agent, model in zip(agent_spec, models)
        ]
    else:
        agents = [
            AGENT_LOOKUP[agent_spec](
                env, models[0], buffer, hparams["warm_start_size"], **agent_params
            )
        ]

    episode_rewards = [0 for x in agents]
    dones = [False for x in agents]
    warm_start_steps = hparams["warm_start_steps"]

    for step in range(warm_start_steps):
        epsilon = 1.0  # forces random action for warmup steps
        if env_type == "zoo":
            actions = {}
            for agent in agents:
                # agent doesn't get to play_step, only env can, for multi-agent env compatibility
                # reward, done = agent.play_step(nets[i], epsilon=1.0)
                actions[agent.name] = agent.get_action(
                    epsilon=epsilon, device=tparams["device"]
                )
            logger.debug("debug actions", actions)
            logger.debug("debug step")
            logger.debug(env.__dict__)

            logger.debug(
                env.step(actions)
            )  # TODO: is this a bug? env.step() is called two times - on this line and below
            observations, rewards, terminateds, truncateds, infos = env.step(action)
            dones = {
                key: terminated or truncateds[key]
                for (key, terminated) in terminateds.items()
            }
        else:
            # the assumption by non-zoo env will be 1 agent generally I think
            for agent in agents:
                reward, done = agent.play_step(agent.model, epsilon, tparams["device"])
                dones = [done]
        if any(dones):
            for agent in agents:
                if agent.done and verbose:
                    logger.warning(
                        f"Uhoh! Your agent {agent.name} terminated during warmup"
                        "on step {step}/{warm_start_steps}"
                    )
        if all(dones):
            break

    while not all(dones):
        epsilon = max(
            hparams["eps_end"],
            hparams["eps_start"] - env.num_moves * 1 / hparams["eps_last_frame"],
        )
        if env_type == "zoo":
            actions = {}
            for agent in agents:
                # agent doesn't get to play_step, only env can, for multi-agent env compatibility
                # reward, done = agent.play_step(nets[i], epsilon=1.0)
                actions[agent.name] = agent.get_action(
                    epsilon=1.0, device=tparams["device"]
                )

            observations, rewards, terminateds, truncateds, infos = env.step(action)
            dones = {
                key: terminated or truncateds[key]
                for (key, terminated) in terminateds.items()
            }
        else:
            # the assumption by non-zoo env will be 1 agent generally I think
            for agent in agents:
                reward, done = agent.play_step(agent.model, epsilon, tparams["device"])
                dones = [done]
                rewards = [reward]
        episode_rewards += rewards
        if render_mode is not None:
            env.render(render_mode)

    if verbose:
        logger.info(
            f"Simple Episode Evaluation completed."
            "Final episode rewards: {episode_rewards}"
        )
