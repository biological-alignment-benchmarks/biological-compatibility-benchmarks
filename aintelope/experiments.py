import glob
import logging
import os
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from aintelope.agents import get_agent_class
from aintelope.analytics import recording as rec
from aintelope.environments import get_env_class
from aintelope.training.dqn_training import Trainer

from pettingzoo import AECEnv, ParallelEnv


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

    dir_out = f"{cfg.log_dir}"
    dir_cp = dir_out + "checkpoints/"

    unit_test_mode = (
        cfg.hparams.unit_test_mode
    )  # is set during tests in order to speed up DQN computations

    # Agents
    agents = []
    dones = {}
    for i in range(env.max_num_agents):
        agent_id = f"agent_{i}"
        agents.append(
            get_agent_class(cfg.hparams.agent_id)(
                agent_id,
                trainer,
                **cfg.hparams.agent_params,
            )
        )

        # TODO: IF agent.reset() below is not needed then it is possible to call
        # env.observation_space(agent_id) directly to get the observation shape.
        # No need to call observe().
        if isinstance(env, ParallelEnv):
            observation = observations[agent_id]
            info = infos[agent_id]
        elif isinstance(env, AECEnv):
            observation = env.observe(agent_id)
            info = env.observe_info(agent_id)

        # TODO: is this reset necessary here? In main loop below,
        # there is also a reset call
        agents[-1].reset(observation, info)
        # Get latest checkpoint if existing
        checkpoint = None
        checkpoints = glob.glob(dir_cp + agent_id + "*")
        if len(checkpoints) > 0:
            checkpoint = max(checkpoints, key=os.path.getctime)
        # Add agent, with potential checkpoint
        trainer.add_agent(
            agent_id,
            (observation[0].shape, observation[1].shape),
            env.action_space,
            unit_test_mode=unit_test_mode,
            checkpoint=checkpoint,
        )
        dones[agent_id] = False

    # Warmup not yet implemented
    # for _ in range(hparams.warm_start_steps):
    #    agents.play_step(self.net, epsilon=1.0)

    # Main loop
    score_titles = [title.keys()[0] for title in cfg.hparams.env_params.scores]
    events = pd.DataFrame(
        columns=[
            "Run_id",
            "Episode",
            "Step",
            "Agent_id",
            "State",
            "Action",
            "Reward",
            "Done",
            "Next_state",
        ]
        + score_titles
    )

    for i_episode in range(cfg.hparams.num_episodes):
        # Reset
        if isinstance(env, ParallelEnv):
            (
                observations,
                infos,
            ) = env.reset()
            for agent in agents:
                agent.reset(observations[agent.id], infos[agent.id])
                dones[agent.id] = False

        elif isinstance(env, AECEnv):
            env.reset()
            for agent in agents:
                agent.reset(env.observe(agent.id), env.observe_info(agent_id))
                dones[agent.id] = False

        # Iterations within the episode
        for step in range(cfg.hparams.env_params.num_iters):
            if isinstance(env, ParallelEnv):
                # loop: get observations and collect actions
                actions = {}
                for agent in agents:  # TODO: exclude terminated agents
                    observation = observations[agent.id]
                    info = infos[agent.id]
                    actions[agent.id] = agent.get_action(observation, info, step)

                # call: send actions and get observations
                observations, scores, terminateds, truncateds, infos = env.step(actions)
                # call update since the list of terminateds will become smaller on
                # second step after agents have died
                dones.update(
                    {
                        key: terminated or truncateds[key]
                        for (key, terminated) in terminateds.items()
                    }
                )

                # loop: update
                for agent in agents:
                    observation = observations[agent.id]
                    info = infos[agent.id]
                    score = scores[agent.id]
                    done = dones[agent.id]
                    terminated = terminateds[agent.id]
                    if terminated:
                        observation = None
                    agent_step_info = agent.update(
                        env,
                        observation,
                        info,
                        sum(
                            score.values()
                        ),  # TODO: make a function to handle obs->rew in Q-agent too, remove this
                        done,  # TODO: should it be "terminated" in place of "done" here?
                        done,  # TODO: should it be "terminated" in place of "done" here?
                    )

                    events.loc[len(events)] = (
                        [cfg.experiment_name, i_episode, step]
                        + agent_step_info
                        + score.values()
                    )

            elif isinstance(env, AECEnv):
                # loop: observe, collect action, send action, get observation, update
                agents_dict = {agent.id: agent for agent in agents}
                for agent_id in env.agent_iter(
                    max_iter=env.num_agents
                ):  # num_agents returns number of alive (non-done) agents
                    agent = agents_dict[agent_id]

                    # Per Zoo API, a dead agent must call .step(None) once more after
                    # becoming dead. Only after that call will this dead agent be
                    # removed from various dictionaries and from .agent_iter loop.
                    if env.terminations[agent.id] or env.truncations[agent.id]:
                        action = None
                    else:
                        observation = env.observe(agent.id)
                        info = env.observe_info(agent.id)
                        action = agent.get_action(observation, info, step)

                    # Env step
                    # NB! both AIntelope Zoo and Gridworlds Zoo wrapper in AIntelope
                    # provide slightly modified Zoo API. Normal Zoo sequential API
                    # step() method does not return values and is not allowed to return
                    # values else Zoo API tests will fail.
                    result = env.step_single_agent(action)

                    if agent.id in env.agents:  # was not "dead step"
                        # NB! This is only initial reward upon agent's own step.
                        # When other agents take their turns then the reward of the
                        # agent may change. If you need to learn an agent's accumulated
                        # reward over other agents turns (plus its own step's reward)
                        # then use env.last property.
                        (
                            observation,
                            score,
                            terminated,
                            truncated,
                            info,
                        ) = result

                        done = terminated or truncated

                        # Agent is updated based on what the env shows.
                        # All commented above included ^
                        if terminated:
                            observation = None  # TODO: why is this here?
                        agent_step_info = agent.update(
                            env,
                            observation,
                            info,
                            sum(score.values()),
                            done,  # TODO: should it be "terminated" in place of "done" here?
                        )  # note that score is used ONLY by baseline

                        # Record what just happened
                        events.loc[len(events)] = (
                            [cfg.experiment_name, i_episode, step]
                            + agent_step_info
                            + score.values()
                        )

                        # NB! any agent could die at any other agent's step
                        for agent_id in env.agents:
                            dones[agent_id] = (
                                env.terminations[agent_id] or env.truncations[agent.id]
                            )
                            # TODO: if the agent died during some other agents step,
                            # should we call agent.update() on the dead agent,
                            # else it will be never called?

            else:
                raise NotImplementedError(f"Unknown environment type {type(env)}")

            # Perform one step of the optimization (on the policy network)
            trainer.optimize_models()

            # Break when all agents are done
            if all(dones.values()):
                break

        # Save models
        # https://pytorch.org/tutorials/recipes/recipes/
        # saving_and_loading_a_general_checkpoint.html
        if i_episode % cfg.hparams.save_frequency == 0:
            os.makedirs(dir_cp, exist_ok=True)
            trainer.save_models(i_episode, dir_cp)

    record_path = Path(f"{cfg.experiment_dir}" + f"{cfg.events_dir}")
    rec.record_events(record_path, events)


# @hydra.main(version_base=None, config_path="config", config_name="config_experiment")
if __name__ == "__main__":
    run_experiment()
