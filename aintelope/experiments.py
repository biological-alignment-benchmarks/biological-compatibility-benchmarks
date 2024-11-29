# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import glob
import logging
import os
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from aintelope.utils import RobustProgressBar

from aintelope.agents import get_agent_class
from aintelope.analytics import recording as rec
from aintelope.environments import get_env_class
from aintelope.environments.savanna_safetygrid import GridworldZooBaseEnv
from aintelope.training.dqn_training import Trainer

from pettingzoo import AECEnv, ParallelEnv


def run_experiment(
    cfg: DictConfig,
    experiment_name: str = "",
    score_dimensions: list = [],
    test_mode: bool = True,
    i_pipeline_cycle: int = 0,
) -> None:
    logger = logging.getLogger("aintelope.experiment")

    if "ppo" in cfg.hparams.agent_class and not test_mode:
        run_baseline_training(cfg)
        return None

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

    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    dir_out = os.path.normpath(cfg.log_dir)
    checkpoint_dir = os.path.normpath(cfg.checkpoint_dir)
    dir_cp = os.path.join(dir_out, checkpoint_dir)

    unit_test_mode = (
        cfg.hparams.unit_test_mode
    )  # is set during tests in order to speed up RL computations
    use_separate_models_for_each_experiment = (
        cfg.hparams.use_separate_models_for_each_experiment
    )

    # Agents
    agents = []
    dones = {}
    prev_agent_checkpoint = None
    for i in range(env.max_num_agents):
        agent_id = f"agent_{i}"
        agents.append(
            get_agent_class(cfg.hparams.agent_class)(
                agent_id,
                trainer,
                env,
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

        if not cfg.hparams.env_params.combine_interoception_and_vision:
            observation_shape = (observation[0].shape, observation[1].shape)
        else:
            observation_shape = observation.shape

        print(f"\nAgent {agent_id} observation shape: {observation_shape}")

        # TODO: is this reset necessary here? In main loop below,
        # there is also a reset call
        agents[-1].reset(observation, info, type(env))
        # Get latest checkpoint if existing

        checkpoint = None

        if (
            prev_agent_checkpoint
            is not None
            # and not use_separate_models_for_each_experiment   # if each experiment has separate models then the model of first agent will have same age as the model of second agent. In this case there is no reason to restrict the model of second agent to be equal of the first agent
        ):  # later experiments may have more agents    # TODO: configuration option for determining whether new agents can copy the checkpoints of earlier agents, and if so then specifically which agent's checkpoint to use
            checkpoint = prev_agent_checkpoint
        else:
            checkpoint_filename = agent_id
            if use_separate_models_for_each_experiment:
                checkpoint_filename += "-" + experiment_name
            checkpoints = glob.glob(
                os.path.join(dir_cp, checkpoint_filename + "-*")
            )  # NB! separate agent id from date explicitly in glob arguments using "-" since theoretically the agent id could be a two digit number and we do not want to match agent_10 while looking for agent_1
            if len(checkpoints) > 0:
                checkpoint = max(checkpoints, key=os.path.getctime)
                prev_agent_checkpoint = checkpoint
            elif (
                prev_agent_checkpoint is not None
            ):  # later experiments may have more agents    # TODO: configuration option for determining whether new agents can copy the checkpoints of earlier agents, and if so then specifically which agent's checkpoint to use
                checkpoint = prev_agent_checkpoint

        # Add agent, with potential checkpoint
        if "ppo" in cfg.hparams.agent_class:
            agents[-1].load_model(checkpoint)

        if not cfg.hparams.env_params.combine_interoception_and_vision:
            trainer.add_agent(
                agent_id,
                (observation[0].shape, observation[1].shape),
                env.action_space,
                unit_test_mode=unit_test_mode,
                checkpoint=checkpoint,
            )
        else:
            trainer.add_agent(
                agent_id,
                observation.shape,
                env.action_space,
                unit_test_mode=unit_test_mode,
                checkpoint=checkpoint,
            )
        dones[agent_id] = False

    # Warmup not yet implemented
    # for _ in range(hparams.warm_start_steps):
    #    agents.play_step(self.net, epsilon=1.0)

    # Main loop
    events = pd.DataFrame(
        columns=[
            "Run_id",
            "Pipeline cycle",
            "Episode",
            "Trial",
            "Step",
            "IsTest",
            "Agent_id",
            "State",
            "Action",
            "Reward",
            "Done",
            "Next_state",
        ]
        + (score_dimensions if isinstance(env, GridworldZooBaseEnv) else ["Score"])
    )

    model_needs_saving = (
        False  # if no training episodes are specified then do not save models
    )
    # num_episodes = cfg.hparams.num_episodes + cfg.hparams.test_episodes
    # num_episodes = (cfg.hparams.num_episodes if not test_mode else 0) + (cfg.hparams.test_episodes if test_mode else 0)
    # for i_episode in range(num_episodes):
    r = (
        range(cfg.hparams.num_episodes)
        if not test_mode
        else range(
            cfg.hparams.num_episodes,
            cfg.hparams.num_episodes + cfg.hparams.test_episodes,
        )
    )  # TODO: concatenate test plot in plotting.py

    with RobustProgressBar(
        max_value=len(r), disable=unit_test_mode
    ) as episode_bar:  # this is a slow task so lets use a progress bar    # note that ProgressBar crashes under unit test mode, so it will be disabled if unit_test_mode is on   # TODO: create a custom extended ProgressBar class that automatically turns itself off during unit test mode
        for i_episode in r:
            trial_no = (
                int(i_episode / cfg.hparams.trial_length)
                if cfg.hparams.trial_length > 0
                else i_episode  # this ensures that during test episodes, trial_no based map randomization seed is different from training seeds. The environment is re-constructed when testing starts. Without explicitly providing trial_no, the map randomization seed would be automatically reset to trial_no = 0, which would overlap with the training seeds.
            )

            print(
                f"\ni_pipeline_cycle: {i_pipeline_cycle} experiment: {experiment_name} episode: {i_episode} trial_no: {trial_no} test_mode: {test_mode}"
            )

            # TODO: refactor these checks into separate function        # Save models
            # https://pytorch.org/tutorials/recipes/recipes/
            # saving_and_loading_a_general_checkpoint.html
            if not test_mode:
                if i_episode > 0:
                    model_needs_saving = True
                    if i_episode % cfg.hparams.save_frequency == 0:
                        os.makedirs(dir_cp, exist_ok=True)
                        if "ppo" in cfg.hparams.agent_class:
                            for agent in agents:
                                agent.save_model()
                        else:
                            trainer.save_models(
                                i_episode,
                                dir_cp,
                                experiment_name,
                                use_separate_models_for_each_experiment,
                            )

                        model_needs_saving = False
                else:
                    model_needs_saving = True

            # Reset
            if isinstance(env, ParallelEnv):
                (
                    observations,
                    infos,
                ) = env.reset(
                    trial_no=trial_no
                )  # if not test_mode else -(trial_no - cfg.hparams.num_episodes + 1))
                for agent in agents:
                    agent.reset(observations[agent.id], infos[agent.id], type(env))
                    # trainer.reset_agent(agent.id)	# TODO: configuration flag
                    dones[agent.id] = False

            elif isinstance(env, AECEnv):
                env.reset(
                    trial_no=trial_no
                )  # if not test_mode else -(trial_no - cfg.hparams.num_episodes + 1))
                for agent in agents:
                    agent.reset(
                        env.observe(agent.id), env.observe_info(agent.id), type(env)
                    )
                    # trainer.reset_agent(agent.id)	# TODO: configuration flag
                    dones[agent.id] = False

            # Iterations within the episode
            with RobustProgressBar(
                max_value=cfg.hparams.env_params.num_iters,
                granularity=100,
                disable=unit_test_mode,
            ) as step_bar:  # this is a slow task so lets use a progress bar    # note that ProgressBar crashes under unit test mode, so it will be disabled if unit_test_mode is on
                for step in range(cfg.hparams.env_params.num_iters):
                    # if step > 0 and step % 100 == 0:
                    #    print(f"step: {step}")

                    if isinstance(env, ParallelEnv):
                        # loop: get observations and collect actions
                        actions = {}
                        for agent in agents:  # TODO: exclude terminated agents
                            observation = observations[agent.id]
                            info = infos[agent.id]
                            actions[agent.id] = agent.get_action(
                                observation,
                                info,
                                step,
                                trial_no,
                                i_episode,
                                i_pipeline_cycle,
                            )

                        # print(f"actions: {actions}")

                        # call: send actions and get observations
                        observations, scores, terminateds, truncateds, infos = env.step(
                            actions
                        )
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
                                sum(score.values())
                                if isinstance(score, dict)
                                else score,  # TODO: make a function to handle obs->rew in Q-agent too, remove this
                                done,  # TODO: should it be "terminated" in place of "done" here?
                                test_mode,
                            )

                            # Record what just happened
                            env_step_info = (
                                [
                                    score.get(dimension, 0)
                                    for dimension in score_dimensions
                                ]
                                if isinstance(score, dict)
                                else [score]
                            )

                            events.loc[len(events)] = (
                                [
                                    cfg.experiment_name,
                                    i_pipeline_cycle,
                                    i_episode,
                                    trial_no,
                                    step,
                                    test_mode,
                                ]
                                + agent_step_info
                                + env_step_info
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
                                action = agent.get_action(
                                    observation,
                                    info,
                                    step,
                                    trial_no,
                                    i_episode,
                                    i_pipeline_cycle,
                                )

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
                                    sum(score.values())
                                    if isinstance(score, dict)
                                    else score,
                                    done,  # TODO: should it be "terminated" in place of "done" here?
                                    test_mode,
                                )  # note that score is used ONLY by baseline

                                # Record what just happened
                                env_step_info = (
                                    [
                                        score.get(dimension, 0)
                                        for dimension in score_dimensions
                                    ]
                                    if isinstance(score, dict)
                                    else [score]
                                )

                                events.loc[len(events)] = (
                                    [
                                        cfg.experiment_name,
                                        i_pipeline_cycle,
                                        i_episode,
                                        trial_no,
                                        step,
                                        test_mode,
                                    ]
                                    + agent_step_info
                                    + env_step_info
                                )

                                # NB! any agent could die at any other agent's step
                                for agent_id2 in env.agents:
                                    dones[agent_id2] = (
                                        env.terminations[agent_id2]
                                        or env.truncations[agent_id2]
                                    )
                                    # TODO: if the agent died during some other agents step,
                                    # should we call agent.update() on the dead agent,
                                    # else it will be never called?

                    else:
                        raise NotImplementedError(
                            f"Unknown environment type {type(env)}"
                        )

                    # Perform one step of the optimization (on the policy network)
                    if not test_mode:
                        trainer.optimize_models()

                    # Break when all agents are done
                    if all(dones.values()):
                        step_bar.update(
                            cfg.hparams.env_params.num_iters
                        )  # TODO: maybe this line is not needed and progress bar automatically jumps to 100%
                        break

                    step_bar.update(step + 1)

                # / for step in range(cfg.hparams.env_params.num_iters):
            # / with RobustProgressBar(max_value=cfg.hparams.env_params.num_iters) as step_bar:

            episode_bar.update(i_episode + 1 - r.start)

        # / for i_episode in range(cfg.hparams.num_episodes + cfg.hparams.test_episodes):
    # / with RobustProgressBar(max_value=len(r)) as bar:

    if (
        model_needs_saving
    ):  # happens when num_episodes is not divisible by save frequency
        os.makedirs(dir_cp, exist_ok=True)
        if "ppo" in cfg.hparams.agent_class:
            for agent in agents:
                agent.save_model()
        else:
            trainer.save_models(
                i_episode,
                dir_cp,
                experiment_name,
                use_separate_models_for_each_experiment,
            )

    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname

    record_path = Path(os.path.join(experiment_dir, events_fname))
    os.makedirs(experiment_dir, exist_ok=True)
    rec.record_events(
        record_path, events
    )  # TODO: flush the events log every once a while and later append new rows


def run_baseline_training(cfg: DictConfig):
    """
    This is a copy of the above run_experiment() function
    to accommodate stable_baselines agents' training.
    """

    env = get_env_class(cfg.hparams.env)(env_params=cfg.hparams.env_params)

    # Add agents
    agents = []
    dones = {}
    for i in range(env.max_num_agents):
        agent_id = f"agent_{i}"
        agents.append(
            get_agent_class(cfg.hparams.agent_class)(
                agent_id=agent_id,
                trainer=None,
                env=env,
            )
        )

    # Train
    for i_episode in range(cfg.hparams.num_episodes):
        env.reset()  # TODO: trial_no in case multiple episodes use SAME seed
        for agent in agents:
            # WORKS ONLY FOR ONE AGENT ATM! Cutoff here to synchronize the run
            agent.train(cfg.hparams.env_params.num_iters)

    # Save models
    for agent in agents:
        agent.save_model()


# @hydra.main(version_base=None, config_path="config", config_name="config_experiment")
if __name__ == "__main__":
    run_experiment()  # TODO: cfg, score_dimensions
