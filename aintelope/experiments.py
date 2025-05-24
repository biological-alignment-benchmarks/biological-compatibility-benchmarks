# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import glob
import logging
import os
from pathlib import Path
import gc

import pandas as pd
from omegaconf import DictConfig

from aintelope.utils import RobustProgressBar

from aintelope.agents import get_agent_class
from aintelope.analytics import recording
from aintelope.environments import get_env_class
from aintelope.environments.savanna_safetygrid import GridworldZooBaseEnv
from aintelope.training.dqn_training import Trainer

from typing import Any, Union
import gymnasium as gym
from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


def run_experiment(
    cfg: DictConfig,
    experiment_name: str = "",  # TODO: remove this argument and read it from cfg.experiment_name
    score_dimensions: list = [],
    test_mode: bool = True,
    i_pipeline_cycle: int = 0,
    num_actual_train_episodes: int = -1,
) -> None:
    if "trial_length" in cfg:  # backwards compatibility
        cfg.env_layout_seed_repeat_sequence_length = cfg.trial_length
    if "eps_last_env_layout_seed" in cfg:  # backwards compatibility
        cfg.eps_last_env_layout_seed = cfg.eps_last_env_layout_seed

    logger = logging.getLogger("aintelope.experiment")

    is_sb3 = cfg.hparams.agent_class.startswith("sb3_")

    # Environment
    env = get_env_class(cfg.hparams.env)(
        env_params=cfg.hparams.env_params,
        ignore_num_iters=not is_sb3
        or test_mode,  # NB! this file implements its own iterations bookkeeping in order to allow the agent to learn from the last step
        scalarize_rewards=is_sb3 and not test_mode,
    )

    # This reset here does not increment episode number since no steps are played before one more reset in the main episode loop takes place
    if isinstance(env, ParallelEnv):
        (
            observations,
            infos,
        ) = env.reset()
    elif isinstance(env, AECEnv):
        env.reset()
    else:
        raise NotImplementedError(f"Unknown environment type {type(env)}")

    events_columns = [
        "Run_id",
        "Pipeline cycle",
        "Episode",
        "Env layout seed",
        "Step",
        "IsTest",
        "Agent_id",
        "State",
        "Action",
        "Reward",
        "Done",
        "Next_state",
    ] + (score_dimensions if isinstance(env, GridworldZooBaseEnv) else ["Score"])

    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname
    events = recording.EventLog(experiment_dir, events_fname, events_columns)

    # Common trainer for each agent's models
    if is_sb3:
        trainer = None
    else:
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
        agent = get_agent_class(cfg.hparams.agent_class)(
            agent_id=agent_id,
            trainer=trainer,
            env=env,
            cfg=cfg,
            test_mode=test_mode,
            **cfg.hparams.agent_params,
        )
        agents.append(agent)

        if is_sb3:
            agent.i_pipeline_cycle = i_pipeline_cycle
            agent.events = events
            agent.score_dimensions = score_dimensions

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
        agent.reset(observation, info, type(env))
        # Get latest checkpoint if existing

        checkpoint = None

        if (
            cfg.hparams.model_params.use_weight_sharing  # The reasoning for this condition is that even if the agents have similar roles, they still see each other on different observation layers. For example, the agent 0 is always on layer 0 and agent 1 is alwyas on layer 1, regardless of which agent is observing. Thus if they were trained on separate models then they need separate models also during test. When PPO weight sharing is enabled, then this is not an issue, because the shared model will learn to differentiate between the agents by looking at the agent currently visible in the center of the observation field. TODO: Swap the self-agent and other-agent layers in the environment side in such a manner that self-agent is always at same layer index for all agents and other-agent is also always at same layer index for all agents. Then we can enable this conditional branch for other models as well in scenarios where the agents have symmetric roles.
            and prev_agent_checkpoint is not None
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
            elif (
                test_mode
                and not cfg.hparams.do_not_enforce_checkpoint_file_existence_during_test
            ):
                raise Exception("No trained model found, cannot run test!")

        # Add agent, with potential checkpoint
        if not cfg.hparams.env_params.combine_interoception_and_vision:
            agent.init_model(
                (observation[0].shape, observation[1].shape),
                env.action_space,
                unit_test_mode=unit_test_mode,
                checkpoint=checkpoint,
            )
        else:
            agent.init_model(
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

    if is_sb3 and not test_mode:
        num_actual_train_episodes = run_baseline_training(
            cfg, i_pipeline_cycle, env, agents
        )

    else:
        model_needs_saving = (
            False  # if no training episodes are specified then do not save models
        )

        if (
            not test_mode
        ):  # non-SB3 loop uses the exact number of episodes specified in the config
            num_actual_train_episodes = cfg.hparams.num_episodes
        elif num_actual_train_episodes == -1:  # and test_mode
            # NB! PPO doing extra episodes causes the episode counting for test episodes to collide with train episodes, therefore need to offset the test episode numbers
            num_actual_train_episodes = cfg.hparams.num_episodes

        # num_episodes = cfg.hparams.num_episodes + cfg.hparams.test_episodes
        # num_episodes = (cfg.hparams.num_episodes if not test_mode else 0) + (cfg.hparams.test_episodes if test_mode else 0)
        # for i_episode in range(num_episodes):
        r = (
            range(cfg.hparams.num_episodes)
            if not test_mode
            else range(
                num_actual_train_episodes,
                num_actual_train_episodes + cfg.hparams.test_episodes,
            )
        )  # TODO: concatenate test plot in plotting.py

        with RobustProgressBar(
            max_value=len(r), disable=unit_test_mode
        ) as episode_bar:  # this is a slow task so lets use a progress bar    # note that ProgressBar crashes under unit test mode, so it will be disabled if unit_test_mode is on   # TODO: create a custom extended ProgressBar class that automatically turns itself off during unit test mode
            for i_episode in r:
                events.flush()

                env_layout_seed = (
                    int(
                        i_episode / cfg.hparams.env_layout_seed_repeat_sequence_length
                    )  # TODO ensure different env_layout_seed during test when num_actual_train_episodes is not divisible by env_layout_seed_repeat_sequence_length
                    if cfg.hparams.env_layout_seed_repeat_sequence_length > 0
                    else i_episode  # this ensures that during test episodes, env_layout_seed based map randomization seed is different from training seeds. The environment is re-constructed when testing starts. Without explicitly providing env_layout_seed, the map randomization seed would be automatically reset to env_layout_seed = 0, which would overlap with the training seeds.
                )

                print(
                    f"\ni_pipeline_cycle: {i_pipeline_cycle} experiment: {experiment_name} episode: {i_episode} env_layout_seed: {env_layout_seed} test_mode: {test_mode}"
                )

                # TODO: refactor these checks into separate function        # Save models
                # https://pytorch.org/tutorials/recipes/recipes/
                # saving_and_loading_a_general_checkpoint.html
                if not test_mode:
                    if (
                        i_episode > 0 and cfg.hparams.save_frequency != 0
                    ):  # cfg.hparams.save_frequency == 0 means that the model is saved only at the end, improving training performance
                        model_needs_saving = True
                        if i_episode % cfg.hparams.save_frequency == 0:
                            os.makedirs(dir_cp, exist_ok=True)
                            for agent in agents:
                                agent.save_model(
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
                        env_layout_seed=env_layout_seed
                    )  # if not test_mode else -(env_layout_seed - cfg.hparams.num_episodes + 1))
                    for agent in agents:
                        agent.reset(observations[agent.id], infos[agent.id], type(env))
                        # trainer.reset_agent(agent.id)	# TODO: configuration flag
                        dones[agent.id] = False

                elif isinstance(env, AECEnv):
                    env.reset(  # TODO: actually savanna_safetygrid wrapper provides observations and infos as a return value, so need for branching here
                        env_layout_seed=env_layout_seed
                    )  # if not test_mode else -(env_layout_seed - cfg.hparams.num_episodes + 1))
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
                                    observation=observation,
                                    info=info,
                                    step=step,
                                    env_layout_seed=env_layout_seed,
                                    episode=i_episode,
                                    pipeline_cycle=i_pipeline_cycle,
                                )

                            # print(f"actions: {actions}")

                            # call: send actions and get observations
                            (
                                observations,
                                scores,
                                terminateds,
                                truncateds,
                                infos,
                            ) = env.step(actions)
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
                                    env=env,
                                    observation=observation,
                                    info=info,
                                    score=sum(score.values())
                                    if isinstance(score, dict)
                                    else score,  # TODO: make a function to handle obs->rew in Q-agent too, remove this
                                    done=done,  # TODO: should it be "terminated" in place of "done" here?
                                    test_mode=test_mode,
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

                                events.log_event(
                                    [
                                        cfg.experiment_name,
                                        i_pipeline_cycle,
                                        i_episode,
                                        env_layout_seed,
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
                                if (
                                    env.terminations[agent.id]
                                    or env.truncations[agent.id]
                                ):
                                    action = None
                                else:
                                    observation = env.observe(agent.id)
                                    info = env.observe_info(agent.id)
                                    action = agent.get_action(
                                        observation=observation,
                                        info=info,
                                        step=step,
                                        env_layout_seed=env_layout_seed,
                                        episode=i_episode,
                                        pipeline_cycle=i_pipeline_cycle,
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
                                        env=env,
                                        observation=observation,
                                        info=info,
                                        score=sum(score.values())
                                        if isinstance(score, dict)
                                        else score,
                                        done=done,  # TODO: should it be "terminated" in place of "done" here?
                                        test_mode=test_mode,
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

                                    events.log_event(
                                        [
                                            cfg.experiment_name,
                                            i_pipeline_cycle,
                                            i_episode,
                                            env_layout_seed,
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
            for agent in agents:
                agent.save_model(
                    i_episode,
                    dir_cp,
                    experiment_name,
                    use_separate_models_for_each_experiment,
                )

    # / if is_sb3 and not test_mode:

    events.close()

    gc.collect()

    return num_actual_train_episodes


def run_baseline_training(
    cfg: DictConfig, i_pipeline_cycle: int, env: Environment, agents: list
):
    # SB3 models are designed for single-agent settings, we get around this by using the same model for every agent
    # https://pettingzoo.farama.org/tutorials/sb3/waterworld/

    # TODO: we could still allow multiple agents WITH SEPARATE MODELS training if we use an appropriate env wrapper

    unit_test_mode = cfg.hparams.unit_test_mode

    # num_total_steps = cfg.hparams.env_params.num_iters * 1
    num_total_steps = cfg.hparams.env_params.num_iters * cfg.hparams.num_episodes

    # During multi-agent multi-model training the actual agents will run in threads/subprocesses because SB3 requires Gym interface. Agent[0] will be used just as an interface to call train(), the SB3BaseAgent base class will automatically set up the actual agents.
    # In case of multi-agent weight-shared model training it is partially similar: Agent[0] will be used just as an interface to call train(), the SB3 weight-shared model will handle the actual agents present in the environment.

    with RobustProgressBar(
        max_value=num_total_steps,  # TODO: somehow obtain the total number of steps PPO plans to actually take, considering that it rounds the number of steps up with some logic. The rounding up seems to be same every time, so it does not depend on the events happending during training.
        granularity=100,
        disable=unit_test_mode,
    ) as step_bar:  # this is a slow task so lets use a progress bar    # note that ProgressBar crashes under unit test mode, so it will be disabled if unit_test_mode is on
        agents[0].progressbar = step_bar
        agents[0].train(num_total_steps)

    # Save models
    # for agent in agents:
    #    agent.save_model()
    agents[0].save_model()

    # NB! PPO doing extra episodes causes the episode index counting for test episodes to collide with train episodes, therefore need to offset the test episode numbers
    num_actual_train_episodes = agents[
        0
    ].next_episode_no  # We assume that the last episode is not followed by a reset. Cannot use env.get_episode_no() here since its counter is reset for each new env_layout_seed.
    return num_actual_train_episodes


# @hydra.main(version_base=None, config_path="config", config_name="config_experiment")
if __name__ == "__main__":
    run_experiment()  # TODO: cfg, score_dimensions
