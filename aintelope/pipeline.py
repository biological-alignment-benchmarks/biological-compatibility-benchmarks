import os
import copy
import logging
import sys
import torch
import gc
import time
import json
import itertools
import subprocess
import asyncio

import hydra
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten
from flatten_dict.reducers import make_reducer

from diskcache import Cache

if os.name == "nt":
    from semaphore_win_ctypes import (
        AcquireSemaphore,
        CreateSemaphore,
        OpenSemaphore,
        Semaphore,
    )
else:
    import posix_ipc

# this one is cross-platform
from filelock import FileLock

from progressbar import ProgressBar

from matplotlib import pyplot as plt

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import (
    archive_code,
    DummyContext,
    get_pipeline_score_dimensions,
    get_score_dimensions,
)
from aintelope.experiments import run_experiment


logger = logging.getLogger("aintelope.__main__")

gpu_count = max(1, torch.cuda.device_count())
worker_count_multiplier = 1  # when running pipeline search, then having more workers than GPU-s will cause all sorts of Python and CUDA errors under Windows for some reason, even though there is plenty of free RAM and GPU memory. Yet, when the pipeline processes are run manually, there is no concurrency limit. # TODO: why?
num_workers = gpu_count * worker_count_multiplier

# needs to be initialised here in order to avoid circular imports in gridsearch
cache_folder = "gridsearch_cache"
cache = Cache(cache_folder)

gridsearch_params_global = None


def aintelope_main() -> None:
    # return run_gridsearch_experiment(gridsearch_params=None)    # TODO: caching support
    run_pipeline()


# this method is used by grid search, but it needs to be in same file as run_pipeline, else sharing the gridsearch_params_global would not work
# TODO: auto-detect need for cache update then pipeline has different configuration
# @cache.memoize(    # TODO: disable this for the duration of evals since then the gridsearch parameters are nulls
#    ignore={"gridsearch_params"},
#    name="__main__.run_gridsearch_experiment_cache_helper"
# )  # use only gridsearch_params_sorted_yaml argument
def run_gridsearch_experiment_cache_helper(
    gridsearch_params: DictConfig, gridsearch_params_sorted_yaml: str
) -> None:  # NB! do not rename this function, else cache will be invalidated
    global gridsearch_params_global

    # cfg.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")    # TODO: re-parse date format from config file

    gridsearch_params_global = gridsearch_params  # TODO: hydra main does not allow multiple arguments, probably there is a more typical way to do it
    test_summaries = run_pipeline()
    return test_summaries  # this result will be cached


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def run_pipeline(cfg: DictConfig) -> None:
    gridsearch_params_in = gridsearch_params_global  # TODO: hydra main does not allow multiple arguments, but probably there is a more typical way to do it
    do_not_show_plot = gridsearch_params_in is not None

    timestamp = str(cfg.timestamp)
    timestamp_pid_uuid = str(cfg.timestamp_pid_uuid)
    logger.info(f"timestamp: {timestamp}")
    logger.info(f"timestamp_pid_uuid: {timestamp_pid_uuid}")

    archive_code(cfg)

    # TODO: ensure to do not use special pipeline config when doing initial gridsearch
    pipeline_config_file = os.environ.get("PIPELINE_CONFIG")
    if pipeline_config_file is None:
        pipeline_config_file = "config_pipeline.yaml"
    pipeline_config = OmegaConf.load(
        os.path.join("aintelope/config", pipeline_config_file)
    )

    test_summaries_to_return = []
    test_summaries_to_jsonl = []

    # use additional semaphore here (in addition to gridsearch multiprocessing worker count) since the user may launch multiple gridsearch as well as non-gridsearch processes manually
    semaphore_name = (
        "AIntelope_pipeline_semaphore"
        + (
            "_" + cfg.hparams.params_set_title
            if cfg.hparams.params_set_title in ["instinct", "random"]
            else ""
        )
        + ("_debug" if sys.gettrace() is not None else "")
    )
    if gridsearch_params_in is None:
        print("Waiting for semaphore...")
    # TODO: refactor this semaphore code into a shared class that handles both Linux and Windows
    with (
        CreateSemaphore(semaphore_name, maximum_count=num_workers)
        if os.name == "nt" and gridsearch_params_in is None
        else DummyContext()
    ) as semaphore:
        with AcquireSemaphore(
            semaphore
        ) if os.name == "nt" and gridsearch_params_in is None else DummyContext():
            with (
                posix_ipc.Semaphore(
                    semaphore_name, flags=posix_ipc.O_CREAT, initial_value=num_workers
                )
                if os.name != "nt" and gridsearch_params_in is None
                else DummyContext()
            ):
                if gridsearch_params_in is None:
                    print("Semaphore acquired...")

                max_pipeline_cycle = (
                    cfg.hparams.num_pipeline_cycles + 1
                    if cfg.hparams.num_pipeline_cycles >= 1
                    else 1
                )  # Last +1 cycle is for testing. In case of 0 pipeline cycle, run testing inside the same cycle immediately after each environment's training ends.
                with ProgressBar(
                    max_value=max_pipeline_cycle
                ) as pipeline_cycle_bar:  # this is a slow task so lets use a progress bar
                    for i_pipeline_cycle in range(0, max_pipeline_cycle):
                        is_last_pipeline_cycle = (
                            i_pipeline_cycle == cfg.hparams.num_pipeline_cycles
                        )

                        with ProgressBar(
                            max_value=len(pipeline_config)
                        ) as pipeline_bar:  # this is a slow task so lets use a progress bar
                            for env_conf_i, env_conf_name in enumerate(pipeline_config):
                                experiment_cfg = copy.deepcopy(
                                    cfg
                                )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
                                OmegaConf.update(
                                    experiment_cfg, "experiment_name", env_conf_name
                                )

                                OmegaConf.update(
                                    experiment_cfg,
                                    "hparams",
                                    pipeline_config[env_conf_name],
                                    force_add=True,
                                )

                                if gridsearch_params_in is not None:
                                    gridsearch_params = copy.deepcopy(
                                        gridsearch_params_in
                                    )
                                    # replace all null-valued params with params from pipeline config, and do aggregated results file check and reporting using these replaced values
                                    gridsearch_params_dict = OmegaConf.to_container(
                                        gridsearch_params, resolve=False
                                    )
                                    flattened_gridsearch_params = flatten(
                                        gridsearch_params_dict,
                                        reducer=make_reducer(delimiter="."),
                                    )  # convert to format {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}
                                    null_entry_keys = [
                                        key
                                        for key, value in flattened_gridsearch_params.items()
                                        if value is None
                                    ]

                                    # OmegaConf does not support dot-path style access, so need to use flattened config dict for that.  # TODO: create a helper method for this instead
                                    pipeline_config_dict = OmegaConf.to_container(
                                        pipeline_config[env_conf_name],
                                        resolve=False,  # do not resolve yet - loop over null valued entries only, not including references to them
                                    )
                                    flattened_pipeline_config = flatten(
                                        pipeline_config_dict,
                                        reducer=make_reducer(delimiter="."),
                                    )  # convert to format {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}

                                    for null_entry_key in null_entry_keys:
                                        value = flattened_pipeline_config[
                                            null_entry_key[len("hparams.") :]
                                        ]
                                        OmegaConf.update(
                                            gridsearch_params,
                                            null_entry_key,
                                            value,
                                            force_add=False,
                                        )

                                    OmegaConf.update(
                                        experiment_cfg,
                                        "hparams",
                                        gridsearch_params.hparams,
                                        force_add=True,
                                    )

                                # check whether this experiment has already been run during an earlier or aborted pipeline run
                                if cfg.hparams.aggregated_results_file:
                                    aggregated_results_file = os.path.normpath(
                                        cfg.hparams.aggregated_results_file
                                    )
                                    if os.path.exists(aggregated_results_file):
                                        aggregated_results_file_lock = FileLock(
                                            aggregated_results_file + ".lock"
                                        )
                                        with aggregated_results_file_lock:
                                            with open(
                                                aggregated_results_file,
                                                mode="r",
                                                encoding="utf-8",
                                            ) as fh:
                                                data = fh.read()

                                        gridsearch_params_dict = OmegaConf.to_container(
                                            gridsearch_params, resolve=True
                                        )

                                        test_summaries2 = []
                                        lines = data.split("\n")
                                        for line in lines:
                                            line = line.strip()
                                            if len(line) == 0:
                                                continue
                                            test_summary = json.loads(line)
                                            if (
                                                test_summary["experiment_name"]
                                                == env_conf_name
                                                and test_summary["gridsearch_params"]
                                                == gridsearch_params_dict
                                            ):  # Python's dictionary comparison is order independent and works with nested dictionaries as well
                                                test_summaries2.append(test_summary)
                                            else:
                                                qqq = True  # for debugging

                                        if len(test_summaries2) > 0:
                                            assert len(test_summaries2) == 1
                                            test_summaries_to_return.append(
                                                test_summaries2[0]
                                            )  # NB! do not add to test_summaries_to_jsonl, else it will be duplicated in the jsonl file
                                            pipeline_bar.update(env_conf_i + 1)
                                            print(
                                                f"\nSkipping experiment that is already in jsonl file: {env_conf_name}"
                                            )
                                            continue

                                    # / if os.path.exists(aggregated_results_file):
                                # / if cfg.hparams.aggregated_results_file:

                                logger.info(
                                    "Running training with the following configuration"
                                )
                                logger.info(
                                    os.linesep
                                    + str(
                                        OmegaConf.to_yaml(experiment_cfg, resolve=True)
                                    )
                                )

                                # Training
                                params_set_title = (
                                    experiment_cfg.hparams.params_set_title
                                )
                                logger.info(
                                    f"params_set: {params_set_title}, experiment: {env_conf_name}"
                                )

                                score_dimensions = get_score_dimensions(experiment_cfg)

                                if (
                                    cfg.hparams.num_pipeline_cycles == 0
                                ):  # in case of 0 pipeline cycle, run testing inside the same cycle immediately after each environment's training ends.
                                    run_experiment(
                                        experiment_cfg,
                                        experiment_name=env_conf_name,
                                        score_dimensions=score_dimensions,
                                        is_last_pipeline_cycle=False,
                                        i_pipeline_cycle=i_pipeline_cycle,
                                    )

                                run_experiment(
                                    experiment_cfg,
                                    experiment_name=env_conf_name,
                                    score_dimensions=score_dimensions,
                                    is_last_pipeline_cycle=is_last_pipeline_cycle,
                                    i_pipeline_cycle=i_pipeline_cycle,
                                )

                                # torch.cuda.empty_cache()
                                # gc.collect()

                                if is_last_pipeline_cycle:
                                    # Not using timestamp_pid_uuid here since it would make the title too long. In case of manual execution with plots, the pid-uuid is probably not needed anyway.
                                    title = (
                                        timestamp
                                        + " : "
                                        + params_set_title
                                        + " : "
                                        + env_conf_name
                                    )
                                    test_summary = analytics(
                                        experiment_cfg,
                                        score_dimensions,
                                        title=title,
                                        experiment_name=env_conf_name,
                                        group_by_pipeline_cycle=cfg.hparams.num_pipeline_cycles
                                        >= 1,
                                        gridsearch_params=gridsearch_params,
                                        do_not_show_plot=do_not_show_plot,
                                    )
                                    test_summaries_to_return.append(test_summary)
                                    test_summaries_to_jsonl.append(test_summary)

                                pipeline_bar.update(env_conf_i + 1)

                            # / for env_conf_name in pipeline_config:
                        # / with ProgressBar(max_value=len(pipeline_config)) as pipeline_bar:

                        pipeline_cycle_bar.update(i_pipeline_cycle + 1)

                    # / for i_pipeline_cycle in range(0, max_pipeline_cycle):
                # / with ProgressBar(max_value=max_pipeline_cycle) as pipeline_cycle_bar:
            # / with posix_ipc.Semaphore():
        # / with AcquireSemaphore(created, timeout_ms=0):
    # / with CreateSemaphore('name', maximum_count=num_workers) as semaphore:

    # Write the pipeline results to file only when entire pipeline has run. Else crashing the program during pipeline run will cause the aggregated results file to contain partial data which will be later duplicated by re-run.
    # TODO: alternatively, cache the results of each experiment separately
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)
        aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
        with aggregated_results_file_lock:
            with open(aggregated_results_file, mode="a", encoding="utf-8") as fh:
                for test_summary in test_summaries_to_jsonl:
                    # Do not write directly to file. If JSON serialization error occurs during json.dump() then a broken line would be written into the file (I have verified this). Therefore using json.dumps() is safer.
                    json_text = json.dumps(test_summary)
                    fh.write(
                        json_text + "\n"
                    )  # \n : Prepare the file for appending new lines upon subsequent append. The last character in the JSONL file is allowed to be a line separator, and it will be treated the same as if there was no line separator present.
                fh.flush()

    torch.cuda.empty_cache()
    gc.collect()

    # keep plots visible until the user decides to close the program
    if not do_not_show_plot:
        if os.name == "nt":
            import msvcrt

            print("\nPipeline done. Press [enter] to continue.")
            msvcrt.getch()  # uses less CPU on Windows than input() function. Note that the graph window will be frozen, but will still show graphs
            # while True:
            #    if msvcrt.kbhit():
            #        break
            #    plt.pause(60)
            #    # time.sleep(1)
        else:
            input("\nPipeline done. Press [enter] to continue.")

    return test_summaries_to_return


def analytics(
    cfg,
    score_dimensions,
    title,
    experiment_name,
    group_by_pipeline_cycle,
    gridsearch_params=DictConfig,
    do_not_show_plot=False,
):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    log_dir = os.path.normpath(cfg.log_dir)
    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname
    num_train_episodes = cfg.hparams.num_episodes
    num_train_pipeline_cycles = cfg.hparams.num_pipeline_cycles

    savepath = os.path.join(log_dir, "plot_" + experiment_name)
    events = recording.read_events(experiment_dir, events_fname)

    (
        test_totals,
        test_averages,
        test_variances,
        test_sfella_totals,
        test_sfella_averages,
        test_sfella_variances,
        sfella_score_total,
        sfella_score_average,
        sfella_score_variance,
        score_dimensions_out,
    ) = plotting.aggregate_test_scores(
        events,
        num_train_pipeline_cycles,
        score_dimensions,
        group_by_pipeline_cycle=group_by_pipeline_cycle,
    )

    test_summary = {
        "timestamp": cfg.timestamp,
        "timestamp_pid_uuid": cfg.timestamp_pid_uuid,
        "experiment_name": experiment_name,
        "title": title,  # timestamp + " : " + params_set_title + " : " + env_conf_name
        "params_set_title": cfg.hparams.params_set_title,
        "gridsearch_params": OmegaConf.to_container(gridsearch_params, resolve=True)
        if gridsearch_params is not None
        else None,  # Object of type DictConfig is not JSON serializable, neither can yaml.dump in plotting.prettyprint digest it, so need to convert it to ordinary dictionary
        "num_train_pipeline_cycles": num_train_pipeline_cycles,
        "score_dimensions": score_dimensions_out,
        "group_by_pipeline_cycle": group_by_pipeline_cycle,
        "test_totals": test_totals,
        "test_averages": test_averages,
        "test_variances": test_variances,
        # per score dimension results
        "test_sfella_totals": test_sfella_totals,
        "test_sfella_averages": test_sfella_averages,
        "test_sfella_variances": test_sfella_variances,
        # over score dimensions results
        # TODO: rename to test_*
        "sfella_score_total": sfella_score_total,
        "sfella_score_average": sfella_score_average,
        "sfella_score_variance": sfella_score_variance,
    }

    plotting.prettyprint(test_summary)

    plotting.plot_performance(
        events,
        num_train_episodes,
        num_train_pipeline_cycles,
        score_dimensions,
        save_path=savepath,
        title=title,
        group_by_pipeline_cycle=group_by_pipeline_cycle,
        do_not_show_plot=do_not_show_plot,
    )

    return test_summary
