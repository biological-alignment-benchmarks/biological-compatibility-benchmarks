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
from collections import OrderedDict

import hydra

from aintelope.config.config_utils import (
    register_resolvers,
    select_gpu,
    set_memory_limits,
    set_priorities,
)
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten
from flatten_dict.reducers import make_reducer

from aintelope.utils import RobustProgressBar, wait_for_enter

from aintelope.analytics import plotting, recording
from aintelope.pipeline import (
    num_workers,
    gpu_count,
    worker_count_multiplier,
    cache,
    run_pipeline,
    run_gridsearch_experiment_cache_helper,
)


def aintelope_main() -> None:
    # return run_gridsearch_experiment(gridsearch_params=None)    # TODO: caching support
    run_pipeline()


async def run_gridsearch_experiments() -> None:
    use_multiprocessing = sys.gettrace() is None  # not debugging
    # use_multiprocessing = False
    if (
        num_workers == 1
    ):  # no need for multiprocessing if only one experiment is run at a time
        use_multiprocessing = False

    gridsearch_config_file = os.environ.get("GRIDSEARCH_CONFIG")
    # if gridsearch_config is None:
    #    gridsearch_config = "initial_config_gridsearch.yaml"
    initial_config_gridsearch = OmegaConf.load(
        os.path.join("aintelope/config", gridsearch_config_file)
    )

    # extract list parameters and compute cross product over their values
    dict_config = OmegaConf.to_container(
        initial_config_gridsearch, resolve=False
    )  # convert DictConfig to dict # NB! do NOT resolve references here since we do NOT want to handle references to lists as lists. Gridsearch should loop over each list only once.
    flattened_config = flatten(
        dict_config, reducer=make_reducer(delimiter=".")
    )  # convert to format {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}
    list_entries = OrderedDict(
        sorted(
            {
                key: value
                for key, value in flattened_config.items()
                if isinstance(value, list)
            }.items()
        )
    )  # select only entries of list type
    list_entries[
        "hparams.gridsearch_trial_no"
    ] = (
        initial_config_gridsearch.hparams.gridsearch_trial_no
    )  # this is a OmegaConf resolver that generates a list

    # create outer product of all list entries stored in the dictionary values
    # http://stephantul.github.io/python/2019/07/20/product-dict/
    keys, values = zip(
        *list_entries.items()
    )  # this performs unzip - split dictionary in to list of keys and list of values
    values_combinations = list(itertools.product(*values))
    with RobustProgressBar(
        max_value=len(values_combinations)
    ) as multiprocessing_bar:  # this is a slow task so lets use a progress bar
        active_coroutines = set()
        completed_coroutine_count = 0
        available_gpus = (
            list(range(0, gpu_count)) * worker_count_multiplier
        )  # repeat gpu index list for worker_count_multiplier times
        coroutine_gpus = {}

        for values_combination_i, values_combination in enumerate(
            values_combinations
        ):  # iterate over value combinations
            gridsearch_combination = dict(
                zip(keys, values_combination)
            )  # zip keys with values in current combination

            # print("gridsearch_combination:")
            gridsearch_combination_for_print = {
                key: value
                for key, value in gridsearch_combination.items()
                if len(flattened_config[key]) > 1
            }  # print only entries of lists that had more than one value in the gridsearch configuration, that is, ignore nested lists which are used for "list escaping" purposes
            # plotting.prettyprint(gridsearch_combination_for_print)

            gridsearch_params = copy.deepcopy(initial_config_gridsearch)
            for key, value in gridsearch_combination.items():
                OmegaConf.update(gridsearch_params, key, value, force_add=True)

            if use_multiprocessing:
                # for each next experiment select next available GPU to maximally balance the load considering multiple running processes
                use_gpu = available_gpus.pop(0)

                arguments = {
                    "gridsearch_params": gridsearch_params,
                    "gridsearch_combination_for_print": gridsearch_combination_for_print,
                    "args": sys.argv,
                    "do_not_create_subprocess": False,
                    "environ": dict(os.environ),
                    "use_gpu_index": use_gpu,
                }
                coroutine = asyncio.create_task(
                    run_gridsearch_experiment_multiprocess(**arguments)
                )  # NB! do not await here yet, awaiting will be done below by waiting for a group of coroutines at once.
                coroutine_gpus[coroutine] = use_gpu

                active_coroutines.add(coroutine)
                if len(active_coroutines) == num_workers:
                    dones, pendings = await asyncio.wait(
                        active_coroutines, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in dones:
                        available_gpus.append(coroutine_gpus[task])
                        del coroutine_gpus[task]

                        ex = (
                            task.exception()
                        )  # https://rotational.io/blog/spooky-asyncio-errors-and-how-to-fix-them/
                        if ex is not None:
                            print(f"\nError in experiment. Exception: {ex}")

                    completed_coroutine_count += len(dones)
                    multiprocessing_bar.update(completed_coroutine_count)
                    active_coroutines = pendings

            else:  # / if use_multiprocessing:
                arguments = {
                    "gridsearch_params": gridsearch_params,
                    "gridsearch_combination_for_print": gridsearch_combination_for_print,
                    "args": sys.argv,
                    "do_not_create_subprocess": True,
                    "environ": dict(os.environ),
                    "use_gpu_index": None,
                }
                try:
                    await run_gridsearch_experiment_multiprocess(**arguments)
                except Exception as ex:
                    print(
                        f"\nError in experiment. Exception: {ex}. params: {plotting.prettyprint(gridsearch_combination_for_print)}"
                    )

                completed_coroutine_count += 1
                multiprocessing_bar.update(completed_coroutine_count)

            # / if use_multiprocessing:

        # / for values_combination_i, values_combination in enumerate(values_combinations):

        # wait for remaining coroutines
        while len(active_coroutines) > 0:
            dones, pendings = await asyncio.wait(
                active_coroutines, return_when=asyncio.FIRST_COMPLETED
            )
            for task in dones:
                ex = (
                    task.exception()
                )  # https://rotational.io/blog/spooky-asyncio-errors-and-how-to-fix-them/
                if ex is not None:
                    print(f"\nError in experiment. Exception: {ex}")
            completed_coroutine_count += len(dones)
            multiprocessing_bar.update(completed_coroutine_count)
            active_coroutines = pendings

    # / with RobustProgressBar(max_value=len(values_combinations)) as multiprocessing_bar:

    wait_for_enter("Gridsearch done. Press [enter] to continue.")
    return


subprocess_exec_lock = asyncio.Lock()


async def run_gridsearch_experiment_multiprocess(
    gridsearch_params: DictConfig,
    gridsearch_combination_for_print: dict,
    args: list = None,
    do_not_create_subprocess: bool = False,
    environ: dict = {},
    use_gpu_index: int = None,
) -> None:
    """Use multiprocessing to conveniently queue the jobs and wait for their results in parallel."""

    # do not start subprocess if the result is already in cache
    cache_key = get_run_gridsearch_experiment_cache_helper_cache_key(gridsearch_params)

    # enable this the commented out lines of code below if you want to remove a some sets from the cache and recompute it
    # delete_param_sets = [
    #    {'hparams': {'gridsearch_trial_no': 0, 'params_set_title': 'mixed', 'batch_size': 16, 'lr': 0.015, 'amsgrad': True, 'use_separate_models_for_each_experiment': True, 'model_params': {'hidden_sizes': [8, 16, 8], 'num_conv_layers': 2, 'conv_size': 2, 'gamma': 0.9, 'tau': 0.05, 'eps_start': 0.66, 'eps_end': 0.0, 'instinct_bias_epsilon_start': 0.5, 'instinct_bias_epsilon_end': 0.0, 'apply_instinct_eps_before_random_eps': True, 'replay_size': 99, 'eps_last_pipeline_cycle': 1, 'eps_last_episode': 30, 'eps_last_trial': -1, 'eps_last_frame': 400}, 'trial_length': -1, 'num_pipeline_cycles': 0, 'num_episodes': 30, 'test_episodes': 10, 'env_params': {'num_iters': 400, 'map_max': 7, 'map_width': 7, 'map_height': 7, 'render_agent_radius': 4}}},
    # ]

    # gridsearch_params_dict = OmegaConf.to_container(gridsearch_params, resolve=True)
    # delete = gridsearch_params_dict in delete_param_sets

    # if delete:
    # cache.delete(cache_key)

    # if not do_not_create_subprocess and cache_key in cache:   # if multiprocessing is disabled then skip cache key checking here and proceed to the run_gridsearch_experiment function, which will decide whether to use cache or generate or regenerate
    if cache_key in cache:
        print("\nSkipping cached gridsearch_combination:")
        plotting.prettyprint(gridsearch_combination_for_print)
        return
    else:
        # NB! this message is printed only once multiprocessing queue gets to this job
        print("\nStarting gridsearch_combination:")
        plotting.prettyprint(gridsearch_combination_for_print)

    if do_not_create_subprocess:
        run_gridsearch_experiment(gridsearch_params)
    else:
        # start subprocess and wait for its completion
        # NB! cannot send the params directly from command line since this params set is partial
        # and we want the subprocess also "see" that partial configuration first.
        # We do not want to let the subprocess to see the full params set since we need only
        # essential gridsearch params fields as diskcache key.
        # Need to use json instead of yaml since environment variables do not allow newlines and
        # yaml format would contain newlines.
        gridsearch_params_json = json.dumps(
            OmegaConf.to_container(
                gridsearch_params, resolve=cache_key is not None
            )  # NB! do not resolve the param values when doing further cycles based on best parameters from grid search. In this case if the referred valus contain nulls then we want to handle that inside pipeline loop first.
        )
        env = dict(
            environ
        )  # clone before modifying  # NB! need to pass whole environ, else the program may not start    # TODO: recheck with current implementation using asyncio.create_subprocess_exec
        # env.pop("CUDA_MODULE_LOADING", None)    # main process does not have this environment variable set, but for some reason os.environ contains CUDA_MODULE_LOADING=LAZY
        env["GRIDSEARCH_PARAMS"] = gridsearch_params_json
        if use_gpu_index is not None:
            env["AINTELOPE_GPU"] = str(use_gpu_index)
        env["PYTHONUNBUFFERED"] = "1"  # disables console buffering in the subprocess
        # TODO: use multiprocessing and keep the subprocesses alive for all time? But for that to work you need to somehow ensure that the multiprocessing subprocesses start with 30 sec intervals, else there will be crashes under Windows.
        async with subprocess_exec_lock:
            proc = await asyncio.create_subprocess_exec(
                "python",
                *args,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            if os.name == "nt":
                await asyncio.sleep(
                    30  # TODO: config parameter for the delay
                )  # there needs to be delay between subprocess creation, else CUDA will crash for some reason. Additionally, even when CUDA does not crash, then python will crash when processes start in a "too short" sequence (like 10 sec intervals). Source: personal experience with multiple Windows machines and operating system versions.

        try:
            # TODO: tee subprocess output during its execution using https://github.com/thearchitector/tee-subprocess
            while proc.returncode is None:
                stdout, stderr = await proc.communicate()
                print("\n" + stdout.decode("utf-8", "ignore"))
            stdout, stderr = await proc.communicate()
            print("\n" + stdout.decode("utf-8", "ignore"))
        except Exception as ex:
            print(f"\nError in experiment worker process. Exception: {ex}. Params:")
            plotting.prettyprint(gridsearch_combination_for_print)

        return


def run_gridsearch_experiment_subprocess(gridsearch_params_json: str) -> None:
    """Use subprocesses to run actual computations, since CUDA does not work in multiprocessing processes."""

    gridsearch_params_dict = json.loads(gridsearch_params_json)
    gridsearch_params = OmegaConf.create(gridsearch_params_dict)

    print("Running subprocess with params:")
    plotting.prettyprint(gridsearch_params_dict)

    return run_gridsearch_experiment(gridsearch_params=gridsearch_params)


def run_gridsearch_experiment(gridsearch_params: DictConfig) -> None:
    """Prepares call to run_gridsearch_experiment_cache_helper which does actual caching"""

    gridsearch_params_sorted_yaml = OmegaConf.to_yaml(
        gridsearch_params, sort_keys=True, resolve=True
    )

    result = run_gridsearch_experiment_cache_helper(
        gridsearch_params=gridsearch_params,
        gridsearch_params_sorted_yaml=gridsearch_params_sorted_yaml,
    )
    return result


# Actual cache is on run_game function, here we prepare the engine_conf and cache_version arguments.
def get_run_gridsearch_experiment_cache_helper_cache_key(gridsearch_params):
    gridsearch_params_sorted_yaml = OmegaConf.to_yaml(
        gridsearch_params, sort_keys=True, resolve=True
    )

    # return None if cache decorator has been commented out for time being
    if not hasattr(run_gridsearch_experiment_cache_helper, "__cache_key__"):
        return None

    return run_gridsearch_experiment_cache_helper.__cache_key__(
        gridsearch_params=gridsearch_params,
        gridsearch_params_sorted_yaml=gridsearch_params_sorted_yaml,
    )


if __name__ == "__main__":
    register_resolvers()

    if (
        sys.gettrace() is None
    ):  # do not set low priority while debugging. Note that unit tests also set sys.gettrace() to not-None
        set_priorities()

    set_memory_limits()

    # Need to choose GPU early before torch fully starts up. Else there may be CUDA errors later.
    # In case of gridsearch this selected GPU has effect only if the calculation runs directly inside the main process which is in case of single GPU.
    select_gpu()

    gridsearch_params_json = os.environ.get(
        "GRIDSEARCH_PARAMS"
    )  # used by gridsearch subprocess
    if gridsearch_params_json is not None:
        run_gridsearch_experiment_subprocess(gridsearch_params_json)
    else:
        asyncio.run(
            run_gridsearch_experiments()
        )  # TODO: use separate python file for starting gridsearch
