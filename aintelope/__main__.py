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
    from semaphore_win_ctypes import AcquireSemaphore, CreateSemaphore, OpenSemaphore, Semaphore
else:
    import posix_ipc

# import mutex. This one is cross-platform
from filelock import FileLock

from progressbar import ProgressBar

from matplotlib import pyplot as plt

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import (
    archive_code,
    DummyContext,
    get_pipeline_score_dimensions,
    get_score_dimensions,
    register_resolvers,
    select_gpu,
    set_memory_limits,
    set_priorities,
)
from aintelope.experiments import run_experiment


logger = logging.getLogger("aintelope.__main__")

cache_folder = "gridsearch_cache"
cache = Cache(cache_folder)

gpu_count = max(1, torch.cuda.device_count())
worker_count_multiplier = 1   # when running pipeline search, then having more workers than GPU-s will cause all sorts of Python and CUDA errors under Windows for some reason, even though there is plenty of free RAM and GPU memory. Yet, when the pipeline processes are run manually, there is no concurrency limit. # TODO: why?
num_workers = gpu_count * worker_count_multiplier

gridsearch_params_global = None


def aintelope_main() -> None:  
    # return run_gridsearch_experiment(gridsearch_params=None)    # TODO: caching support
    run_pipeline() 


async def run_gridsearch_experiments() -> None:  
    
    use_multiprocessing = sys.gettrace() is None   # not debugging
    # use_multiprocessing = True   # TODO: currently CUDA fails in subprocesses

    gridsearch_config = os.environ.get("GRIDSEARCH_CONFIG")
    #if gridsearch_config is None:
    #    gridsearch_config = "aintelope/config/config_gridsearch.yaml"
    config_gridsearch = OmegaConf.load(gridsearch_config)

    # extract list parameters and compute cross product over their values
    dict_config = OmegaConf.to_container(config_gridsearch, resolve=False) # convert DictConfig to dict # NB! do not resolve references here since we do not want to handle references to lists as lists
    flattened_config = flatten(dict_config, reducer=make_reducer(delimiter='.'))    # convert to format {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}
    list_entries = {key: value for key, value in flattened_config.items() if isinstance(value, list)}   # select only entries of list type
    list_entries["hparams.gridsearch_trial_no"] = config_gridsearch.hparams.gridsearch_trial_no

    # create outer product of all list entries stored in the dictionary values
    # http://stephantul.github.io/python/2019/07/20/product-dict/
    keys, values = zip(*list_entries.items())   # this performs unzip - split dictionary in to list of keys and list of values
    values_combinations = list(itertools.product(*values))
    with ProgressBar(max_value=len(values_combinations)) as multiprocessing_bar:  # this is a slow task so lets use a progress bar

        active_coroutines = set()
        completed_coroutine_count = 0
        available_gpus = list(range(0, gpu_count)) * worker_count_multiplier    # repeat gpu index list for worker_count_multiplier times
        coroutine_gpus = {}

        for values_combination_i, values_combination in enumerate(values_combinations):   # iterate over value combinations
            gridsearch_combination = dict(zip(keys, values_combination))    # zip keys with values in current combination

            # print("gridsearch_combination:")
            gridsearch_combination_for_print = {key: value for key, value in gridsearch_combination.items() if len(flattened_config[key]) > 1}   # print only entries of lists that had more than one value in the gridsearch configuration, that is, ignore nested lists which are used for "list escaping" purposes
            # plotting.prettyprint(gridsearch_combination_for_print)

            gridsearch_params = copy.deepcopy(config_gridsearch)
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
                coroutine = asyncio.create_task(run_gridsearch_experiment_multiprocess(**arguments))   # NB! do not await here yet, awaiting will be done below by waiting for a group of coroutines at once.
                coroutine_gpus[coroutine] = use_gpu

                active_coroutines.add(coroutine)
                if len(active_coroutines) == num_workers:   
                    dones, pendings = await asyncio.wait(active_coroutines, return_when=asyncio.FIRST_COMPLETED)
                    for task in dones:
                        available_gpus.append(coroutine_gpus[task])
                        del coroutine_gpus[task]

                        ex = task.exception()  # https://rotational.io/blog/spooky-asyncio-errors-and-how-to-fix-them/
                        if ex is not None:
                            print(f"\nExperiment seems to have choked. Exception: {ex}")

                    completed_coroutine_count += 1
                    multiprocessing_bar.update(completed_coroutine_count)
                    active_coroutines = pendings

            else: #/ if use_multiprocessing:
                arguments = {
                    "gridsearch_params": gridsearch_params, 
                    "gridsearch_combination_for_print": gridsearch_combination_for_print,
                    "args": sys.argv, 
                    "do_not_create_subprocess": True,
                    "environ": dict(os.environ),
                    "use_gpu_index": None,
                }
                try:
                    result = await run_gridsearch_experiment_multiprocess(**arguments)
                except Exception as ex:
                    print(f"\nExperiment seems to have choked. Exception: {ex}. params: {plotting.prettyprint(gridsearch_combination_for_print)}")
                
                completed_coroutine_count += 1
                multiprocessing_bar.update(completed_coroutine_count)

            #/ if use_multiprocessing:

        #/ for values_combination_i, values_combination in enumerate(values_combinations):

        # wait for remaining coroutines
        while len(active_coroutines) > 0:
            dones, pendings = await asyncio.wait(active_coroutines, return_when=asyncio.FIRST_COMPLETED)
            for task in dones:
                ex = task.exception()  # https://rotational.io/blog/spooky-asyncio-errors-and-how-to-fix-them/
                if ex is not None:
                    print(f"\nExperiment seems to have choked. Exception: {ex}")
            completed_coroutine_count += 1
            multiprocessing_bar.update(completed_coroutine_count)
            active_coroutines = pendings

    #/ with ProgressBar(max_value=len(values_combinations)) as multiprocessing_bar:

    input("Gridsearch done. Press [enter] to continue.")
    return


subprocess_exec_lock = asyncio.Lock()

async def run_gridsearch_experiment_multiprocess(gridsearch_params: DictConfig, gridsearch_combination_for_print: dict, args: list = None, do_not_create_subprocess: bool = False, environ: dict = {}, use_gpu_index: int = None) -> None:
    """Use multiprocessing to conveniently queue the jobs and wait for their results in parallel.
    """

    # do not start subprocess if the result is already in cache
    cache_key = get_run_gridsearch_experiment_cache_helper_cache_key(gridsearch_params)

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
        gridsearch_params_json = json.dumps(OmegaConf.to_container(gridsearch_params, resolve=True))
        env = dict(environ)   # clone before modifying  # NB! need to pass whole environ, else the program may not start    # TODO: recheck with current implementation using asyncio.create_subprocess_exec
        # env.pop("CUDA_MODULE_LOADING", None)    # main process does not have this environment variable set, but for some reason os.environ contains CUDA_MODULE_LOADING=LAZY
        env["GRIDSEARCH_PARAMS"] = gridsearch_params_json
        if use_gpu_index is not None:
            env["GRIDSEARCH_GPU"] = str(use_gpu_index)
        env["PYTHONUNBUFFERED"] = "1"   # disables console buffering in the subprocess
        # TODO: use multiprocessing and keep the subprocesses alive for all time? But for that to work you need to somehow ensure that the multiprocessing subprocesses start with 30 sec intervals, else there will be crashes under Windows.
        async with subprocess_exec_lock:
            proc = await asyncio.create_subprocess_exec("python", *args, env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            if os.name == "nt":
                await asyncio.sleep(30)    # there needs to be delay between subprocess creation, else CUDA will crash for some reason. Additionally, even when CUDA does not crash, then python will crash when processes start in a "too short" sequence (like 10 sec intervals). Source: personal experience with multiple Windows machines and operating system versions.

        try:
            # TODO: tee subprocess output during its execution using https://github.com/thearchitector/tee-subprocess
            while proc.returncode is None:
                stdout, stderr = await proc.communicate()
                print("\n" + stdout.decode("utf-8", "ignore"))
            stdout, stderr = await proc.communicate()
            print("\n" + stdout.decode("utf-8", "ignore"))
        except Exception:
            print(f"\nExperiment worker process seems to have choked. Exception: {ex}. Params:") 
            plotting.prettyprint(gridsearch_combination_for_print)

        return


def run_gridsearch_experiment_subprocess(gridsearch_params_json: str) -> None:
    """Use subprocesses to run actual computations, since CUDA does not work in multiprocessing processes.
    """

    gridsearch_params_dict = json.loads(gridsearch_params_json)
    gridsearch_params = OmegaConf.create(gridsearch_params_dict)

    print(f"Running subprocess with params:")
    plotting.prettyprint(gridsearch_params_dict)

    return run_gridsearch_experiment(gridsearch_params=gridsearch_params)


def run_gridsearch_experiment(gridsearch_params: DictConfig) -> None:
    """Prepares call to run_gridsearch_experiment_cache_helper which does actual caching
    """
    
    gridsearch_params_sorted_yaml = OmegaConf.to_yaml(gridsearch_params, sort_keys=True, resolve=True)

    result = run_gridsearch_experiment_cache_helper(gridsearch_params=gridsearch_params, gridsearch_params_sorted_yaml=gridsearch_params_sorted_yaml)
    return result


# Actual cache is on run_game function, here we prepare the engine_conf and cache_version arguments.
def get_run_gridsearch_experiment_cache_helper_cache_key(gridsearch_params):

    gridsearch_params_sorted_yaml = OmegaConf.to_yaml(gridsearch_params, sort_keys=True, resolve=True)
    
    return run_gridsearch_experiment_cache_helper.__cache_key__(gridsearch_params=gridsearch_params, gridsearch_params_sorted_yaml=gridsearch_params_sorted_yaml)


@cache.memoize(ignore={'gridsearch_params'})   # use only gridsearch_params_sorted_yaml argument
def run_gridsearch_experiment_cache_helper(gridsearch_params: DictConfig, gridsearch_params_sorted_yaml: str) -> None:    # NB! do not rename this function, else cache will be invalidated
    global gridsearch_params_global  

    # cfg.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")    # TODO: re-parse date format from config file

    gridsearch_params_global = gridsearch_params    # TODO: hydra main does not allow multiple arguments, probably there is a more typical way to do it
    test_summaries = run_pipeline()
    return test_summaries # this result will be cached


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def run_pipeline(cfg: DictConfig) -> None:  #, gridsearch_params: DictConfig, do_not_show_plot: bool = False) -> None:

    gridsearch_params = gridsearch_params_global    # TODO: hydra main does not allow multiple arguments, probably there is a more typical way to do it
    do_not_show_plot = gridsearch_params is not None

    timestamp = str(cfg.timestamp)
    timestamp_pid_uuid = str(cfg.timestamp_pid_uuid)
    logger.info(f"timestamp: {timestamp}")
    logger.info(f"timestamp_pid_uuid: {timestamp_pid_uuid}")

    archive_code(cfg)

    pipeline_config = OmegaConf.load("aintelope/config/config_pipeline.yaml")
    # score_dimensions = get_pipeline_score_dimensions(cfg, pipeline_config)

    test_summaries = []

    # use additional semaphore here (in addition to gridsearch multiprocessing worker count) since the user may launch multiple gridsearch as well as non-gridsearch processes manually
    semaphore_name = (
                      "AIntelope_pipeline_semaphore" 
                      + ("_" + cfg.hparams.params_set_title 
                         if cfg.hparams.params_set_title 
                         in ["instinct", "random"] 
                         else "") 
                      + ("_debug" if sys.gettrace() is not None else "")
                    )
    print("Waiting for semaphore...")
    with (CreateSemaphore(semaphore_name, maximum_count=num_workers) if os.name == "nt" else DummyContext()) as semaphore:  
        with (AcquireSemaphore(semaphore) if os.name == "nt" else DummyContext()):
            with (posix_ipc.Semaphore(semaphore_name) if os.name != "nt" else DummyContext()):
                print("Semaphore acquired...")

                max_pipeline_cycle = cfg.hparams.num_pipeline_cycles + 1 if cfg.hparams.num_pipeline_cycles >= 1 else 1  # Last +1 cycle is for testing. In case of 0 pipeline cycle, run testing inside the same cycle immediately after each environment's training ends.
                with ProgressBar(max_value=max_pipeline_cycle) as pipeline_cycle_bar:  # this is a slow task so lets use a progress bar
                    for i_pipeline_cycle in range(0, max_pipeline_cycle):

                        is_last_pipeline_cycle = (i_pipeline_cycle == cfg.hparams.num_pipeline_cycles)

                        with ProgressBar(max_value=len(pipeline_config)) as pipeline_bar:  # this is a slow task so lets use a progress bar
                            for env_conf_i, env_conf_name in enumerate(pipeline_config):

                                experiment_cfg = copy.deepcopy(
                                    cfg
                                )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
                                OmegaConf.update(experiment_cfg, "experiment_name", env_conf_name)
                                if gridsearch_params is not None:
                                    OmegaConf.update(
                                        experiment_cfg, "hparams", gridsearch_params.hparams, force_add=True
                                    )   # NB! gridsearch params is applied before pipeline params
                                OmegaConf.update(
                                    experiment_cfg, "hparams", pipeline_config[env_conf_name], force_add=True
            )
            logger.info("Running training with the following configuration")
                                logger.info(os.linesep + str(OmegaConf.to_yaml(experiment_cfg, resolve=True)))

            # Training
            params_set_title = experiment_cfg.hparams.params_set_title
            logger.info(f"params_set: {params_set_title}, experiment: {env_conf_name}")

            score_dimensions = get_score_dimensions(experiment_cfg)
                                if cfg.hparams.num_pipeline_cycles == 0:  # in case of 0 pipeline cycle, run testing inside the same cycle immediately after each environment's training ends.
                                    run_experiment(experiment_cfg, experiment_name=env_conf_name, score_dimensions=score_dimensions, is_last_pipeline_cycle=False, i_pipeline_cycle=i_pipeline_cycle)
            run_experiment(experiment_cfg, experiment_name=env_conf_name, score_dimensions=score_dimensions, is_last_pipeline_cycle=is_last_pipeline_cycle, i_pipeline_cycle=i_pipeline_cycle)

            # torch.cuda.empty_cache()
            # gc.collect()

                                if is_last_pipeline_cycle:
                                    # Not using timestamp_pid_uuid here since it would make the title too long. In case of manual execution with plots, the pid-uuid is probably not needed anyway.
                                    title = timestamp + " : " + params_set_title + " : " + env_conf_name
                                    test_summary = analytics(experiment_cfg, score_dimensions, title=title, experiment_name=env_conf_name, group_by_pipeline_cycle=cfg.hparams.num_pipeline_cycles >= 1, gridsearch_params=gridsearch_params, do_not_show_plot=do_not_show_plot)
                                    test_summaries.append(test_summary)

                                pipeline_bar.update(env_conf_i + 1)

                            #/ for env_conf_name in pipeline_config:
                        #/ with ProgressBar(max_value=len(pipeline_config)) as pipeline_bar:

                        pipeline_cycle_bar.update(i_pipeline_cycle + 1)

                    #/ for i_pipeline_cycle in range(0, max_pipeline_cycle):
                #/ with ProgressBar(max_value=max_pipeline_cycle) as pipeline_cycle_bar:
            #/ with posix_ipc.Semaphore(name):
        #/ with AcquireSemaphore(created, timeout_ms=0):
    #/ with CreateSemaphore('name', maximum_count=num_workers) as semaphore:


    # Write the pipeline results to file only when entire pipeline has run. Else crashing the program during pipeline run will cause the aggregated results file to contain partial data which will be later duplicated by re-run.
    # TODO: alternatively, cache the results of each experiment separately
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)
        aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
        with aggregated_results_file_lock:   
            with open(aggregated_results_file, mode='a', encoding='utf-8') as fh:
                for test_summary in test_summaries:
                    # Do not write directly to file. If JSON serialization error occurs during json.dump() then a broken line would be written into the file (I have verified this). Therefore using json.dumps() is safer.
                    json_text = json.dumps(test_summary)
                    fh.write(json_text + "\n")  # \n : Prepare the file for appending new lines upon subsequent append. The last character in the JSONL file is allowed to be a line separator, and it will be treated the same as if there was no line separator present. 
                fh.flush()


    torch.cuda.empty_cache()
    gc.collect()

    # keep plots visible until the user decides to close the program
    if not do_not_show_plot:
        if os.name == "nt":
            import msvcrt
            print("\nPipeline done. Press [enter] to continue.")
            msvcrt.getch()    # uses less CPU on Windows than input() function. Note that the graph window will be frozen, but will still show graphs
            #while True:
            #    if msvcrt.kbhit():
            #        break
            #    plt.pause(60)
            #    # time.sleep(1)
        else:
            input("\nPipeline done. Press [enter] to continue.")

    return test_summaries


def analytics(cfg, score_dimensions, title, experiment_name, group_by_pipeline_cycle, gridsearch_params=DictConfig, do_not_show_plot=False):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    log_dir = os.path.normpath(cfg.log_dir)
    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname
    num_train_episodes = cfg.hparams.num_episodes
    num_train_pipeline_cycles = cfg.hparams.num_pipeline_cycles

    savepath = os.path.join(log_dir, "plot_" + experiment_name)
    events = recording.read_events(experiment_dir, events_fname)


    (test_totals, test_averages, test_variances, test_sfella_totals, test_sfella_averages, test_sfella_variances, sfella_score_total, sfella_score_average, sfella_score_variance, score_dimensions_out) = plotting.aggregate_test_scores(events, num_train_pipeline_cycles, score_dimensions, group_by_pipeline_cycle=group_by_pipeline_cycle)

    test_summary = {
        "timestamp": cfg.timestamp,
        "timestamp_pid_uuid": cfg.timestamp_pid_uuid,

        "experiment_name": experiment_name,
        "title": title,
        "params_set_title": cfg.hparams.params_set_title,
        "gridsearch_params": OmegaConf.to_container(gridsearch_params, resolve=True) if gridsearch_params is not None else None, # Object of type DictConfig is not JSON serializable, neither can yaml.dump in plotting.prettyprint digest it, so need to convert it to ordinary dictionary

        "score_dimensions": score_dimensions_out,

        "test_totals": test_totals, 
        "test_averages": test_averages, 
        "test_variances": test_variances,

        "test_sfella_totals": test_sfella_totals,
        "test_sfella_averages": test_sfella_averages,
        "test_sfella_variances": test_sfella_variances,

        "sfella_score_total": sfella_score_total, 
        "sfella_score_average": sfella_score_average, 
        "sfella_score_variance": sfella_score_variance,
    }

    plotting.prettyprint(test_summary)

    plotting.plot_performance(events, num_train_episodes, num_train_pipeline_cycles, score_dimensions, save_path=savepath, title=title, group_by_pipeline_cycle=group_by_pipeline_cycle, do_not_show_plot=do_not_show_plot)

    return test_summary


if __name__ == "__main__":

    register_resolvers()

    if sys.gettrace() is None:    # do not set low priority while debugging. Note that unit tests also set sys.gettrace() to not-None
        set_priorities()

    set_memory_limits()

    # Need to choose GPU early before torch fully starts up. Else there may be CUDA errors later.
    gridsearch_gpu = os.environ.get("GRIDSEARCH_GPU")
    if gridsearch_gpu is not None:
        gridsearch_gpu = int(gridsearch_gpu)
        torch.cuda.set_device(gridsearch_gpu)
        device_name = torch.cuda.get_device_name(gridsearch_gpu)
        print(f"Using CUDA GPU {gridsearch_gpu} : {device_name}")
    else:
        # for each next experiment select next available GPU to maximally balance the load considering multiple running processes
        select_gpu()

    gridsearch_params_json = os.environ.get("GRIDSEARCH_PARAMS")
    gridsearch_config = os.environ.get("GRIDSEARCH_CONFIG")
    if gridsearch_params_json is not None:        
        run_gridsearch_experiment_subprocess(gridsearch_params_json)
    elif gridsearch_config is not None:
        asyncio.run(run_gridsearch_experiments())     # TODO: use separate python file for starting gridsearch
    else:
    aintelope_main()
