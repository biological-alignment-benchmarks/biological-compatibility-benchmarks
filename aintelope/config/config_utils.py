import copy
from ast import literal_eval
from pathlib import Path
import os

from omegaconf import DictConfig, OmegaConf


def get_project_path(path_from_root: str) -> Path:
    project_root = Path(__file__).parents[2]
    return project_root / path_from_root


def register_resolvers() -> None:
    OmegaConf.register_resolver("abs_path", get_project_path)


def get_score_dimensions(cfg: DictConfig):
    scores = cfg.hparams.env_params.scores
    dimensions = set()
    for event_name, score_dims_dict in scores.items():
        score_dims_dict = literal_eval(score_dims_dict)
        for dimension, value in score_dims_dict.items():
            if value != 0:  # ignore zero valued score dimensions
                dimensions.add(dimension)
    dimensions = list(dimensions)
    dimensions.sort()
    return dimensions


def get_pipeline_score_dimensions(cfg: DictConfig, pipeline_config: DictConfig):
    dimensions = set()
    for env_conf in pipeline_config:
        experiment_cfg = copy.deepcopy(
            cfg
        )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
        OmegaConf.update(  # need to merge configs here too since dimensions inside scores are not merged, but instead overwritten by experiment config. If main config has some score dimension that experiment does not have, then then that score dimension should not be used
            experiment_cfg, "hparams", pipeline_config[env_conf], force_add=True
        )
        experiment_score_dimensions = get_score_dimensions(experiment_cfg)
        for dimension in experiment_score_dimensions:
            dimensions.add(dimension)

    dimensions = list(dimensions)
    dimensions.sort()
    return dimensions


def set_priorities():
    """Sets CPU priorities in order to avoid slowing down the system"""

    try:
        import psutil

        if hasattr(psutil, "Process"):
            pid = os.getpid()

            p = psutil.Process(pid)
    
            # set to lowest  priority, this is Windows only, on Unix use ps.nice(19)
            # On UNIX this is a number which usually goes from -20 to 20. The higher the nice value, the lower the priority of the process.
            # https://psutil.readthedocs.io/en/latest/#psutil.Process.nice
            p.nice(psutil.IDLE_PRIORITY_CLASS if os.name == "nt" else 20)  # TODO: config

            # On Windows only *ioclass* is used and it can be set to 2
            # (normal), 1 (low) or 0 (very low).
            p.ionice(0 if os.name == "nt" else psutil.IOPRIO_CLASS_IDLE)
    
            # print("Priorities set...")
    
    except Exception as msg:
        print("run pip install psutil")


    if os.name == "nt":
        try:  # psutil fails to set IO priority under Windows for some reason
            import win32process   # TODO: use ctypes.windll.kernel32 instead?
  
            win32process.SetThreadPriority(-2, -15)  # NB! -2: win32api.GetCurrentThread()  # -15: Idle priority, is lower than THREAD_MODE_BACKGROUND_BEGIN
            win32process.SetThreadPriorityBoost(-2, False)
   
            # NB! do not call win32process.SetPriorityClass(-1, 0x00100000) (PROCESS_MODE_BACKGROUND_BEGIN) since that would significantly reduce GPU load. Also, it would also cause constant page faults and swap disk writes.

            win32process.SetProcessPriorityBoost(-1, False)
 
        except Exception as msg:
            print("run pip install pywin32")
    

def set_memory_limits():
    """Sets memory usage limits in order to avoid crashing the system"""

    # TODO: read limits from config
    if os.name == 'nt':
        mem_limit = 20 * 1024 * 1024 * 1024
        min_free_swap = 5 * 1024 * 1024 * 1024

        from aintelope.config.windows_jobobject import set_mem_commit_limit
        try:
            set_mem_commit_limit(os.getpid(), mem_limit, min_free_swap)
        except Exception as msg:
            print("run pip install psutil")
    else:   #/ if os.name == 'nt':
        data_size_limit = 20 * 1024 * 1024 * 1024
        address_space_size_limit = 400 * 1024 * 1024 * 1024

        from aintelope.config.linux_rlimit import set_mem_limits
        set_mem_limits(data_size_limit, address_space_size_limit)
