import copy
from ast import literal_eval
from pathlib import Path
import zipfile
import os
import uuid
import torch

from omegaconf import DictConfig, OmegaConf


def get_project_path(path_from_root: str) -> Path:
    project_root = Path(__file__).parents[2]
    return project_root / path_from_root


def append_pid_and_uuid(timestamp: str) -> str:
    pid = os.getpid()
    unique_id = uuid.uuid1()  # Generate a UUID from a host ID, sequence number, and the current time. If node is not given, getnode() is used to obtain the hardware address. If clock_seq is given, it is used as the sequence number; otherwise a random 14-bit sequence number is chosen.  # TODO: uuid1 is good for multi-machine cloud computing. Other uuid methods might be preferrable in case of privacy considerations. See https://docs.python.org/3/library/uuid.html
    result = f"{timestamp}_{pid}_{unique_id}"
    return result


def create_range(start, exclusive_end):
    return list(range(start, exclusive_end))


def register_resolvers() -> None:
    OmegaConf.register_new_resolver("abs_path", get_project_path)
    OmegaConf.register_new_resolver("append_pid_and_uuid", append_pid_and_uuid, use_cache=True)   # NB! need to enable caching else the pid_and_uuid will change at random moments during execution, leading to errors
    OmegaConf.register_new_resolver("minus_3", lambda x: x - 3)
    OmegaConf.register_new_resolver("range", create_range)


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


def select_gpu():
    """Rotates over available GPU-s, selecting a next GPU for each subsequent 
    program launch in order to distribute the workload of concurrently running 
    programs over all available GPU-s.

    If you want to restrict the set of GPU-s the program selects from, then 
    specify the CUDA_VISIBLE_DEVICES environment variable before starting the 
    program like this:

    In Linux:
    export CUDA_VISIBLE_DEVICES=3,4

    In Windows:
    set CUDA_VISIBLE_DEVICES=3,4

    In VSCode, modify launch.json:
    "env": {
        "CUDA_VISIBLE_DEVICES": "3,4"
    }

    where 3,4 are the device numbers you want to enable for the program to choose from.
    The device numbers start from 0.
    """

    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        print(f"No CUDA GPU available")
        return

    elif gpu_count == 1:
        # Even if only one device is enabled for CUDA, it might be not the first one by number
        # if it is set by CUDA_VISIBLE_DEVICES environment variable. Therefore, lets log it.
        gpu_counter = torch.cuda.current_device()
        print(f"Using the only available CUDA GPU")

    else:
        import sqlite3
        conn = sqlite3.connect("gpu_counter.db")

        cmd = """
        CREATE TABLE IF NOT EXISTS gpu_counter_table 
        (gpu_counter INTEGER);
        INSERT OR IGNORE INTO gpu_counter_table (gpu_counter) VALUES (0);
        """;
        cursor = conn.cursor()
        cursor.executescript(cmd)
        conn.commit()

        cursor = conn.cursor()
        cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
        cursor.execute("UPDATE gpu_counter_table SET gpu_counter = gpu_counter + 1")
        cursor = cursor.execute("SELECT gpu_counter FROM gpu_counter_table")
        gpu_counter = cursor.fetchone()[0]
        conn.commit()

        conn.close()

        gpu_counter = gpu_counter % gpu_count

        torch.cuda.set_device(gpu_counter)

    device_name = torch.cuda.get_device_name(gpu_counter)
    print(f"Using CUDA GPU {gpu_counter} : {device_name}")
    return


def archive_code(cfg):
    """Archives the current version of the program to log folder"""

    code_directory_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")   # archive only files under aintelope folder, no need to archive the tests folder
    zip_path = os.path.join(os.path.normpath(cfg.log_dir), "aintelope_code_archive.zip")
    archive_code_in_dir(code_directory_path, zip_path)
  
    code_directory_path = os.path.join(code_directory_path, "..", "ai_safety_gridworlds")
    zip_path = os.path.join(os.path.normpath(cfg.log_dir), "gridworlds_code_archive.zip")
    archive_code_in_dir(code_directory_path, zip_path)
    

def archive_code_in_dir(directory_path, zip_path):

    with zipfile.ZipFile(zip_path, 'w') as ziph:
        for root, dirs, files in os.walk(directory_path, topdown=True, followlinks=False):

            # When topdown is True, the caller can modify the dirnames list in-place (perhaps using del or slice assignment), and walk() will only recurse into the subdirectories whose names remain in dirnames; this can be used to prune the search
            # https://docs.python.org/3/library/os.html#os.walk
            dirs_to_skip = []
            for dir in dirs:
                if dir[:1] == ".":   # ignore dirs that start with dot (.vshistory, etc)
                    dirs_to_skip.append(dir)    # cannot remove dir directly from the list that is being iterated, else following dirs may be skipped from check
            for dir in dirs_to_skip:
                dirs.remove(dir)

            for file in files:
                extension = os.path.splitext(file)[1]
                if extension in [".py", ".ipynb", ".yaml"]:
                    ziph.write(
                        os.path.join(root, file), 
                        os.path.relpath(
                            os.path.join(root, file), 
                            os.path.join(directory_path, '..')
                        )
                    )


# used for disabling context objects like multiprocessing pool or progressbar
class DummyContext(object):

  def __init__(self, *args, **kwargs):
    pass

  # context manager functionality requires this method to be explicitly implemented
  def __enter__(self):
    return self

  # context manager functionality requires this method to be explicitly implemented
  def __exit__(self, type, value, traceback):
    return

  def _blackHoleMethod(*args, **kwargs):
    return

  def __getattr__(self, attr):
    return self._blackHoleMethod

#/ class DummyContext(object):

