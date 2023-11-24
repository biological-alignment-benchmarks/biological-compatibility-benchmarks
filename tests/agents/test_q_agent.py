import sys
import pytest
from typing import Tuple

from omegaconf import OmegaConf, DictConfig

from tests.test_config import (
    root_dir,
    tparams_hparams,
)
from aintelope.training.simple_eval import run_episode


# TODO: implementation and tests for sequential zoo envs


def test_qagent_in_savanna_zoo_parallel(  # TODO
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_zoo_sequential = {
        "agent_id": "q_agent",
        "env": "savanna-zoo-parallel-v2",
        "env_entry_point": "aintelope.environments.savanna_zoo:SavannaZooParallelEnv",
        "env_type": "zoo",
        "sequential_env": True,
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
        },
        "agent_params": {},
    }
    OmegaConf.merge(hparams, params_zoo_sequential)
    run_episode(tparams=tparams, hparams=hparams)


def test_qagent_in_savanna_gridworlds_parallel(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_zoo_parallel = {
        "agent_id": "q_agent",
        "env": "savanna-safetygrid-parallel-v1",
        "env_entry_point": "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv",
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
        },
        "agent_params": {},
    }
    OmegaConf.merge(hparams, params_zoo_parallel)
    run_episode(tparams=tparams, hparams=hparams)


if __name__ == "__main__" and sys.gettrace() is not None:  # detect debugging
    pytest.main([__file__])  # run tests only in this file
