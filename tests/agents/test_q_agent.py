import sys
import os
import pytest
from typing import Tuple

from omegaconf import OmegaConf, DictConfig

from tests.test_config import (
    root_dir,
    tparams_hparams,
)
from aintelope.training.simple_eval import run_episode


def test_qagent_in_savanna_zoo_sequential(  # TODO
    tparams_hparams: Tuple[DictConfig, DictConfig, OmegaConf]
) -> None:
    full_params = tparams_hparams
    params_savanna_zoo = {
        "agent_id": "q_agent",
        "env": "savanna-zoo-sequential-v2",
        "env_entry_point": "aintelope.environments.savanna_zoo:SavannaZooSequentialEnv",
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
    OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


def test_qagent_in_savanna_gridworlds_sequential(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    full_params = tparams_hparams
    params_savanna_zoo = {
        "agent_id": "q_agent",
        "env": "savanna-safetygrid-sequential-v1",
        "env_entry_point": "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv",
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
    OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


# TODO: parallel API support

# def test_qagent_in_savanna_zoo_parallel(  # TODO
#    tparams_hparams: Tuple[DictConfig, DictConfig]
# ) -> None:
#    full_params = tparams_hparams
#    params_savanna_zoo = {
#        "agent_id": "q_agent",
#        "env": "savanna-zoo-parallel-v2",
#        "env_entry_point": "aintelope.environments.savanna_zoo:SavannaZooParallelEnv",
#        "env_type": "zoo",
#        "sequential_env": True,
#        "env_params": {
#            "num_iters": 40,  # duration of the game
#            "map_min": 0,
#            "map_max": 20,
#            "render_map_max": 20,
#            "amount_agents": 1,  # for now only one agent
#            "amount_grass_patches": 2,
#            "amount_water_holes": 0,
#        },
#        "agent_params": {},
#    }
#    OmegaConf.merge(full_params.hparams, params_savanna_zoo)
#    run_episode(full_params=full_params)


# def test_qagent_in_savanna_gridworlds_parallel(
#    tparams_hparams: Tuple[DictConfig, DictConfig]
# ) -> None:
#    full_params = tparams_hparams
#    params_savanna_zoo = {
#        "agent_id": "q_agent",
#        "env": "savanna-safetygrid-parallel-v1",
#        "env_entry_point": "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv",
#        "env_type": "zoo",
#        "env_params": {
#            "num_iters": 40,  # duration of the game
#            "map_min": 0,
#            "map_max": 20,
#            "render_map_max": 20,
#            "amount_agents": 1,  # for now only one agent
#            "amount_grass_patches": 2,
#            "amount_water_holes": 0,
#        },
#        "agent_params": {},
#    }
#    OmegaConf.merge(full_params.hparams, params_savanna_zoo)
#    run_episode(full_params=full_params)


if (
    __name__ == "__main__" and os.name == "nt" and sys.gettrace() is not None
):  # detect debugging
    pytest.main([__file__])  # run tests only in this file
