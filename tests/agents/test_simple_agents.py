import sys
from typing import Tuple

from omegaconf import OmegaConf, DictConfig

from tests.test_config import root_dir, tparams_hparams, root_dir_debug, tparams_hparams_debug
from aintelope.training.simple_eval import run_episode


# TODO: implementation and tests for sequential zoo envs


def test_randomwalkagent_in_savanna_zoo_parallel(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_randomwalkagent = {
        "agent": "random_walk_agent",
        "env": "savanna-zoo-parallel-v2",
        "env_entry_point": "aintelope.environments.savanna_zoo:SavannaZooParallelEnv",
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
    OmegaConf.merge(hparams, params_randomwalkagent)
    run_episode(tparams=tparams, hparams=hparams)


def test_onestepperfectpredictionagent_in_savanna_zoo_parallel(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_perfectpredictionagent = {
        "agent": "one_step_perfect_prediction_agent",
        "env": "savanna-zoo-parallel-v2",
        "env_entry_point": "aintelope.environments.savanna_zoo:SavannaZooParallelEnv",
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
    OmegaConf.merge(hparams, params_perfectpredictionagent)
    run_episode(tparams=tparams, hparams=hparams)


def test_iterativeweightoptimizationagent_in_savanna_zoo_parallel(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_weightoptimizationagent = {
        "agent": "iterative_weight_optimization_agent",
        "env": "savanna-zoo-parallel-v2",
        "env_entry_point": "aintelope.environments.savanna_zoo:SavannaZooParallelEnv",
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
    OmegaConf.merge(hparams, params_weightoptimizationagent)
    run_episode(tparams=tparams, hparams=hparams)


def test_randomwalkagent_in_savanna_gridworlds_parallel(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_randomwalkagent = {
        "agent": "random_walk_agent",
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
    OmegaConf.merge(hparams, params_randomwalkagent)
    run_episode(tparams=tparams, hparams=hparams)


def test_onestepperfectpredictionagent_in_savanna_gridworlds_parallel(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_perfectpredictionagent = {
        "agent": "one_step_perfect_prediction_agent",
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
    OmegaConf.merge(hparams, params_perfectpredictionagent)
    run_episode(tparams=tparams, hparams=hparams)


def test_iterativeweightoptimizationagent_in_savanna_gridworlds_parallel(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_weightoptimizationagent = {
        "agent": "iterative_weight_optimization_agent",
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
    OmegaConf.merge(hparams, params_weightoptimizationagent)
    run_episode(tparams=tparams, hparams=hparams)


if __name__ == "__main__" and sys.gettrace() is not None:  # detect debugging
    tparams_hparams = tparams_hparams_debug(root_dir_debug())
    test_randomwalkagent_in_savanna_zoo_parallel(tparams_hparams)
    test_onestepperfectpredictionagent_in_savanna_zoo_parallel(tparams_hparams)
    test_iterativeweightoptimizationagent_in_savanna_zoo_parallel(tparams_hparams)
    test_randomwalkagent_in_savanna_gridworlds_parallel(tparams_hparams)
    test_onestepperfectpredictionagent_in_savanna_gridworlds_parallel(tparams_hparams)
    test_iterativeweightoptimizationagent_in_savanna_gridworlds_parallel(
        tparams_hparams
    )
