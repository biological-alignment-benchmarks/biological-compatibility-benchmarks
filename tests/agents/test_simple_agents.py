from typing import Tuple

from omegaconf import OmegaConf, DictConfig

from tests.test_config import root_dir, tparams_hparams
from aintelope.environments.env_utils.cleanup import cleanup_gym_envs
from aintelope.training.simple_eval import run_episode


def test_randomwalkagent_in_savanna_gym(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_randomwalkagent = {
        "agent": "random_walk_agent",
        "env": "savanna-gym-v2",
        "env_type": "gym",
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
    cleanup_gym_envs()


def test_onestepperfectpredictionagent_in_savanna_gym(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_perfectpredictionagent = {
        "agent": "one_step_perfect_prediction_agent",
        "env": "savanna-gym-v2",
        "env_type": "gym",
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
    cleanup_gym_envs()


def test_iterativeweightoptimizationagent_in_savanna_gym(
    tparams_hparams: Tuple[DictConfig, DictConfig]
) -> None:
    tparams, hparams = tparams_hparams
    params_weightoptimizationagent = {
        "agent": "iterative_weight_optimization_agent",
        "env": "savanna-gym-v2",
        "env_type": "gym",
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
    cleanup_gym_envs()
