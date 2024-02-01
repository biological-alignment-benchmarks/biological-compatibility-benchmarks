import os
from typing import Dict

import pytest
import numpy as np
from omegaconf import OmegaConf

from aintelope.training.simple_eval import run_episode


@pytest.mark.parametrize("execution_number", range(10))
def test_qagent_in_savanna_zoo_sequential(
    tparams_hparams: Dict, execution_number
) -> None:
    np.random.seed(execution_number)

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
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(10))
def test_qagent_in_savanna_zoo_sequential_with_death(
    tparams_hparams: Dict, execution_number
) -> None:
    np.random.seed(execution_number)

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
            "amount_agents": 2,  # needed for death test
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "test_death": True,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(10))
def test_qagent_in_savanna_gridworlds_sequential(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    params_savanna_zoo = {
        "agent_id": "q_agent",
        "env": "savanna-safetygrid-sequential-v1",
        "env_entry_point": (
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "seed": execution_number,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(10))
def test_qagent_in_savanna_gridworlds_sequential_with_death(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    params_savanna_zoo = {
        "agent_id": "q_agent",
        "env": "savanna-safetygrid-sequential-v1",
        "env_entry_point": (
            "aintelope.environments.savanna_safetygrid:SavannaGridworldSequentialEnv"
        ),
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 2,  # needed for death test
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "test_death": True,
            "seed": execution_number,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(10))
def test_qagent_in_savanna_zoo_parallel(
    tparams_hparams: Dict, execution_number
) -> None:
    np.random.seed(execution_number)

    full_params = tparams_hparams
    params_savanna_zoo = {
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
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(10))
def test_qagent_in_savanna_zoo_parallel_with_death(
    tparams_hparams: Dict, execution_number
) -> None:
    np.random.seed(execution_number)

    full_params = tparams_hparams
    params_savanna_zoo = {
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
            "amount_agents": 2,  # needed for death test
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "test_death": True,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(10))
def test_qagent_in_savanna_gridworlds_parallel(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    params_savanna_zoo = {
        "agent_id": "q_agent",
        "env": "savanna-safetygrid-parallel-v1",
        "env_entry_point": (
            "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv"
        ),
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "seed": execution_number,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(10))
def test_qagent_in_savanna_gridworlds_parallel_with_death(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    params_savanna_zoo = {
        "agent_id": "q_agent",
        "env": "savanna-safetygrid-parallel-v1",
        "env_entry_point": (
            "aintelope.environments.savanna_safetygrid:SavannaGridworldParallelEnv"
        ),
        "env_type": "zoo",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 2,  # needed for death test
            "amount_grass_patches": 2,
            "amount_water_holes": 0,
            "test_death": True,
            "seed": execution_number,
        },
        "agent_params": {},
    }
    full_params.hparams = OmegaConf.merge(full_params.hparams, params_savanna_zoo)
    run_episode(full_params=full_params)


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])  # run tests only in this file
    # pytest.main([__file__ + "::test_qagent_in_savanna_zoo_sequential_with_death"])
