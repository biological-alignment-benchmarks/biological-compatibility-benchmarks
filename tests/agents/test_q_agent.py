# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import os
from typing import Dict

import pytest

from omegaconf import OmegaConf

from aintelope.training.simple_eval import run_episode

@pytest.mark.parametrize("execution_number", range(1))
def test_qagent_in_savanna_gridworlds_sequential(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    hparams = {
        "agent_class": "q_agent",
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
    full_params.hparams = OmegaConf.merge(full_params.hparams, hparams)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(1))
def test_qagent_in_savanna_gridworlds_sequential_with_death(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    hparams = {
        "agent_class": "q_agent",
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
    full_params.hparams = OmegaConf.merge(full_params.hparams, hparams)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(1))
def test_qagent_in_savanna_gridworlds_parallel(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    hparams = {
        "agent_class": "q_agent",
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
    full_params.hparams = OmegaConf.merge(full_params.hparams, hparams)
    run_episode(full_params=full_params)


@pytest.mark.parametrize("execution_number", range(1))
def test_qagent_in_savanna_gridworlds_parallel_with_death(
    tparams_hparams: Dict, execution_number
) -> None:
    full_params = tparams_hparams
    hparams = {
        "agent_class": "q_agent",
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
    full_params.hparams = OmegaConf.merge(full_params.hparams, hparams)
    run_episode(full_params=full_params)


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])  # run tests only in this file
    # pytest.main([__file__ + "::test_qagent_in_savanna_zoo_sequential_with_death"])
