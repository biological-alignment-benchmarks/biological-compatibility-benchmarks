import pytest
import yaml
from yaml.loader import SafeLoader


from aintelope.training.simple_eval import run_episode


def test_qagent_in_savanna_zoo_sequential():
    # get the default params from training.lightning.yaml
    # then override with these test params

    # Open the file and load the file
    import os
    print(os.getcwd())
    with open('aintelope/training/lightning.yaml') as f:
        full_params = yaml.load(f, Loader=SafeLoader)
        hparams = full_params['hparams']
        print(hparams)
    # TODO: refactor out into test constants? Or leave here? /shrug
    test_params = {
        "agent": "q_agent",
        "env": "savanna-zoo-v2",
        "env_entry_point": None,
        "env_type": "zoo",
        "sequential_env": True,
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0
        },
        "agent_params": {}
    }
    hparams.update(test_params)
    run_episode(hparams=hparams)
    
    
    
def test_qagent_in_savanna_zoo_parallel():
    # get the default params from training.lightning.yaml
    # then override with these test params

    # Open the file and load the file
    import os
    print(os.getcwd())
    with open('aintelope/training/lightning.yaml') as f:
        full_params = yaml.load(f, Loader=SafeLoader)
        hparams = full_params['hparams']
        print(hparams)
    # TODO: refactor out into test constants? Or leave here? /shrug
    test_params = {
        "agent": "q_agent",
        "env": "savanna-zoo-v2",
        "env_entry_point": None,
        "env_type": "zoo",
        "sequential_env": False,
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0
        },
        "agent_params": {}
    }
    hparams.update(test_params)
    run_episode(hparams=hparams)
    
    
def test_qagent_in_savanna_gym():
    # get the default params from training.lightning.yaml
    # then override with these test params

    # Open the file and load the file
    import os
    print(os.getcwd())
    with open('aintelope/training/lightning.yaml') as f:
        full_params = yaml.load(f, Loader=SafeLoader)
        hparams = full_params['hparams']
        print(hparams)
    # TODO: refactor out into test constants? Or leave here? /shrug
    test_params = {
        "agent": "q_agent",
        "env": "savanna-gym-v2",
        "env_type": "gym",
        "env_params": {
            "num_iters": 40,  # duration of the game
            "map_min": 0,
            "map_max": 20,
            "render_map_max": 20,
            "amount_agents": 1,  # for now only one agent
            "amount_grass_patches": 2,
            "amount_water_holes": 0
        },
        "agent_params": {}
    }
    hparams.update(test_params)
    run_episode(hparams=hparams)
