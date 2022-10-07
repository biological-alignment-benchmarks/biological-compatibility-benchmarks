import pytest
import yaml
from yaml.loader import SafeLoader

from aintelope.environments.env_utils.cleanup import cleanup_gym_envs
from aintelope.training.simple_eval import run_episode


def test_randomwalkagent_in_savanna_gym():
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
            "amount_water_holes": 0
        },
        "agent_params": {}
    }
    hparams.update(test_params)
    run_episode(hparams=hparams)
    cleanup_gym_envs()
    
    
def test_onestepperfectpredictionagent_in_savanna_gym():
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
            "amount_water_holes": 0
        },
        "agent_params": {}
    }
    hparams.update(test_params)
    run_episode(hparams=hparams)
    cleanup_gym_envs()
    

def test_iterativeweightoptimizationagent_in_savanna_gym():
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
            "amount_water_holes": 0
        },
        "agent_params": {}
    }
    hparams.update(test_params)
    run_episode(hparams=hparams)
    cleanup_gym_envs()