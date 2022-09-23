import pytest
import numpy as np
import numpy.testing as npt
from gym.spaces import Discrete
from pettingzoo.test import (
    max_cycles_test,
    render_test,
    performance_benchmark,
)
from pettingzoo.test.parallel_test import parallel_api_test
from pettingzoo.test import api_test
from pettingzoo.test.seed_test import parallel_seed_test
from pettingzoo.utils import parallel_to_aec

from aintelope.aintelope.environments import savanna_zoo as sut
from aintelope.aintelope.environments.savanna_zoo import SavannaZooEnv


def test_pettingzoo_api_parallel():
    parallel_api_test(sut.env(), num_cycles=1000)
    
    
    
def test_pettingzoo_api_sequential():
    # TODO: refactor these values out to a test-params file
    env_params = {
        'num_iters': 500,  # duration of the game
        'map_min': 0,
        'map_max': 100,
        'render_map_max': 100,
        'amount_agents': 1,  # for now only one agent
        'amount_grass_patches': 2,
        'amount_water_holes': 2,
    }
    parallel_env = SavannaZooEnv(env_params=env_params)
    # TODO: Nathan was able to get the sequential-turn env to work, using this conversion, but not the parallel env. why??
    sequential_env = parallel_to_aec(parallel_env)
    api_test(sequential_env, num_cycles=10, verbose_progress=True)


def test_seed():
    parallel_seed_test(sut.env, num_cycles=10, test_kept_state=True)


def test_max_cycles():
    # currently the environment does not accept parameters like max_cycles
    # max_cycles_test(sut.env)
    pass


def test_render():
    # TODO: close method not implemented
    # render_test(sut.env)
    pass


def test_performance_benchmark():
    # will print only timing to stdout; not shown per default
    # performance_benchmark(sut.env())
    pass
