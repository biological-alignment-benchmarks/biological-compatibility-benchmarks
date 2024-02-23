import os

import numpy as np
import numpy.testing as npt
import pytest

from aintelope.environments import savanna_safetygrid as safetygrid
from aintelope.environments.env_utils.distance import distance_to_closest_item
from aintelope.environments.savanna import ACTION_MAP
from aintelope.environments.savanna_safetygrid import SavannaGridworldSequentialEnv
from gymnasium.spaces import Discrete, MultiDiscrete
from pettingzoo.test import (
    api_test,
    max_cycles_test,
    performance_benchmark,
    render_test,
)
from pettingzoo.test.seed_test import seed_test

# from pettingzoo.utils import parallel_to_aec


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_api_sequential(execution_number):
    # TODO: refactor these values out to a test-params file
    # seed = int(time.time()) & 0xFFFFFFFF
    # np.random.seed(seed)
    # print(seed)
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 1,  # for now only one agent
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
        "use_old_aintelope_rewards": True,  # Zoo does not handle dictionary rewards well in sequential env test
    }
    env = safetygrid.SavannaGridworldSequentialEnv(env_params=env_params)
    env.seed(execution_number)

    # env = parallel_to_aec(parallel_env)
    api_test(env, num_cycles=10, verbose_progress=True)


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_api_sequential_with_death(execution_number):
    # TODO: refactor these values out to a test-params file
    # seed = int(time.time()) & 0xFFFFFFFF
    # np.random.seed(seed)
    # print(seed)

    # for Gridworlds, the seed needs to be specified during environment construction
    # since it affects map randomisation, while seed called later does not change map
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 2,  # needed for death test
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
        "test_death": False,
        "seed": execution_number,
        "use_old_aintelope_rewards": True,  # Zoo does not handle dictionary rewards well in sequential env test
    }
    env = safetygrid.SavannaGridworldSequentialEnv(env_params=env_params)

    # env = parallel_to_aec(parallel_env)
    api_test(env, num_cycles=10, verbose_progress=True)


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_seed(execution_number):
    # override_infos: Zoo seed_test is unable to compare infos unless they have simple structure.
    # seed: for Gridworlds, the seed needs to be specified during environment construction
    # since it affects map randomisation, while seed called later does not change map
    env_params = {
        "override_infos": True,
        "seed": execution_number,
    }

    def get_env_instance() -> safetygrid.SavannaGridworldSequentialEnv:
        """Method for seed_test"""
        return safetygrid.SavannaGridworldSequentialEnv(env_params=env_params)

    try:
        seed_test(get_env_instance, num_cycles=10)
    except TypeError:
        # for some reason the test env in Git does not recognise the num_cycles neither
        # as named or positional argument
        seed_test(get_env_instance)


def test_gridworlds_agent_states():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states


def test_gridworlds_reward_agent():
    pass  # safetygrid.SavannaGridworldEnv has no reward_agent()


def test_gridworlds_move_agent():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states and move_agent()


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_step_result(execution_number):
    # default is 1 iter which means that the env is done after 1 step below and the
    # test will fail
    env = safetygrid.SavannaGridworldSequentialEnv(
        env_params={
            "num_iters": 2,
            "seed": execution_number,
        }
    )
    num_agents = len(env.possible_agents)
    assert num_agents, f"expected 1 agent, got: {num_agents}"
    env.reset()

    agent = env.agent_selection
    action = env.action_space(agent).sample()

    env.step(action)
    # NB! env.last() provides observation from NEXT agent in case of multi-agent
    # environment
    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.last()  # TODO: multi-agent iteration
    done = terminated or truncated

    assert not done
    assert isinstance(
        observation[0], np.ndarray
    ), "observation[0] of agent is not an array"
    assert isinstance(
        observation[1], np.ndarray
    ), "observation[1] of agent is not an array"
    assert isinstance(reward, dict), "reward of agent is not a dict"


@pytest.mark.parametrize("execution_number", range(1))
def test_gridworlds_done_step(execution_number):
    env = safetygrid.SavannaGridworldSequentialEnv(
        env_params={
            "amount_agents": 1,
            "seed": execution_number,
        }
    )
    assert len(env.possible_agents) == 1
    env.reset()

    for _ in range(env.metadata["num_iters"]):
        agent = env.agent_selection
        action = env.action_space(agent).sample()
        env.step(action)
        # env.last() provides observation from NEXT agent in case of multi-agent
        # environment
        terminated = env.terminations[agent]
        truncated = env.truncations[agent]
        done = terminated or truncated

    assert done
    with pytest.raises(ValueError):
        action = env.action_space(agent).sample()
        env.step(action)


def test_gridworlds_agents():
    env = safetygrid.SavannaGridworldSequentialEnv()

    # assert len(env.possible_agents) == env.metadata["amount_agents"]  # TODO: this is now determined by the environment, not by config
    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_gridworlds_action_spaces():
    env = safetygrid.SavannaGridworldSequentialEnv()

    for agent in env.possible_agents:
        assert isinstance(env.action_space(agent), MultiDiscrete)
        assert env.action_space(agent).n == 5  # includes no-op


def test_gridworlds_action_space_valid_step():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states and move_agent()


def test_max_cycles():
    # currently the environment does not accept parameters like max_cycles
    # max_cycles_test(zoo.SavannaZooParallelEnv)
    pass


def test_render():
    # TODO: close method not implemented
    # render_test(zoo.SavannaZooParallelEnv)
    pass


def test_performance_benchmark():
    # will print only timing to stdout; not shown per default
    # performance_benchmark(zoo.SavannaZooParallelEnv())
    pass


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])  # run tests only in this file
    # test_gridworlds_api_sequential_with_death()
