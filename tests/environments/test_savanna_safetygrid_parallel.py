import os

import numpy as np
import pytest
from gymnasium.spaces import MultiDiscrete
from pettingzoo.test.parallel_test import parallel_api_test
from pettingzoo.test.seed_test import parallel_seed_test

from aintelope.environments import savanna_safetygrid as safetygrid


@pytest.mark.parametrize("execution_number", range(10))
def test_gridworlds_api_parallel(execution_number):
    # TODO: refactor these values out to a test-params file
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 1,  # for now only one agent
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
    }
    env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)
    env.seed(execution_number)

    # sequential_env = parallel_to_aec(env)
    parallel_api_test(env, num_cycles=10)


@pytest.mark.parametrize("execution_number", range(10))
def test_gridworlds_api_parallel_with_death(execution_number):
    # TODO: refactor these values out to a test-params file
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
        "test_death": True,
        "seed": execution_number,
    }
    env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)

    # sequential_env = parallel_to_aec(env)
    parallel_api_test(env, num_cycles=10)


@pytest.mark.parametrize("execution_number", range(10))
def test_gridworlds_seed(execution_number):
    # Zoo parallel_seed_test is unable to compare infos unless they have simple
    # structure.
    # for Gridworlds, the seed needs to be specified during environment construction
    # since it affects map randomisation, while seed called later does not change map
    env_params = {
        "override_infos": True,
        "seed": execution_number,
    }

    def env_instance() -> safetygrid.SavannaGridworldParallelEnv:
        """Method for seed_test"""
        return safetygrid.SavannaGridworldParallelEnv(env_params=env_params)

    try:
        parallel_seed_test(env_instance, num_cycles=10)
    except TypeError:
        # for some reason the test env in Git does not recognise the num_cycles
        # neither as named or positional argument
        parallel_seed_test(env_instance)


def test_gridworlds_agent_states():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states


def test_gridworlds_reward_agent():
    pass  # safetygrid.SavannaGridworldEnv has no reward_agent()


def test_gridworlds_move_agent():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states and move_agent()


@pytest.mark.parametrize("execution_number", range(10))
def test_gridworlds_step_result(execution_number):
    # default is 1 iter which means that the env is done after 1 step below and the
    # test will fail
    env = safetygrid.SavannaGridworldParallelEnv(
        env_params={
            "num_iters": 2,
            "seed": execution_number,
        }
    )
    num_agents = len(env.possible_agents)
    assert num_agents, f"expected 1 agent, got: {num_agents}"
    env.reset()

    agent = env.possible_agents[0]
    action = {agent: env.action_space(agent).sample()}

    observations, rewards, terminateds, truncateds, infos = env.step(action)
    dones = {
        key: terminated or truncateds[key] for (key, terminated) in terminateds.items()
    }

    assert not dones[agent]
    assert isinstance(observations, dict), "observations is not a dict"
    assert isinstance(
        observations[agent], np.ndarray
    ), "observations of agent is not an array"
    assert isinstance(rewards, dict), "rewards is not a dict"
    assert isinstance(rewards[agent], np.float64), "reward of agent is not a float64"


@pytest.mark.parametrize("execution_number", range(10))
def test_gridworlds_done_step(execution_number):
    env = safetygrid.SavannaGridworldParallelEnv(
        env_params={
            "amount_agents": 1,
            "seed": execution_number,
        }
    )
    assert len(env.possible_agents) == 1
    env.reset()

    agent = env.possible_agents[0]  # TODO: multi-agent iteration
    for _ in range(env.metadata["num_iters"]):
        action = {agent: env.action_space(agent).sample()}
        _, _, terminateds, truncateds, _ = env.step(action)
        dones = {
            key: terminated or truncateds[key]
            for (key, terminated) in terminateds.items()
        }

    assert dones[agent]
    with pytest.raises(ValueError):
        action = {agent: env.action_space(agent).sample()}
        env.step(action)


def test_gridworlds_agents():
    env = safetygrid.SavannaGridworldParallelEnv()

    assert len(env.possible_agents) == env.metadata["amount_agents"]
    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_gridworlds_action_spaces():
    env = safetygrid.SavannaGridworldParallelEnv()

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
    # test_gridworlds_api_parallel_with_death()
