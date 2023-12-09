import os
import sys
import pytest
import numpy as np
import numpy.testing as npt

from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.test import (
    max_cycles_test,
    render_test,
    performance_benchmark,
)
from pettingzoo.test.parallel_test import parallel_api_test
from pettingzoo.test.seed_test import parallel_seed_test

# from pettingzoo.utils import parallel_to_aec


from aintelope.environments import savanna_safetygrid as safetygrid
from aintelope.environments.savanna import ACTION_MAP
from aintelope.environments.savanna_safetygrid import SavannaGridworldParallelEnv
from aintelope.environments.env_utils.distance import distance_to_closest_item


def test_gridworlds_api_parallel():
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
    parallel_env = safetygrid.SavannaGridworldParallelEnv(env_params=env_params)
    # TODO: Nathan was able to get the sequential-turn env to work, using this conversion, but not the parallel env. why??
    # sequential_env = parallel_to_aec(parallel_env)
    parallel_api_test(parallel_env, num_cycles=10)


def test_gridworlds_seed():
    env_params = {
        "override_infos": True  # Zoo parallel_seed_test is unable to compare infos unless they have simple structure.
    }
    parallel_env = lambda: safetygrid.SavannaGridworldParallelEnv(
        env_params=env_params
    )  # seed test requires lambda
    try:
        parallel_seed_test(parallel_env, num_cycles=10)
    except TypeError:
        # for some reason the test env in Git does not recognise the num_cycles neither as named or positional argument
        parallel_seed_test(parallel_env)


def test_gridworlds_agent_states():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states


def test_gridworlds_reward_agent():
    pass  # safetygrid.SavannaGridworldEnv has no reward_agent()


def test_gridworlds_move_agent():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states and move_agent()


def test_gridworlds_step_result():
    env = safetygrid.SavannaGridworldParallelEnv(
        env_params={"num_iters": 2}
    )  # default is 1 iter which means that the env is done after 1 step below and the test will fail
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


def test_gridworlds_done_step():
    env = safetygrid.SavannaGridworldParallelEnv()
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


if __name__ == "__main__" and sys.gettrace() is not None:  # detect debugging
    pytest.main([__file__])  # run tests only in this file
