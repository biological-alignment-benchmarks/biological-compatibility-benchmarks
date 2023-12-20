import os
import sys
import time
import pytest
import numpy as np
import numpy.testing as npt

from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo.test import (
    max_cycles_test,
    render_test,
    performance_benchmark,
)
from pettingzoo.test import api_test
from pettingzoo.test.seed_test import seed_test

# from pettingzoo.utils import parallel_to_aec


from aintelope.environments import savanna_safetygrid as safetygrid
from aintelope.environments.savanna import ACTION_MAP
from aintelope.environments.savanna_safetygrid import SavannaGridworldSequentialEnv
from aintelope.environments.env_utils.distance import distance_to_closest_item


def test_gridworlds_api_sequential():
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
        # "seed": seed,    # TODO
    }
    env = safetygrid.SavannaGridworldSequentialEnv(env_params=env_params)

    # env = parallel_to_aec(parallel_env)
    api_test(env, num_cycles=10, verbose_progress=True)


def test_gridworlds_seed():
    seed = 0
    for index in range(
        0, 3
    ):  # construct the environment multiple times with different seeds
        seed += 1
        env_params = {
            "override_infos": True,  # Zoo seed_test is unable to compare infos unless they have simple structure.
            "seed": seed,
        }
        env = lambda: safetygrid.SavannaGridworldSequentialEnv(
            env_params=env_params
        )  # seed test requires lambda
        try:
            seed_test(env, num_cycles=10)
        except TypeError:
            # for some reason the test env in Git does not recognise the num_cycles neither as named or positional argument
            seed_test(env)


def test_gridworlds_agent_states():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states


def test_gridworlds_reward_agent():
    pass  # safetygrid.SavannaGridworldEnv has no reward_agent()


def test_gridworlds_move_agent():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states and move_agent()


def test_gridworlds_step_result():
    env = safetygrid.SavannaGridworldSequentialEnv(
        env_params={"num_iters": 2}
    )  # default is 1 iter which means that the env is done after 1 step below and the test will fail
    num_agents = len(env.possible_agents)
    assert num_agents, f"expected 1 agent, got: {num_agents}"
    env.reset()

    agent = env.agent_selection
    action = env.action_space(agent).sample()

    env.step(action)
    # NB! env.last() provides observation from NEXT agent in case of multi-agent environment
    (
        observation,
        reward,
        terminated,
        truncated,
        info,
    ) = env.last()  # TODO: multi-agent iteration
    done = terminated or truncated

    assert not done
    assert isinstance(observation, np.ndarray), "observation of agent is not an array"
    assert isinstance(reward, np.float64), "reward of agent is not a float64"


def test_gridworlds_done_step():
    env = safetygrid.SavannaGridworldSequentialEnv(env_params={"amount_agents": 1})
    assert len(env.possible_agents) == 1
    env.reset()

    for _ in range(env.metadata["num_iters"]):
        agent = env.agent_selection
        action = env.action_space(agent).sample()
        env.step(action)
        # env.last() provides observation from NEXT agent in case of multi-agent environment
        terminated = env.terminations[agent]
        truncated = env.truncations[agent]
        done = terminated or truncated

    assert done
    with pytest.raises(ValueError):
        action = env.action_space(agent).sample()
        env.step(action)


def test_gridworlds_agents():
    env = safetygrid.SavannaGridworldSequentialEnv()

    assert len(env.possible_agents) == env.metadata["amount_agents"]
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
