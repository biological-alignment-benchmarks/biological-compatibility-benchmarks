import os

import numpy as np
import numpy.testing as npt
import pytest
from gymnasium.spaces import Discrete
from pettingzoo.test import api_test
from pettingzoo.test.seed_test import seed_test

from aintelope.environments.env_utils.distance import distance_to_closest_item
from aintelope.environments.savanna import ACTION_MAP, move_agent, reward_agent
from aintelope.environments.savanna_zoo import SavannaZooSequentialEnv
from aintelope.environments.typing import PositionFloat


@pytest.mark.parametrize("execution_number", range(10))
def test_zoo_api_sequential(execution_number):
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
    env = SavannaZooSequentialEnv(env_params=env_params)
    env.seed(execution_number)

    # env = parallel_to_aec(parallel_env)
    api_test(env, num_cycles=10, verbose_progress=True)


@pytest.mark.parametrize("execution_number", range(10))
def test_zoo_api_sequential_with_death(execution_number):
    # TODO: refactor these values out to a test-params file
    env_params = {
        "num_iters": 500,  # duration of the game
        "map_min": 0,
        "map_max": 100,
        "render_map_max": 100,
        "amount_agents": 2,  # needed for death test
        "amount_grass_patches": 2,
        "amount_water_holes": 2,
        "test_death": True,
    }
    env = SavannaZooSequentialEnv(env_params=env_params)
    env.seed(execution_number)

    # env = parallel_to_aec(parallel_env)
    api_test(env, num_cycles=10, verbose_progress=True)


@pytest.mark.parametrize("execution_number", range(10))
def test_zoo_seed(execution_number):
    np.random.seed(execution_number)

    try:
        seed_test(SavannaZooSequentialEnv, num_cycles=10)
    except TypeError:
        # for some reason the test env in Git does not recognise the num_cycles
        # neither as named or positional argument
        seed_test(SavannaZooSequentialEnv)


def test_zoo_agent_states():
    env = SavannaZooSequentialEnv()

    env.reset()
    assert isinstance(env.unwrapped.agent_states, dict)
    assert all(
        isinstance(agent_state, np.ndarray)
        for agent_state in env.unwrapped.agent_states.values()
    )
    assert all(
        agent_state.shape == (2,) for agent_state in env.unwrapped.agent_states.values()
    )


@pytest.mark.parametrize("execution_number", range(10))
def test_zoo_reward_agent(execution_number):
    env = SavannaZooSequentialEnv()
    env.reset(seed=execution_number)
    # single grass patch
    agent_pos = np.random.randint(env.metadata["map_min"], env.metadata["map_max"], 2)
    grass_patch = np.random.randint(env.metadata["map_min"], env.metadata["map_max"], 2)
    min_grass_distance = distance_to_closest_item(agent_pos, grass_patch)
    reward_single = reward_agent(min_grass_distance)
    exp_reward = min_grass_distance <= 1
    assert reward_single == exp_reward

    # multiple grass patches
    grass_patches = np.random.randint(
        env.metadata["map_min"], env.metadata["map_max"], size=(10, 2)
    )
    min_grass_distance = distance_to_closest_item(agent_pos, grass_patches)
    reward_many = reward_agent(min_grass_distance)
    exp_reward = min_grass_distance <= 1
    assert reward_many == exp_reward


def test_zoo_move_agent():
    env = SavannaZooSequentialEnv()
    env.reset()

    agent = env.possible_agents[0]
    agent_states = env.unwrapped.agent_states

    for _ in range(1000):
        prev_state = np.copy(agent_states[agent])
        action = env.action_space(agent).sample()
        agent_states[agent] = move_agent(
            agent_states[agent],
            action,
            map_min=env.metadata["map_min"],
            map_max=env.metadata["map_max"],
        )
        npt.assert_array_equal(
            np.clip(
                prev_state + ACTION_MAP[action],
                env.metadata["map_min"],
                env.metadata["map_max"],
            ),
            agent_states[agent],
        )
        assert (
            env.metadata["map_min"] <= agent_states[agent][0] <= env.metadata["map_max"]
        )
        assert (
            env.metadata["map_min"] <= agent_states[agent][1] <= env.metadata["map_max"]
        )
        assert agent_states[agent].dtype == PositionFloat


@pytest.mark.parametrize("execution_number", range(10))
def test_zoo_step_result(execution_number):
    # default is 1 iter which means that the env is done after 1 step below and the
    # test will fail
    env = SavannaZooSequentialEnv(env_params={"num_iters": 2})
    num_agents = len(env.possible_agents)
    assert num_agents, f"expected 1 agent, got: {num_agents}"
    env.seed(execution_number)
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
    assert isinstance(observation, np.ndarray), "observation of agent is not an array"
    assert isinstance(reward, np.float64), "reward of agent is not a float64"


@pytest.mark.parametrize("execution_number", range(10))
def test_zoo_done_step(execution_number):
    env = SavannaZooSequentialEnv(env_params={"amount_agents": 1})
    assert len(env.possible_agents) == 1
    env.seed(execution_number)
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


def test_zoo_agents():
    env = SavannaZooSequentialEnv()

    assert len(env.possible_agents) == env.metadata["amount_agents"]
    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_zoo_action_spaces():
    env = SavannaZooSequentialEnv()

    for agent in env.possible_agents:
        assert isinstance(env.action_space(agent), Discrete)
        assert env.action_space(agent).n == 4


def test_zoo_action_space_valid_step():
    env = SavannaZooSequentialEnv()
    env.reset()
    map_min, map_max = env.metadata["map_min"], env.metadata["map_max"]

    agent = env.possible_agents[0]
    agent_states = env.unwrapped.agent_states

    for it in range(1000):
        prev_state = np.copy(agent_states[agent])
        action = env.action_space(agent).sample()
        agent_states[agent] = move_agent(
            agent_states[agent], action, map_min=map_min, map_max=map_max
        )
        step_vec = agent_states[agent] - prev_state
        if np.array_equal(step_vec, np.array([0, 0])):
            outside_state = prev_state + ACTION_MAP[action]
            assert (outside_state < map_min).any() or (outside_state > map_max).any()
        else:
            assert (
                step_vec.tolist() in ACTION_MAP.tolist()
            ), f"Invalid step occured {step_vec} at iteration {it}"


def test_max_cycles():
    # currently the environment does not accept parameters like max_cycles
    # max_cycles_test(SavannaZooSequentialEnv)
    pass


def test_render():
    # TODO: close method not implemented
    # render_test(SavannaZooSequentialEnv)
    pass


def test_performance_benchmark():
    # will print only timing to stdout; not shown per default
    # performance_benchmark(SavannaZooSequentialEnv())
    pass


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])  # run tests only in this file
    # test_zoo_api_sequential_with_death()
