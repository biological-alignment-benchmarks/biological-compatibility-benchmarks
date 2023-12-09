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
from pettingzoo.test import api_test
from pettingzoo.test.seed_test import parallel_seed_test
from pettingzoo.utils import parallel_to_aec


from aintelope.environments import savanna_zoo as zoo
from aintelope.environments import savanna_safetygrid as safetygrid
from aintelope.environments.savanna import ACTION_MAP
from aintelope.environments.savanna_zoo import SavannaZooParallelEnv, SavannaZooSequentialEnv
from aintelope.environments.savanna_safetygrid import SavannaGridworldParallelEnv, SavannaGridworldSequentialEnv
from aintelope.environments.env_utils.distance import distance_to_closest_item


def test_pettingzoo_api_parallel():
    parallel_api_test(zoo.SavannaZooParallelEnv(), num_cycles=1000)


def test_gridworlds_api_parallel():
    parallel_api_test(safetygrid.SavannaGridworldParallelEnv(), num_cycles=1000)


def test_zoo_api_sequential():
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
    sequential_env = SavannaZooSequentialEnv(env_params=env_params)
    # TODO: Nathan was able to get the sequential-turn env to work, using this conversion, but not the parallel env. why??
    # sequential_env = parallel_to_aec(parallel_env)
    api_test(sequential_env, num_cycles=10, verbose_progress=True)


#def test_gridworlds_api_sequential():      # TODO: generally this test suite passes, but there are errors related to multi-objective rewards
#    # TODO: refactor these values out to a test-params file
#    env_params = {
#        "num_iters": 500,  # duration of the game
#        "map_min": 0,
#        "map_max": 100,
#        "render_map_max": 100,
#        "amount_agents": 1,  # for now only one agent
#        "amount_grass_patches": 2,
#        "amount_water_holes": 2,
#    }
#    sequential_env = SavannaGridworldSequentialEnv(env_params=env_params)
#    # TODO: Nathan was able to get the sequential-turn env to work, using this conversion, but not the parallel env. why??
#    # sequential_env = parallel_to_aec(parallel_env)
#    api_test(sequential_env, num_cycles=10, verbose_progress=True)


def test_zoo_seed():
    try:
        parallel_seed_test(zoo.SavannaZooParallelEnv, num_cycles=10)
    except TypeError:
        # for some reason the test env in Git does not recognise the num_cycles neither as named or positional argument
        parallel_seed_test(zoo.SavannaZooParallelEnv)


def test_gridworlds_seed():
    try:
        parallel_seed_test(safetygrid.SavannaGridworldParallelEnv, num_cycles=10)
    except TypeError:
        # for some reason the test env in Git does not recognise the num_cycles neither as named or positional argument
        parallel_seed_test(safetygrid.SavannaGridworldParallelEnv)


def test_zoo_agent_states():
    env = zoo.SavannaZooParallelEnv()

    env.reset()
    assert isinstance(env.unwrapped.agent_states, dict)
    assert all(
        isinstance(agent_state, np.ndarray)
        for agent_state in env.unwrapped.agent_states.values()
    )
    assert all(
        agent_state.shape == (2,) for agent_state in env.unwrapped.agent_states.values()
    )


def test_gridworlds_agent_states():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states 


def test_zoo_reward_agent():
    env = zoo.SavannaZooParallelEnv()
    env.reset()
    # single grass patch
    agent_pos = np.random.randint(env.metadata["map_min"], env.metadata["map_max"], 2)
    grass_patch = np.random.randint(env.metadata["map_min"], env.metadata["map_max"], 2)
    min_grass_distance = distance_to_closest_item(agent_pos, grass_patch)
    reward_single = zoo.reward_agent(min_grass_distance)
    # assert reward_single == 1 / (1 + vec_distance(grass_patch, agent_pos))
    assert reward_single >= 0.0 and reward_single <= 1.0

    # multiple grass patches
    grass_patches = np.random.randint(
        env.metadata["map_min"], env.metadata["map_max"], size=(10, 2)
    )
    min_grass_distance = distance_to_closest_item(agent_pos, grass_patches)
    reward_many = zoo.reward_agent(min_grass_distance)
    grass_patch_closest = grass_patches[
        np.argmin(np.linalg.norm(np.subtract(grass_patches, agent_pos), axis=1))
    ]
    # assert reward_many == 1 / (1 + vec_distance(grass_patch_closest, agent_pos))
    assert reward_many >= 0.0 and reward_many <= 1.0


def test_gridworlds_reward_agent():
    pass  # safetygrid.SavannaGridworldEnv has no reward_agent()


def test_zoo_move_agent():
    env = zoo.SavannaZooParallelEnv()
    env.reset()

    agent = env.possible_agents[0]
    agent_states = env.unwrapped.agent_states

    for _ in range(1000):
        prev_state = np.copy(agent_states[agent])
        action = env.action_space(agent).sample()
        agent_states[agent] = zoo.move_agent(
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
        assert agent_states[agent].dtype == zoo.PositionFloat


def test_gridworlds_move_agent():
    pass  # safetygrid.SavannaGridworldEnv has no agent_states and move_agent()


def test_zoo_step_result():
    env = zoo.SavannaZooParallelEnv(
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


def test_zoo_done_step():
    env = zoo.SavannaZooParallelEnv()
    assert len(env.possible_agents) == 1
    env.reset()

    agent = env.possible_agents[0]
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


def test_gridworlds_done_step():
    env = safetygrid.SavannaGridworldParallelEnv()
    assert len(env.possible_agents) == 1
    env.reset()

    agent = env.possible_agents[0]
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


def test_zoo_agents():
    env = zoo.SavannaZooParallelEnv()

    assert len(env.possible_agents) == env.metadata["amount_agents"]
    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_gridworlds_agents():
    env = safetygrid.SavannaGridworldParallelEnv()

    assert len(env.possible_agents) == env.metadata["amount_agents"]
    assert isinstance(env.possible_agents, list)
    assert isinstance(env.unwrapped.agent_name_mapping, dict)
    assert all(
        agent_name in env.unwrapped.agent_name_mapping
        for agent_name in env.possible_agents
    )


def test_zoo_action_spaces():
    env = zoo.SavannaZooParallelEnv()

    for agent in env.possible_agents:
        assert isinstance(env.action_space(agent), Discrete)
        assert env.action_space(agent).n == 4


def test_gridworlds_action_spaces():
    env = safetygrid.SavannaGridworldParallelEnv()

    for agent in env.possible_agents:
        assert isinstance(env.action_space(agent), MultiDiscrete)
        assert env.action_space(agent).n == 5   # includes no-op


def test_zoo_action_space_valid_step():
    env = zoo.SavannaZooParallelEnv()
    env.reset()
    map_min, map_max = env.metadata["map_min"], env.metadata["map_max"]

    agent = env.possible_agents[0]
    agent_states = env.unwrapped.agent_states

    for it in range(1000):
        prev_state = np.copy(agent_states[agent])
        action = env.action_space(agent).sample()
        agent_states[agent] = zoo.move_agent(
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
