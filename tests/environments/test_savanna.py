import pytest
import numpy as np
import numpy.testing as npt

from gymnasium.spaces import Discrete


from aintelope.environments import savanna as sut

# Base Savanna is not yet PettingZoo or Gym env
# just the shared elements those will depend on


def test_max_cycles():
    # currently the environment does not accept parameters like max_cycles
    # max_cycles_test(sut.env)
    pass


def test_render():
    # TODO: close method not implemented
    # render_test(sut.env)
    pass


def test_grass_patches():
    env = sut.SavannaEnv()

    with pytest.raises(AttributeError):
        env.grass_patches
    with pytest.raises(AttributeError):
        env.grass_patches

    env.reset()
    assert len(env.grass_patches) == env.metadata["amount_grass_patches"]
    assert isinstance(env.grass_patches, np.ndarray)
    assert env.grass_patches.shape[1] == 2


def test_get_agent_pos_from_stats():
    env = sut.SavannaEnv()
    env.reset()
    assert isinstance(env.agent_states, dict)
    for agent_state in env.agent_states:
        agent_state_env = agent_state
        agent_state_func = sut.get_agent_pos_from_state(agent_state_env)
        assert [agent_state_env[0], agent_state_env[1]] == agent_state_func


def test_observation_spaces():
    pass  # TODO
