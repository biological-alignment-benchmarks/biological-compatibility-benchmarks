import sys
import os
import pytest
import numpy as np

from aintelope.environments import savanna as sut


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
    for (
        agent_name,
        agent_state,
    ) in (
        env.agent_states.items()
    ):  # TODO: currently old AIntelope agent_state does not contain interoception. This is not used anyway and is compatible with rest of the code though
        agent_state_env = agent_state
        agent_state_func = sut.get_agent_pos_from_state(
            agent_state_env, info={}, agent_name="agent_0"
        )
        assert agent_state_env[0] == agent_state_func[0]


def test_observation_spaces():
    pass  # TODO


if __name__ == "__main__" and os.name == "nt":  # detect debugging
    pytest.main([__file__])  # run tests only in this file
    # pytest.main([__file__ + "::test_grass_patches"])
