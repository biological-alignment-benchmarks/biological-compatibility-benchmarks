import pytest
import numpy as np
import numpy.testing as npt
from gym.spaces import Discrete


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
    assert len(env.grass_patches) == env.metadata['amount_grass_patches']
    assert isinstance(env.grass_patches, np.ndarray)
    assert env.grass_patches.shape[1] == 2


def test_observation_spaces():
    pass  # TODO
