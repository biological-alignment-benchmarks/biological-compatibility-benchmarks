import pytest
import numpy as np
import numpy.testing as npt

try:
    from gymnasium.spaces import Discrete

    gym_v26 = True
except:
    from gym.spaces import Discrete

    gym_v26 = False


from aintelope.environments.savanna_gym import SavannaGymEnv
