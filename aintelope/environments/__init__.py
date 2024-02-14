from typing import Union, Mapping, Type
from abc import ABC

import gymnasium as gym

from pettingzoo import AECEnv, ParallelEnv

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


ENV_REGISTRY: Mapping[str, Type[Environment]] = {}


class Irrelevant(
    ABC
):  # TODO CLEANUP: Needed to import the below functions. Discuss in slack
    ...


def register_env_class(env_id: str, env_class: Type[Environment]):
    if env_id in ENV_REGISTRY:
        raise ValueError(f"{env_id} is already registered")
    ENV_REGISTRY[env_id] = env_class


def get_env_class(env_id: str) -> Type[Environment]:
    if env_id not in ENV_REGISTRY:
        raise ValueError(f"{env_id} is not found in env registry")
    return ENV_REGISTRY[env_id]
