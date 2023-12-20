from typing import Union, Mapping, Optional, Type
from abc import ABC, abstractmethod

import gymnasium as gym

from pettingzoo import AECEnv, ParallelEnv
import pandas as pd

PettingZooEnv = Union[AECEnv, ParallelEnv]
Environment = Union[gym.Env, PettingZooEnv]


class Agent(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def get_action(self, epsilon: float, device: str) -> Optional[int]:
        ...

    @abstractmethod
    def get_history() -> pd.DataFrame:
        ...


AGENT_REGISTRY: Mapping[str, Type[Agent]] = {}


def register_agent_class(agent_id: str, agent_class: Type[Agent]):
    if agent_id in AGENT_REGISTRY:
        raise ValueError(f"{agent_id} is already registered")
    AGENT_REGISTRY[agent_id] = agent_class


def get_agent_class(agent_id: str) -> Type[Agent]:
    if agent_id not in AGENT_REGISTRY:
        raise ValueError(f"{agent_id} is not found in agent registry")
    return AGENT_REGISTRY[agent_id]
