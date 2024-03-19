from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy.typing as npt

from aintelope.aintelope_typing import ObservationFloat
from pettingzoo import AECEnv, ParallelEnv

Environment = Union[AECEnv, ParallelEnv]


class Agent(ABC):
    @abstractmethod
    def reset(self, state, info, env_class) -> None:
        ...

    @abstractmethod
    def get_action(
        self,
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        step: int = 0,
        trial: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
    ) -> Optional[int]:
        ...

    @abstractmethod
    def update(
        self,
        env: Environment = None,
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        score: float = 0.0,
        done: bool = False,
        test_mode: bool = False,
        save_path: Optional[str] = None,  # TODO: this is unused right now
    ) -> list:
        ...
