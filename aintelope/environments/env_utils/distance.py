from typing import List, Union

import numpy as np

from aintelope.environments.typing import PositionFloat


def vec_distance(
    vec_a: Union[np.ndarray, List[PositionFloat]],
    vec_b: Union[np.ndarray, List[PositionFloat]],
) -> np.float64:
    return np.linalg.norm(np.subtract(vec_a, vec_b))


def distance_to_closest_item(
    agent_pos: Union[np.ndarray, List[PositionFloat]], items: np.ndarray
) -> np.float64:
    if len(items.shape) == 1:
        items = np.expand_dims(items, 0)

    closest_item = items[
        np.argmin(np.linalg.norm(np.subtract(items, agent_pos), axis=1))
    ]
    return vec_distance(closest_item, agent_pos)
