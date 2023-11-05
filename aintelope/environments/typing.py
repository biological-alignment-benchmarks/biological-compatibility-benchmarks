from typing import Dict

import numpy as np

ObservationFloat = np.float32
PositionFloat = np.float32
Action = int
AgentId = str
AgentStates = Dict[AgentId, np.ndarray]

Observation = np.ndarray
Reward = float
Info = dict
