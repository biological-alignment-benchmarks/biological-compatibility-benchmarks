import typing as typ
from collections import deque, namedtuple

import numpy as np
from torch.utils.data.dataset import IterableDataset

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "done", "next_state"],
)


class ReplayBuffer:
    """Replay Buffer for storing past experiences allowing the agent to learn
    from them.

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> typ.Tuple:
        if batch_size > len(self.buffer):
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices)
        )

        return (
            states,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states),
        )

    def fetch_recent_states(self, n):
        indices = list(range(max(0, len(self.buffer) - n), len(self.buffer)))
        if len(indices) == 0:
            return []
        states, actions, rewards, dones, next_states = zip(
            *(self.buffer[idx] for idx in indices)
        )
        return states

    def fetch_recent_memories(self, n):
        indices = list(range(max(0, len(self.buffer) - n), len(self.buffer)))
        if len(indices) == 0:
            return []
        memories = zip(*(self.buffer[idx] for idx in indices))
        return memories

    def get_action_from_memory(self, memory):
        return memory[1]

    def get_reward_from_memory(self, memory):
        return memory[2]


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated
    with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> typ.Iterable:
        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
