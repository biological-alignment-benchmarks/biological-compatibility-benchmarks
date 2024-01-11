from collections import deque, namedtuple
import random


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


class ReplayMemory(object):
    """
    Replay memory for each agent, saves transitions (from RL literature).
    """

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
