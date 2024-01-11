from torch import nn


class DQN(nn.Module):
    """Simple MLP network."""

    def __init__(self, obs_size: tuple, n_actions: int, hidden_size: int = 128):
        """
        Args:
            obs_size (tuple): observation/state size of the environment (numpy shape)
            n_actions (int): number of discrete actions available in the environment
            hidden_size (int): size of hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                obs_size[0], hidden_size
            ),  # TODO: consider all obs_size dimensions
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())
