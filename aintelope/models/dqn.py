import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# class DQN(nn.Module):
#    """Simple MLP network."""

#    def __init__(self, obs_size: tuple, n_actions: int, hidden_size: int = 128):
#        """
#        Args:
#            obs_size (tuple): tuple of [observation/state size of the environment (numpy shape), interoception size (numpy shape)]
#            n_actions (int): number of discrete actions available in the environment
#            hidden_size (int): size of hidden layers
#        """
#        super().__init__()
#        self.net = nn.Sequential(
#            nn.Linear(
#                np.prod(obs_size[0]) + obs_size[1][0],
#                hidden_size,
#            ),
#            nn.ReLU(),
#            nn.Linear(hidden_size, n_actions),
#        )

#    def forward(self, x):
#        return self.net(x.float())


class DQN(nn.Module):
    """Simple CNN network."""

    # TODO: use https://github.com/yuezuegu/torchshape
    def conv2d_shape(self, input_size, layer):
        # needed to calculate CNN to FCN flattening layer input shape since it varies depending on vision xy dimensions
        # https://stackoverflow.com/questions/70231487/output-dimensions-of-convolution-in-pytorch
        return (
            np.floor(
                (
                    np.array(input_size)
                    - np.array(layer.kernel_size)
                    + 2 * np.array(layer.padding)
                )
                / np.array(layer.stride)
            ).astype(int)
            + 1
        )

    def __init__(
        self,
        obs_size: tuple,
        n_actions: int,
        unit_test_mode: bool,
        hidden_sizes: list = [8, 8],
    ):
        """
        Args:
            obs_size (tuple): tuple of [observation/state size of the environment (numpy shape), interoception size (numpy shape)]
            n_actions (int): number of discrete actions available in the environment
            hidden_size (int): size of hidden layers
        """
        super().__init__()
        self.unit_test_mode = unit_test_mode
        if (
            unit_test_mode
        ):  # constrain the size of networks during unit testing for performance purposes
            hidden_sizes = [8, 8]

        (vision_size, interoception_size) = obs_size  # TODO: interoception

        if len(vision_size) == 1:  # old AIntelope environment
            self.cnn = False

            # 1D vision network
            self.fc4 = nn.Linear(vision_size[0], hidden_sizes[0])

            # interoception network
            self.fc5 = nn.Linear(interoception_size[0], hidden_sizes[0])

            # combined network
            self.fc6 = nn.Linear(hidden_sizes[0] + hidden_sizes[0], n_actions)

        else:  # Gridworlds environment
            self.cnn = True

            num_vision_features = vision_size[
                0
            ]  # feature vector is the first dimension
            vision_xy = vision_size[1:]  # vision xy shape starts from index 1

            # 3D vision network
            self.conv1 = nn.Conv2d(
                num_vision_features, hidden_sizes[0], kernel_size=1, stride=1
            )  # this layer with kernel_size=1 enables mixing of information across feature vector channels
            output_size = self.conv2d_shape(vision_xy, self.conv1)

            if not unit_test_mode:
                self.conv2 = nn.Conv2d(
                    hidden_sizes[0], hidden_sizes[0], kernel_size=3, stride=1
                )
                output_size = self.conv2d_shape(output_size, self.conv2)

                self.conv3 = nn.Conv2d(
                    hidden_sizes[0], hidden_sizes[0], kernel_size=3, stride=1
                )
                output_size = self.conv2d_shape(output_size, self.conv3)

            self.fc4 = nn.Linear(
                np.prod(output_size) * hidden_sizes[0], hidden_sizes[1]
            )  # flattens convolutional layers output

            # interoception network
            self.fc5 = nn.Linear(interoception_size[0], hidden_sizes[0])

            # combined network
            self.fc6 = nn.Linear(hidden_sizes[0] + hidden_sizes[1], n_actions)

    def forward(self, observation):
        (vision_batch, interoception_batch) = observation

        x = vision_batch.float()
        y = interoception_batch.float()

        if not self.cnn:  # old AIntelope environment
            # 1D vision network
            x = F.relu(self.fc4(x))

            # interoception network
            y = F.relu(self.fc5(y))

            # combined network
            z = torch.cat([x, y], axis=1)
            return self.fc6(z)

        else:  # Gridworlds environment
            # 3D vision network
            x = F.relu(self.conv1(x))

            if not self.unit_test_mode:
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))

            x = x.view(
                x.size(0), -1
            )  # keep batch size at dimension 0, flatten the remaining output dimensions of conv2 layer into 1D : (batch size, channels, height, width) -> (batch size, features)
            x = F.relu(self.fc4(x))

            # interoception network
            y = F.relu(self.fc5(y))

            # combined network
            z = torch.cat([x, y], axis=1)
            return self.fc6(z)
