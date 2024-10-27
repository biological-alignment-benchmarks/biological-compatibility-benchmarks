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
        hidden_sizes: list = [8, 16, 8],
        num_conv_layers: int = 1,
        conv_size: int = 3,
        combine_interoception_and_vision: bool = False,
    ):
        """
        Args:
            obs_size (tuple): tuple of [observation/state size of the environment (numpy shape), interoception size (numpy shape)]
            n_actions (int): number of discrete actions available in the environment
            hidden_size (int): size of hidden layers
        """
        super().__init__()
        self.unit_test_mode = unit_test_mode
        self.combine_interoception_and_vision = combine_interoception_and_vision
        self.num_conv_layers = (
            num_conv_layers if not unit_test_mode else min(1, num_conv_layers)
        )

        if (
            unit_test_mode
        ):  # constrain the size of networks during unit testing for performance purposes
            hidden_sizes = [8, 8, 8]

        if not self.combine_interoception_and_vision:
            (vision_size, interoception_size) = obs_size
        else:
            vision_size = obs_size
            # interoception_size = 0

        if len(vision_size) == 1:  # old AIntelope environment
            self.cnn = False

            # 1D vision network
            self.fc4 = nn.Linear(vision_size[0], hidden_sizes[0])

            if not self.combine_interoception_and_vision:
                # interoception network
                self.fc5 = nn.Linear(interoception_size[0], hidden_sizes[0])

                # combined network
                self.fc6 = nn.Linear(hidden_sizes[0] + hidden_sizes[0], n_actions)

            else:
                self.fc6 = nn.Linear(hidden_sizes[0], n_actions)

        else:  # Gridworlds environment
            self.cnn = True

            if self.num_conv_layers == 0:
                output_size = vision_size
                conv_output_hidden_size = 1
            else:
                num_vision_features = vision_size[
                    0
                ]  # feature vector is the first dimension
                output_size = vision_size[1:]  # vision xy shape starts from index 1

                # 3D vision network
                self.conv = nn.ModuleList(
                    []
                )  # without ModuleList, the layers would not be transferred to GPU

                # NB! the first conv layer has different kernel size
                self.conv.append(
                    nn.Conv2d(
                        num_vision_features, hidden_sizes[0], kernel_size=1, stride=1
                    )
                )  # this layer with kernel_size=1 enables mixing of information across feature vector channels
                output_size = self.conv2d_shape(output_size, self.conv[0])

                for i in range(1, self.num_conv_layers):
                    self.conv.append(
                        nn.Conv2d(
                            hidden_sizes[0],
                            hidden_sizes[0],
                            kernel_size=conv_size,
                            stride=1,
                        )
                    )
                    output_size = self.conv2d_shape(output_size, self.conv[1])

                conv_output_hidden_size = hidden_sizes[0]

            self.fc4 = nn.Linear(
                np.prod(output_size) * conv_output_hidden_size, hidden_sizes[1]
            )  # flattens convolutional layers output

            if not self.combine_interoception_and_vision:
                # interoception network
                self.fc5 = nn.Linear(interoception_size[0], hidden_sizes[2])

                # combined network
                self.fc6 = nn.Linear(hidden_sizes[1] + hidden_sizes[2], n_actions)
            else:
                self.fc6 = nn.Linear(hidden_sizes[1], n_actions)

    def forward(self, observation):
        if not self.combine_interoception_and_vision:
            (vision_batch, interoception_batch) = observation
        else:
            vision_batch = observation

        x = vision_batch.float()
        if not self.combine_interoception_and_vision:
            y = interoception_batch.float()

        if not self.cnn:  # old AIntelope environment
            # 1D vision network
            x = F.relu(self.fc4(x))

            if not self.combine_interoception_and_vision:
                # interoception network
                y = F.relu(self.fc5(y))

                # combined network
                z = torch.cat([x, y], axis=1)
                return self.fc6(z)
            else:
                return self.fc6(x)

        else:  # Gridworlds environment
            # 3D vision network
            for i in range(0, self.num_conv_layers):
                x = F.relu(self.conv[i](x))

            x = x.view(
                x.size(0), -1
            )  # keep batch size at dimension 0, flatten the remaining output dimensions of conv2 layer into 1D : (batch size, channels, height, width) -> (batch size, features)
            x = F.relu(self.fc4(x))

            if not self.combine_interoception_and_vision:
                # interoception network
                y = F.relu(self.fc5(y))

                # combined network
                z = torch.cat([x, y], axis=1)
                return self.fc6(z)
            else:
                return self.fc6(x)
