# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import datetime
import logging
import os
from collections import namedtuple
from typing import Optional, Tuple
from gymnasium.spaces import Discrete

import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
from torch import nn

from aintelope.aintelope_typing import ObservationFloat

logger = logging.getLogger("aintelope.training.dqn_training")
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


def load_checkpoint(
    path,
    obs_size,
    action_space_size,
    unit_test_mode,
    hidden_sizes,
    num_conv_layers,
    conv_size,
    combine_interoception_and_vision,
):
    """
    https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
    Load a model from a checkpoint. Commented parts optional for later.

    Args:
        path: str
        obs_size: tuple, input size, numpy shape
        action_space_size: int, output size

    Returns:
        model: torch.nn.Module
    """

    model = None  # TODO

    if not unit_test_mode:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])

        model.eval()

    return model


class Trainer:
    """
    Trainer class, entry point to all things pytorch. Init a single instance for
    handling the models, register agents in for their personal models.
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, params):
        self.observation_shapes = {}
        self.action_spaces = {}

        self.hparams = params.hparams
        self.combine_interoception_and_vision = (
            params.hparams.env_params.combine_interoception_and_vision
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using GPU: " + str(self.device not in ["cpu"]))

    def reset_agent(self, agent_id):
        # TODO
        pass

    def add_agent(
        self,
        agent_id,
        observation_shape,
        action_space,
        unit_test_mode: bool,
        checkpoint: Optional[str] = None,
    ):
        """
        Register an agent.

        Args:
            agent_id (str): same as elsewhere (f.ex. "agent_0")
            observation_shape (tuple of tuples): numpy shapes of the observations (vision, interoception)
            action_space (Discrete): action_space from environment
            checkpoint: Path (string) to checkpoint, None if not available

        Returns:
            None
        """
        self.observation_shapes[agent_id] = observation_shape
        self.action_spaces[agent_id] = action_space(agent_id)

    def tiebreaking_argmax(self, arr):
        max_values_bitmap = np.isclose(arr, arr.max())
        max_values_indexes = np.flatnonzero(max_values_bitmap)

        if (
            len(max_values_indexes) == 0
        ):  # Happens when all values are infinities or nans. This would cause np.random.choice to throw.
            result = np.random.randint(0, len(arr))
        else:
            result = np.random.choice(
                max_values_indexes
            )  # TODO: seed for this random generator

        return result

    @torch.no_grad()
    def get_action(
        self,
        agent_id: str = "",
        observation: Tuple[
            npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]
        ] = None,
        info: dict = {},
        step: int = 0,
        env_layout_seed: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
    ) -> npt.NDArray:
        """
        Get action from an agent

        Args:
            agent_id (str): same as elsewhere ("agent_0" among them)
            observation (npt.NDArray[ObservationFloat]): input for the net
            step (int): used to calculate epsilon

        Returns:
            Q values array
        """

        logger.debug("debug observation", type(observation))

        if not self.combine_interoception_and_vision:
            observation = (
                torch.tensor(
                    np.expand_dims(
                        observation[0], 0
                    )  # vision     # call .flatten() in case you want to force 1D network even on 3D vision
                ),
                torch.tensor(np.expand_dims(observation[1], 0)),  # interoception
            )
            logger.debug(
                "debug observation tensor",
                (type(observation[0]), type(observation[1])),
                (observation[0].shape, observation[1].shape),
            )

            if str(self.device) not in ["cpu"]:
                observation = (
                    observation[0].cuda(self.device),
                    observation[1].cuda(self.device),
                )
        else:
            observation = torch.tensor(
                np.expand_dims(
                    observation, 0
                )  # vision     # call .flatten() in case you want to force 1D network even on 3D vision
            )
            logger.debug(
                "debug observation tensor",
                type(observation),
                observation.shape,
            )

            if str(self.device) not in ["cpu"]:
                observation = observation.cuda(self.device)

        q_values = [1]  # TODO

        return q_values

    def update_memory(
        self,
        agent_id: str,
        state: Tuple[npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]],
        action: int,
        reward: float,
        done: bool,
        next_state: Tuple[npt.NDArray[ObservationFloat], npt.NDArray[ObservationFloat]],
    ):
        """
        Add transition into agent specific ReplayMemory.

        Args:
            agent_id (str): same as elsewhere ("agent_0" among them)
            state (npt.NDArray[ObservationFloat]): input for the net
            action (int): index of action
            reward (float): reward signal
            done (bool): if agent is done
            next_state (npt.NDArray[ObservationFloat]): input for the net

        Returns:
            None
        """

        if done:
            return

        action_space = self.action_spaces[agent_id]
        if isinstance(action_space, Discrete):
            min_action = action_space.start
        else:
            min_action = action_space.min_action
        action -= min_action  # offset the action index if min_action is not zero

        # TODO

    def optimize_models(self):
        # TODO
        pass

    def save_model(
        self,
        agent_id,
        episode,
        path,
        experiment_name,
        use_separate_models_for_each_experiment,
    ):
        """
        Save model artifacts to 'path'.

        Args:
            episode (int): number of environment cycle; each cycle is divided into steps
            path (str): location where artifact is saved

        Returns:
            None
        """

        # TODO

        checkpoint_filename = agent_id
        if use_separate_models_for_each_experiment:
            checkpoint_filename += "-" + experiment_name

        filename = os.path.join(
            path,
            checkpoint_filename
            + "-"
            + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f"),
        )

        logger.info(f"Saving agent {agent_id} models to disk at {filename}")
        torch.save(
            {
                "epoch": episode,
                # TODO
            },
            filename,
        )
