import datetime
import logging
import os
from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.optim as optim
from torch import nn

from aintelope.aintelope_typing import ObservationFloat
from aintelope.models.dqn import DQN
from aintelope.training.memory import ReplayMemory

logger = logging.getLogger("aintelope.training.dqn_training")
Transition = namedtuple(
    "Transition", ("state", "action", "reward", "done", "next_state")
)


def load_checkpoint(
    path, obs_size, action_space_size, unit_test_mode, hidden_sizes, num_conv_layers, conv_size
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

    model = DQN(
        obs_size,
        action_space_size,
        unit_test_mode=unit_test_mode,
        hidden_sizes=hidden_sizes,
        num_conv_layers=num_conv_layers,
        conv_size=conv_size,
    )

    if not unit_test_mode:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch = checkpoint['epoch']
        # loss = checkpoint['loss']

        model.eval()

    return model


class Trainer:
    """
    Trainer class, entry point to all things pytorch. Init a single instance for
    handling the models, register agents in for their personal models.
    https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    """

    def __init__(self, params):
        self.policy_nets = {}
        self.target_nets = {}
        self.losses = {}
        self.replay_memories = {}
        self.optimizers = {}
        self.observation_shapes = {}
        self.action_spaces = {}

        self.hparams = params.hparams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using GPU: " + str(self.device not in ["cpu"]))

    def reset_agent(self, agent_id):
        self.replay_memories[agent_id] = ReplayMemory(
            self.hparams.model_params.replay_size
        )

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
        self.replay_memories[agent_id] = ReplayMemory(
            self.hparams.model_params.replay_size
        )

        if not checkpoint:
            self.policy_nets[agent_id] = DQN(
                self.observation_shapes[agent_id],
                self.action_spaces[agent_id].n,
                unit_test_mode=unit_test_mode,
                hidden_sizes=self.hparams.model_params.hidden_sizes,
                num_conv_layers=self.hparams.model_params.num_conv_layers,
                conv_size=self.hparams.model_params.conv_size,
            ).to(self.device)
        else:
            self.policy_nets[agent_id] = load_checkpoint(
                checkpoint,
                self.observation_shapes[agent_id],
                self.action_spaces[agent_id].n,
                unit_test_mode=unit_test_mode,
                hidden_sizes=self.hparams.model_params.hidden_sizes,
                num_conv_layers=self.hparams.model_params.num_conv_layers,
                conv_size=self.hparams.model_params.conv_size,
            ).to(self.device)

        self.target_nets[agent_id] = DQN(
            self.observation_shapes[agent_id],
            self.action_spaces[agent_id].n,
            unit_test_mode=unit_test_mode,
            hidden_sizes=self.hparams.model_params.hidden_sizes,
            num_conv_layers=self.hparams.model_params.num_conv_layers,
            conv_size=self.hparams.model_params.conv_size,
        ).to(self.device)
        self.target_nets[agent_id].load_state_dict(
            self.policy_nets[agent_id].state_dict()
        )
        self.optimizers[agent_id] = optim.AdamW(
            self.policy_nets[agent_id].parameters(),
            lr=self.hparams.model_params.lr,
            amsgrad=self.hparams.amsgrad,
        )

    def tiebreaking_argmax(self, arr):
        """Avoids the agent from repeatedly taking move-left action when the instinct tells the agent to move away from current cell in any direction. Then the instinct will not provide any q value difference in its q values for the different directions, they would be equal. Naive np.argmax would just return the index of first moving action, which happens to be always move-left action."""
        max_values_bitmap = np.isclose(arr, arr.max())
        max_values_indexes = np.flatnonzero(max_values_bitmap)
        result = np.random.choice(max_values_indexes)
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
        trial: int = 0,
        episode: int = 0,
        pipeline_cycle: int = 0,
        action_biases: list = None,
    ) -> Optional[int]:
        """
        Get action from an agent

        Args:
            agent_id (str): same as elsewhere ("agent_0" among them)
            observation (npt.NDArray[ObservationFloat]): input for the net
            step (int): used to calculate epsilon

        Returns:
            None
        """

        # TODO: warn if last_frame=0/1 or last_trial=0/1 or last_episode=0/1 in any of the below values: for disabling the epsilon counting for corresponding variable one should use -1
        epsilon = self.hparams.model_params.eps_start - self.hparams.model_params.eps_end        
        if self.hparams.model_params.eps_last_frame > 1:     
            epsilon *= max(0, 1 - step / self.hparams.model_params.eps_last_frame)             
        if self.hparams.model_params.eps_last_trial > 1:
            epsilon *= max(
                0, 1 - trial / self.hparams.model_params.eps_last_trial
            ) 
        if self.hparams.model_params.eps_last_episode > 1:  
            epsilon *= max(
                0, 1 - episode / self.hparams.model_params.eps_last_episode
            )
        if self.hparams.model_params.eps_last_pipeline_cycle > 1: 
            epsilon *= max(
                0, 1 - pipeline_cycle / self.hparams.model_params.eps_last_pipeline_cycle
            )
        epsilon += self.hparams.model_params.eps_end

        # print(f"Epsilon: {epsilon}")

        if np.random.random() < epsilon:
            action = self.action_spaces[agent_id].sample()
        else:
            logger.debug("debug observation", type(observation))

            observation = (
                torch.tensor(
                    np.expand_dims(
                        observation[0], 0
                    ),  # vision     # call .flatten() in case you want to force 1D network even on 3D vision
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

            q_values = self.policy_nets[agent_id](observation).cpu().numpy()
            action = self.tiebreaking_argmax(q_values) + action_space.min_action    # when no axis is provided, argmax returns index into flattened array

            #q_values = self.policy_nets[agent_id](observation)
            #_, action = torch.max(q_values, dim=1)
            #action = int(action.item()) + action_space.min_action

        return action

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
        # add experience to torch device if bugged    # TODO: what does bugging mean here?
        if done:
            return

        state = (
            torch.tensor(state[0], dtype=torch.float32, device=self.device).unsqueeze(
                0
            ),
            torch.tensor(state[1], dtype=torch.float32, device=self.device).unsqueeze(
                0
            ),
        )

        action -= self.action_spaces[agent_id].min_action   # offset the action index if min_action is not zero
        action = torch.tensor(action, device=self.device).unsqueeze(0).view(1, 1)
        reward = torch.tensor(
            reward, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        next_state = (
            torch.tensor(
                next_state[0], dtype=torch.float32, device=self.device
            ).unsqueeze(0),
            torch.tensor(
                next_state[1], dtype=torch.float32, device=self.device
            ).unsqueeze(0),
        )

        self.replay_memories[agent_id].push(state, action, reward, done, next_state)

    def optimize_models(self):
        """
        Optimize personal models based on contents of ReplayMemory of each agent.
        Check: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

        Args:
            None

        Returns:
            None
        """
        for agent_id in self.policy_nets.keys():
            if (
                len(self.replay_memories[agent_id])
                < self.hparams.model_params.batch_size
            ):
                continue  # TODO: there was return, I guess continue is more correct here?

            transitions = self.replay_memories[agent_id].sample(
                self.hparams.model_params.batch_size
            )
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, batch.next_state)),
                device=self.device,
                dtype=torch.bool,
            )
            non_final_next_states = (
                torch.cat([s[0] for s in batch.next_state if s is not None]),
                torch.cat([s[1] for s in batch.next_state if s is not None]),
            )
            state_batch = (
                torch.cat([s[0] for s in batch.state]),
                torch.cat([s[1] for s in batch.state]),
            )
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            policy_net = self.policy_nets[agent_id]
            target_net = self.target_nets[agent_id]
            state_action_values = policy_net(state_batch).gather(1, action_batch.long())

            next_state_values = torch.zeros(
                self.hparams.model_params.batch_size, device=self.device
            )
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(
                    non_final_next_states
                ).max(1)[0]

            expected_state_action_values = (
                next_state_values * self.hparams.model_params.gamma
            ) + reward_batch

            criterion = nn.SmoothL1Loss()
            loss = criterion(
                state_action_values, expected_state_action_values.unsqueeze(1)
            )
            self.losses[agent_id] = loss

            self.optimizers[agent_id].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
            self.optimizers[agent_id].step()

            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[
                    key
                ] * self.hparams.model_params.tau + target_net_state_dict[key] * (
                    1 - self.hparams.model_params.tau
                )
            target_net.load_state_dict(target_net_state_dict)

    def save_models(self, episode, path, experiment_name, use_separate_models_for_each_experiment):
        """
        Save model artifacts to 'path'.

        Args:
            episode (int): number of environment cycle; each cycle is divided into steps
            path (str): location where artifact is saved

        Returns:
            None
        """
        for agent_id in self.policy_nets.keys():
            model = self.policy_nets[agent_id]
            optimizer = self.optimizers[agent_id]
            loss = 1.0
            if agent_id in self.losses:
                loss = self.losses[agent_id]

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
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                filename,
            )
