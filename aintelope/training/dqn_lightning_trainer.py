import typing as typ
from collections import OrderedDict

import gym
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities import DistributedType
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from aintelope.agents.memory import ReplayBuffer, RLDataset
from aintelope.agents.q_agent import Agent
from aintelope.models.dqn import DQN

from aintelope.environments.savanna_gym import SavannaEnv

AVAIL_GPUS = min(1, torch.cuda.device_count())


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(
        self,
        batch_size: int = 16,
        lr: float = 1e-3,
        env: str = "CartPole-v1",
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 500,
        warm_start_steps: int = 1000,
        env_params: dict = {}
    ) -> None:
        """
        Args:
            batch_size: size of the batches
            lr: learning rate
            env: gym environment tag
            gamma: discount factor
            sync_rate: how many frames do we update the target network
            replay_size: capacity of the replay buffer
            warm_start_size: how many samples do we use to fill our buffer at
                the start of training
            eps_last_frame: what frame should epsilon stop decaying
            eps_start: starting value of epsilon
            eps_end: final value of epsilon
            episode_length: max length of an episode
            warm_start_steps: max episode reward in the environment
        """
        super().__init__()
        self.save_hyperparameters()
        if self.hparams.env == 'savanna_v1':
            self.env = SavannaEnv(env_params=env_params)
            obs_size = self.env.observation_space.shape[0]
        else:
            # GYM_INTERACTION
            # can't register as a gym env unless we rewrite as a gym env
            # gym.envs.register(
            #     id=self.hparams.env,
            #     entry_point='aintelope.environments.savanna:RawEnv',
            #     kwargs={'env_params': env_params}
            # )
            self.env = gym.make(self.hparams.env)
            obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to
        initially fill up the replay buffer with experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of
        each action as an output.

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: typ.Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = (
            self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = (
            next_state_values * self.hparams.gamma + rewards
        )

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(
        self, batch: typ.Tuple[Tensor, Tensor], nb_batch
    ) -> OrderedDict:
        """Carries out a single step through the environment to update the
        replay buffer. Then calculates loss based on the minibatch recieved.

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start
            - self.global_step * 1 / self.hparams.eps_last_frame,
        )

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        # if self.trainer._distrib_type in {
        # DistributedType.DP, DistributedType.DDP2
        # }:
        # loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            "total_reward": torch.tensor(self.total_reward).to(device),
            "reward": torch.tensor(reward).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward).to(device),
        }

        self.log("episode_reward", self.episode_reward)
        self.log("total_reward", log["total_reward"])
        self.log("reward", log["reward"])
        self.log("train_loss", log["train_loss"])
        self.log("steps", status["steps"])
        self.log("epsilon", epsilon)
        self.log("done", done)

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def configure_optimizers(self) -> typ.List[Optimizer]:
        """Initialize Adam optimizer."""
        optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving
        experiences.
        """
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"


def run_experiment(hparams={}):
    model = DQNLightning(**hparams)
    trainer = Trainer(gpus=AVAIL_GPUS, max_epochs=1000, val_check_interval=100)
    trainer.fit(model)


if __name__ == "__main__":

    hparams = {
        'batch_size': 16,
        'lr': 1e-3,
        'env': "savanna_v1",
        'gamma': 0.99,
        'sync_rate': 10,
        'replay_size': 1000,
        'warm_start_size': 1000,
        'eps_last_frame': 1000,
        'eps_start': 1.0,
        'eps_end': 0.01,
        'episode_length': 500,
        'warm_start_steps': 1000,
        'env_params': {
            'NUM_ITERS': 500,  # duration of the game
            'MAP_MIN': 0,
            'MAP_MAX': 100,
            'render_map_max': 100,
            'AMOUNT_AGENTS': 1,  # for now only one agent
            'AMOUNT_GRASS_PATCHES': 2
        }
    }
    run_experiment(hparams)
