import typing as typ
import logging
from pathlib import Path
from collections import OrderedDict

from omegaconf import DictConfig

import gymnasium as gym

import torch
from torch import Tensor, nn
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer, loggers as pl_loggers
from pytorch_lightning.utilities.enums import DistributedType
from pytorch_lightning.callbacks import ModelCheckpoint

from aintelope.agents.memory import ReplayBuffer, RLDataset
from aintelope.agents import get_agent_class
from aintelope.agents.instinct_agent import InstinctAgent
from aintelope.models.dqn import DQN
from aintelope.environments.savanna_gym import SavannaGymEnv

logger = logging.getLogger("aintelope.training.lightning_trainer")

AVAIL_GPUS = min(1, torch.cuda.device_count())


class DQNLightning(LightningModule):
    """Basic DQN Model."""

    def __init__(self, hparams: DictConfig) -> None:
        """Main model class

        Args:
            hparams (DictConfig): hyperparameter dictionary
        """
        super().__init__()
        self.save_hyperparameters(hparams)  # make hparams available as self.hparams

        if hparams.env == "savanna-gym-v2":
            self.env = SavannaGymEnv(env_params=hparams.env_params)
            obs_size = self.env.observation_space.shape[0]
        else:
            # GYM_INTERACTION
            # can't register as a gym env unless we rewrite as a gym env
            # gym.envs.register(
            #     id=self.hparams.env,
            #     entry_point='aintelope.environments.savanna:RawEnv',
            #     kwargs={'env_params': env_params}
            # )
            self.env = gym.make(hparams.env)
            obs_size = self.env.observation_space.shape[0]

        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(hparams.replay_size)
        self.agent = get_agent_class(hparams.agent_id)(
            self.env,
            self.net,
            self.buffer,
            hparams.warm_start_steps,
            **hparams.agent_params,
        )
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to
        initially fill up the replay buffer with experiences.

        Args:
            steps: number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def dqn_mse_loss(self, batch: typ.Tuple[Tensor, Tensor]) -> Tensor:
        """Calculates the mse loss using a mini batch from the replay buffer.

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = (
            self.net(states).gather(1, actions.unsqueeze(-1).long()).squeeze(-1)
        )

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: typ.Tuple[Tensor, Tensor], nb_batch) -> OrderedDict:
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
            self.hparams.eps_start - self.global_step * 1 / self.hparams.eps_last_frame,
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
            "total_reward": torch.tensor(self.total_reward, dtype=torch.float32).to(
                device
            ),
            "reward": torch.tensor(reward, dtype=torch.float32).to(device),
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
            "total_reward": torch.tensor(self.total_reward, dtype=torch.float32).to(
                device
            ),
        }

        self.log("episode_reward", self.episode_reward)
        self.log("total_reward", log["total_reward"])
        self.log("reward", log["reward"])
        self.log("train_loss", log["train_loss"])
        self.log("steps", status["steps"].type(torch.float32))
        self.log("epsilon", epsilon)
        self.log("done", torch.tensor(done).type(torch.float32))

        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})

    def on_train_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.hparams.log_figures_every_n_epochs == 0:
            self.logger.experiment.add_figure(
                "train_images/agent_history_thickness",
                self.agent.plot_history(),
                self.current_epoch,
            )
            self.logger.experiment.add_figure(
                "train_images/agent_history_colormap",
                self.agent.plot_history(style="colormap"),
                self.current_epoch,
            )

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


def run_experiment(cfg: DictConfig) -> None:
    dir_out, exp_name = f"{cfg.experiment_dir}", f"{cfg.experiment_name}"
    dir_experiment = Path(dir_out)

    lightning_module = DQNLightning(cfg.hparams)

    checkpoint_callback = ModelCheckpoint(
        dirpath=dir_experiment / "checkpoints",
        filename="{epoch}-{val_loss:.2f}",
        auto_insert_metric_name=True,
        save_last=True,
        save_top_k=-1,
        save_on_train_epoch_end=True,
        every_n_epochs=cfg.hparams.every_n_epochs,
    )

    if cfg.trainer_params.resume_from_checkpoint:
        checkpoint = cfg.trainer_params.checkpoint / "last.ckpt"
        logger.info(f"Checkpoint path: {checkpoint}")
    else:
        checkpoint = None
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=dir_out, name=exp_name, version=exp_name
    )

    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=cfg.trainer_params.max_epochs,
        val_check_interval=100,
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
        logger=tb_logger,
    )

    trainer.fit(lightning_module, ckpt_path=checkpoint)

    record_path = dir_experiment / "memory_records.csv"
    logger.info(f"Saving training records to disk at {record_path}")
    record_path.parent.mkdir(exist_ok=True, parents=True)
    lightning_module.agent.get_history().to_csv(record_path, index=False)

    # Notes
    # resume from a specific checkpoint
    # trainer = Trainer(resume_from_checkpoint="some/path/to/my_checkpoint.ckpt")

    # this will be handy for adding tests
    # runs only 1 training and 1 validation batch and the program ends, avoids side-effects
    # trainer = Trainer(fast_dev_run=True, enable_progress_bar=False)

    # retrieve the best checkpoint after training

    # checkpoint_callback.best_model_path
