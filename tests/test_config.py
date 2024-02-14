from typing import Tuple, Dict
import pathlib

from omegaconf import DictConfig, OmegaConf
import pytest


def constants() -> DictConfig:
    constants_dict = {
        "PROJECT": "aintelope",
        "BASELINE": "run-training-baseline",
        "INSTINCT": "run-training-instinct",
    }
    return OmegaConf.create(constants_dict)


@pytest.fixture
def root_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


@pytest.fixture
def tparams_hparams(root_dir: pathlib.Path) -> Dict:
    full_params = OmegaConf.load(root_dir / "aintelope/config/config_experiment.yaml")

    # override some parameters during tests in order to speed up computations
    # TODO: move these overrides to a separate config?
    full_params.hparams.unit_test_mode = True
    full_params.hparams.num_episodes = min(5, full_params.hparams.num_episodes)
    full_params.hparams.env_params.num_iters = min(
        50, full_params.hparams.env_params.num_iters
    )
    full_params.hparams.warm_start_steps = min(10, full_params.hparams.warm_start_steps)

    return full_params
