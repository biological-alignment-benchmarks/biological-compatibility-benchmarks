from typing import Tuple
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
def tparams_hparams(root_dir: pathlib.Path) -> Tuple[DictConfig, DictConfig]:
    full_params = OmegaConf.load(root_dir / "aintelope/config/config_experiment.yaml")
    tparams = full_params.trainer_params
    hparams = full_params.hparams
    return tparams, hparams


def root_dir_debug() -> pathlib.Path:
    return pathlib.Path(__file__).parents[1]


def tparams_hparams_debug(root_dir: pathlib.Path) -> Tuple[DictConfig, DictConfig]:
    full_params = OmegaConf.load(root_dir / "aintelope/config/config_experiment.yaml")
    tparams = full_params.trainer_params
    hparams = full_params.hparams
    return tparams, hparams
