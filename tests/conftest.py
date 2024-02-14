import pathlib
from typing import Union

import pytest
from omegaconf import DictConfig, ListConfig, OmegaConf


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
def tparams_hparams(root_dir: pathlib.Path) -> Union[DictConfig, ListConfig]:
    full_params = OmegaConf.load(root_dir / "aintelope/config/config_experiment.yaml")
    return full_params
