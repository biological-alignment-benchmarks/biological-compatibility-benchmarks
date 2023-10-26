import logging
import sys

import hydra

# from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers
from aintelope.training.lightning_trainer import run_experiment

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def aintelope_main(cfg: DictConfig) -> None:  # config_file: str
    # initialize(config_path="config", job_name="test")
    # cfg = compose(config_name=config_file, overrides=[])
    logger.info("Running training with the following configuration")
    logger.info(OmegaConf.to_yaml(cfg))
    run_experiment(cfg)


if __name__ == "__main__":
    register_resolvers()
    # conf = "config_experiment.yaml"
    # if len(sys.argv) > 1:
    #    conf = sys.argv[1]
    aintelope_main()  # conf
