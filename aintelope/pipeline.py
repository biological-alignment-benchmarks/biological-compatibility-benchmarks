import logging

import hydra

from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers
from aintelope.experiments import run_experiment

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def pipeline(cfg: DictConfig) -> None:
    pipeline_configs = OmegaConf.load("aintelope/config/config_pipeline.yaml")
    for env_conf in pipeline_configs:
        OmegaConf.update(cfg, "experiment_name", env_conf)
        OmegaConf.update(cfg, "hparams", pipeline_configs[env_conf])
        logger.info("Running training with the following configuration")
        logger.info(OmegaConf.to_yaml(cfg))
        run_experiment(cfg)


if __name__ == "__main__":
    register_resolvers()
    pipeline()
