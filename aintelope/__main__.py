import copy
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import (
    get_pipeline_score_dimensions,
    register_resolvers,
)
from aintelope.experiments import run_experiment

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def aintelope_main(cfg: DictConfig) -> None:
  
    archive_code(cfg)

    pipeline_config = OmegaConf.load("aintelope/config/config_pipeline.yaml")
    score_dimensions = get_pipeline_score_dimensions(cfg, pipeline_config)
    for env_conf in pipeline_config:
        experiment_cfg = copy.deepcopy(
            cfg
        )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
        OmegaConf.update(experiment_cfg, "experiment_name", env_conf)
        OmegaConf.update(
            experiment_cfg, "hparams", pipeline_config[env_conf], force_add=True
        )
        logger.info("Running training with the following configuration")
        logger.info(OmegaConf.to_yaml(experiment_cfg))
		title = env_conf

        # Training
        OmegaConf.update(experiment_cfg, "hparams.traintest_mode", "train")
        run_experiment(experiment_cfg, score_dimensions)
        analytics(experiment_cfg, score_dimensions, title=title)

        # Testing
        OmegaConf.update(experiment_cfg, "hparams.traintest_mode", "test")
        run_experiment(experiment_cfg, score_dimensions)
        analytics(experiment_cfg, score_dimensions, title=title)


def analytics(cfg, score_dimensions):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    experiment_dir = os.path.normpath(cfg.experiment_dir)
    events_fname = cfg.events_fname

    savepath = os.path.join(experiment_dir, cfg.hparams.traintest_mode + "_plot.png")
    events = recording.read_events(experiment_dir, events_fname)
    plotting.plot_performance(events, score_dimensions, savepath)


if __name__ == "__main__":
    register_resolvers()

    set_priorities()
    set_memory_limits() 
    select_gpu()

    aintelope_main()
