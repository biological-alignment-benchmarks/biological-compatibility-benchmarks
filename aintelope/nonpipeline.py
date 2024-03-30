import logging
import os
import sys
import torch
import gc

import hydra
from omegaconf import DictConfig, OmegaConf

from aintelope.config.config_utils import register_resolvers, get_score_dimensions
from aintelope.experiments import run_experiment

from aintelope.analytics import plotting, recording

logger = logging.getLogger("aintelope.__main__")


@hydra.main(version_base=None, config_path="config", config_name="config_experiment")
def aintelope_main(cfg: DictConfig) -> None:
    timestamp = str(cfg.timestamp)
    logger.info(f"timestamp: {timestamp}")

    logger.info("Running training with the following configuration")
    logger.info(OmegaConf.to_yaml(cfg))
    score_dimensions = get_score_dimensions(cfg)
    run_experiment(
        cfg,
        experiment_name="Nonpipeline",
        score_dimensions=score_dimensions,
        is_last_pipeline_cycle=False,
        i_pipeline_cycle=0,
    )

    title = timestamp + " : Nonpipeline"
    do_not_show_plot = cfg.hparams.unit_test_mode
    analytics(
        cfg,
        score_dimensions,
        title=title,
        experiment_name="Nonpipeline",
        do_not_show_plot=do_not_show_plot,
    )

    if not do_not_show_plot:
        # keep plots visible until the user decides to close the program
        if os.name == "nt":
            import msvcrt

            print("Press [enter] to continue.")
            msvcrt.getch()  # uses less CPU on Windows than input() function. Note that the graph window will be frozen, but will still show graphs
        else:
            input("Press [enter] to continue.")


def analytics(cfg, score_dimensions, title, experiment_name, do_not_show_plot=False):
    # normalise slashes in paths. This is not mandatory, but will be cleaner to debug
    log_dir = os.path.normpath(cfg.log_dir)
    events_fname = cfg.events_fname
    num_train_episodes = cfg.hparams.num_episodes
    num_train_pipeline_cycles = cfg.hparams.num_pipeline_cycles

    savepath = os.path.join(log_dir, "plot_" + experiment_name + ".png")
    events = recording.read_events(log_dir, events_fname)

    torch.cuda.empty_cache()
    gc.collect()

    plotting.plot_performance(
        events,
        num_train_episodes,
        num_train_pipeline_cycles,
        score_dimensions,
        save_path=savepath,
        title=title,
        group_by_pipeline_cycle=False,
        do_not_show_plot=do_not_show_plot,
    )


if __name__ == "__main__":
    register_resolvers()
    aintelope_main()
