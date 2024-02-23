import copy
from ast import literal_eval
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def get_project_path(path_from_root: str) -> Path:
    project_root = Path(__file__).parents[2]
    return project_root / path_from_root


def register_resolvers() -> None:
    OmegaConf.register_resolver("abs_path", get_project_path)


def get_score_dimensions(cfg: DictConfig):
    scores = cfg.hparams.env_params.scores
    dimensions = set()
    for event_name, score_dims_dict in scores.items():
        score_dims_dict = literal_eval(score_dims_dict)
        for dimension, value in score_dims_dict.items():
            if value != 0:  # ignore zero valued score dimensions
                dimensions.add(dimension)
    dimensions = list(dimensions)
    dimensions.sort()
    return dimensions


def get_pipeline_score_dimensions(cfg: DictConfig, pipeline_config: DictConfig):
    dimensions = set()
    for env_conf in pipeline_config:
        experiment_cfg = copy.deepcopy(
            cfg
        )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
        OmegaConf.update(  # need to merge configs here too since dimensions inside scores are not merged, but instead overwritten by experiment config. If main config has some score dimension that experiment does not have, then then that score dimension should not be used
            experiment_cfg, "hparams", pipeline_config[env_conf], force_add=True
        )
        experiment_score_dimensions = get_score_dimensions(experiment_cfg)
        for dimension in experiment_score_dimensions:
            dimensions.add(dimension)

    dimensions = list(dimensions)
    dimensions.sort()
    return dimensions
