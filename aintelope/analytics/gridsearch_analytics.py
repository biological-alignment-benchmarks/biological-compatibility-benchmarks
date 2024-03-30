import os
import copy
import json
import pandas as pd
import numpy as np
from collections import OrderedDict

import hydra
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten, unflatten
from flatten_dict.reducers import make_reducer

# this one is cross-platform
from filelock import FileLock

from aintelope.config.config_utils import register_resolvers
from aintelope.utils import wait_for_enter, try_df_to_csv_write, RobustProgressBar


# need to specify config_path since we are in a subfolder and hydra does not automatically pay attention to current working directory. By default, hydra uses the directory of current file instead.
@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "aintelope", "config"),
    config_name="config_experiment",
)
def gridsearch_analytics(cfg: DictConfig) -> None:
    # TODO: refactor into a shared method
    # TODO: automatically select correct gridsearch config file based on main cfg
    gridsearch_config_file = os.environ.get("GRIDSEARCH_CONFIG")
    # if gridsearch_config is None:
    #    gridsearch_config = "initial_config_gridsearch.yaml"
    initial_config_gridsearch = OmegaConf.load(
        os.path.join("aintelope/config", gridsearch_config_file)
    )

    OmegaConf.update(cfg, "hparams", initial_config_gridsearch.hparams, force_add=True)

    use_separate_parameters_for_each_experiment = (
        cfg.hparams.use_separate_models_for_each_experiment
    )
    use_separate_parameters_for_each_experiment = (
        False  # TODO: override function argument
    )

    # extract list parameters and compute cross product over their values
    # dict_config = OmegaConf.to_container(initial_config_gridsearch, resolve=True) # convert DictConfig to dict # NB! DO resolve references here since we DO want to handle references to lists as lists
    dict_config = OmegaConf.to_container(
        initial_config_gridsearch, resolve=False
    )  # convert DictConfig to dict # NB! do NOT resolve references here since we do NOT want to handle references to lists as lists. Gridsearch should loop over each list only once.
    flattened_config = flatten(
        dict_config, reducer=make_reducer(delimiter=".")
    )  # convert to format {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}
    list_entries = OrderedDict(
        sorted(
            {
                key: value
                for key, value in flattened_config.items()
                if isinstance(value, list)
                or value
                is None  # value is None means that a column was previously part of grid search, but has been fixed to a best value per experiment
            }.items()
        )
    )  # select only entries of list type
    list_entries[
        "hparams.gridsearch_trial_no"
    ] = (
        initial_config_gridsearch.hparams.gridsearch_trial_no
    )  # this is a resolver that generates a list

    # preserve only the config parameters that were NOT searched over
    # TODO: is this needed?
    flattened_config_without_gridsearch_keys = dict(
        flattened_config
    )  # Omegaconf does not have entry removal method, so we need to work with dict
    gridsearch_cols = set()
    non_gridsearch_cols = set()
    for key, value in list_entries.items():
        if (
            value is None or len(value) > 1
        ):  # value is None means that a column was previously part of grid search, but has been fixed to a best value per experiment
            del flattened_config_without_gridsearch_keys[key]
            gridsearch_cols.add(key)
        else:
            flattened_config_without_gridsearch_keys[key] = value[0]

    unflattened_config_without_gridsearch_keys = unflatten(
        flattened_config_without_gridsearch_keys, splitter="dot"
    )
    config_gridsearch_without_lists = OmegaConf.create(
        unflattened_config_without_gridsearch_keys
    )

    test_summaries = []
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)

        # parts = os.path.splitext(aggregated_results_file)
        # aggregated_results_file2 = parts[0] + "_recalculated" + parts[1]
        # if False and os.path.exists(aggregated_results_file2):
        #    aggregated_results_file = aggregated_results_file2
        #    print(f"Using recalculated results file: {aggregated_results_file2}")
        # else:
        print(f"Using results file: {aggregated_results_file}")

        if os.path.exists(aggregated_results_file):
            aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
            with aggregated_results_file_lock:
                with open(aggregated_results_file, mode="r", encoding="utf-8") as fh:
                    data = fh.read()

            lines = data.split("\n")
            with RobustProgressBar(
                max_value=len(lines), granularity=10
            ) as bar:  # this takes a few moments of time
                for line_index, line in enumerate(lines):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    test_summary = json.loads(line)
                    # make nested dictionaries into one level
                    flattened_test_summary = flatten(
                        test_summary, reducer=make_reducer(delimiter=".")
                    )
                    for key, value in flattened_test_summary.items():
                        if isinstance(value, list):
                            flattened_test_summary[key] = str(
                                value
                            )  # pandas cannot handle list types when they are used as groupby keys
                        elif not isinstance(value, str) and np.isnan(
                            value
                        ):  # in current setup the values can overflow only in -inf direction    # TODO: remove nan-s in the main program
                            flattened_test_summary[key] = -np.inf

                    test_summaries.append(flattened_test_summary)

                    bar.update(line_index + 1)
                # / for line_index, line in enumerate(lines):

        else:  # / if os.path.exists(aggregated_results_file):
            raise Exception("Aggregated results file not found")
    else:
        raise Exception("Aggregated results file not configured")

    if len(test_summaries) == 0:
        raise Exception("Aggregated results file is empty")

    df = pd.DataFrame.from_records(test_summaries)

    score_cols = [
        col
        for col in df.columns
        if col.startswith("test_totals.")
        or col.startswith("test_averages.")
        or col.startswith("test_variances.")
        # per score dimension results
        or col.startswith("test_sfella_totals.")
        or col.startswith("test_sfella_averages.")
        or col.startswith("test_sfella_variances.")
        # TODO: rename to test_*
        or col
        in ["sfella_score_total", "sfella_score_average", "sfella_score_variance"]
    ]

    df[score_cols] = df[score_cols].fillna(
        0
    )  # fill missing score dimensions with zeros

    # columns in file:

    # "timestamp", "experiment_name", "title", "params_set_title", "gridsearch_params", "score_dimensions", "test_totals", "test_averages", "test_variances", "test_sfella_totals", "test_sfella_averages", "test_sfella_variances"

    parameter_grouping_cols = [
        col
        for col in df.columns
        if
        # consider experiment name in the output groupings, but not necessarily during best parameter selection
        (
            col == "experiment_name"
        )  # TODO: add also params_set_title to the grouping cols
        or (
            col.startswith("gridsearch_params.")
            and not col.endswith(
                ".gridsearch_trial_no"
            )  # calculate mean over gridsearch_trial_no
        )
    ]

    gridsearch_parameter_grouping_cols = [
        col
        for col in df.columns
        if
        # consider experiment name in the output groupings, but not necessarily during best parameter selection
        (
            use_separate_parameters_for_each_experiment and col == "experiment_name"
        )  # TODO: add also params_set_title to the grouping cols
        or (
            col.startswith("gridsearch_params.")
            and not col.endswith(
                ".gridsearch_trial_no"
            )  # calculate mean over gridsearch_trial_no
        )
    ]

    gridsearch_cycle_count = 4  # max cycle count   # TODO: read from config
    eval_cycle_count = 25  # gridsearch_cycle_count if gridsearch_cycle_count is not None else 25    # min cycle count       # TODO: read from config

    df = df[
        df["gridsearch_params.hparams.gridsearch_trial_no"] < eval_cycle_count
    ]  # ignore results past eval_cycle_count
    # df = df[df["gridsearch_params.hparams.model_params.conv_size"] == 3]

    params_and_results_cols = parameter_grouping_cols + score_cols
    df_params_and_results = df[params_and_results_cols]

    if gridsearch_cycle_count is not None:  # keep only rows up to given trial no
        gridsearch_df = df[
            df["gridsearch_params.hparams.gridsearch_trial_no"] < gridsearch_cycle_count
        ]
        gridsearch_df_params_and_results = gridsearch_df[
            gridsearch_parameter_grouping_cols
            + ["test_averages.Score", "sfella_score_average"]
        ]
    else:
        gridsearch_df_params_and_results = df[
            gridsearch_parameter_grouping_cols
            + ["test_averages.Score", "sfella_score_average"]
        ]

    score_cols_mean_renames = {col: "mean." + col for col in score_cols}
    score_cols_std_renames = {col: "std." + col for col in score_cols}

    # TODO: refactor this into a separate helper function since this code does not seem to be quite straightforward, but is a general use case
    # preserve the groupby cols after aggregation by mean(). See https://stackoverflow.com/questions/40139184/keeping-key-column-when-using-groupby-with-transform-in-pandas
    averages_per_parameter_combination = pd.concat(
        [
            df_params_and_results[
                parameter_grouping_cols
            ],  # returns parameter_grouping_cols
            df_params_and_results.groupby(parameter_grouping_cols)
            .transform("size")  # count rows per group  # returns averaged score_cols
            .rename("count", inplace=False),
            df_params_and_results.groupby(parameter_grouping_cols)
            .transform("mean")  # returns averaged score_cols
            .rename(columns=score_cols_mean_renames, inplace=False),
            df_params_and_results.groupby(parameter_grouping_cols)
            .transform("std", ddof=0)  # TODO: ddof  # returns averaged score_cols
            .rename(columns=score_cols_std_renames, inplace=False),
        ],
        axis=1,
    ).drop_duplicates()
    averages_per_parameter_combination["score_mean_minus_std"] = (
        averages_per_parameter_combination["mean.test_averages.Score"]
        - averages_per_parameter_combination["std.test_averages.Score"]
    )

    gridsearch_averages_per_parameter_combination = pd.concat(
        [
            gridsearch_df_params_and_results[
                gridsearch_parameter_grouping_cols
            ],  # returns gridsearch_parameter_grouping_cols
            gridsearch_df_params_and_results.groupby(gridsearch_parameter_grouping_cols)
            .transform("size")  # count rows per group  # returns averaged score_cols
            .rename("count", inplace=False),
            gridsearch_df_params_and_results.groupby(gridsearch_parameter_grouping_cols)
            .transform("mean")  # returns averaged score_cols
            .rename(columns=score_cols_mean_renames, inplace=False),
            gridsearch_df_params_and_results.groupby(gridsearch_parameter_grouping_cols)
            .transform("std", ddof=0)  # TODO: ddof  # returns averaged score_cols
            .rename(columns=score_cols_std_renames, inplace=False),
        ],
        axis=1,
    ).drop_duplicates()
    gridsearch_averages_per_parameter_combination["score_mean_minus_std"] = (
        gridsearch_averages_per_parameter_combination["mean.test_averages.Score"]
        - gridsearch_averages_per_parameter_combination["std.test_averages.Score"]
    )

    # keep only parameter combinations with sufficient cycle count
    averages_per_parameter_combination = averages_per_parameter_combination[
        averages_per_parameter_combination["count"] >= eval_cycle_count
    ]
    gridsearch_averages_per_parameter_combination = (
        gridsearch_averages_per_parameter_combination[
            gridsearch_averages_per_parameter_combination["count"] >= eval_cycle_count
        ]
    )

    # group by columns: hparams, map_max, experiment_name

    # TODO: in case multiple rows have same best value, take all of them

    environment_grouping_dims = ["gridsearch_params.hparams.env_params.map_max"]
    environment_grouping_dims.append("experiment_name")

    gridsearch_environment_grouping_dims = [
        "gridsearch_params.hparams.env_params.map_max"
    ]
    if use_separate_parameters_for_each_experiment:
        gridsearch_environment_grouping_dims.append("experiment_name")

    test = (
        gridsearch_averages_per_parameter_combination.groupby(
            gridsearch_environment_grouping_dims
        )
        .transform("max")
        .drop_duplicates()
    )

    # TODO: index by parameters, not by row index
    gridsearch_best_parameters_by_score_row_indexes = (
        gridsearch_averages_per_parameter_combination.groupby(
            gridsearch_environment_grouping_dims
        )["mean.test_averages.Score"].idxmax()
    )
    gridsearch_best_parameters_by_score = (
        gridsearch_averages_per_parameter_combination.loc[
            gridsearch_best_parameters_by_score_row_indexes
        ]
    )

    # select rows from full dataset which match the gridsearch best parameters
    rows = []
    with RobustProgressBar(
        max_value=len(averages_per_parameter_combination), granularity=10
    ) as bar:
        for index, (pd_index, row) in enumerate(
            averages_per_parameter_combination.iterrows()
        ):
            mask = (
                gridsearch_best_parameters_by_score[gridsearch_parameter_grouping_cols]
                == row[gridsearch_parameter_grouping_cols]
            ).all(
                axis=1
            )  # all() over columns
            if mask.sum() > 0:
                assert mask.sum() == 1
                # assert row["mean.test_averages.Score"] == gridsearch_best_parameters_by_score[mask]["mean.test_averages.Score"].item()
                rows.append(row)
            bar.update(index + 1)
    if use_separate_parameters_for_each_experiment:
        assert len(rows) == len(gridsearch_best_parameters_by_score)
    best_parameters_by_score = pd.concat(rows, axis=1).transpose()
    # best_parameters_by_score = averages_per_parameter_combination.loc[best_parameters_by_score_row_indexes]
    # assert list(test["mean.test_averages.Score"]) == list(best_parameters_by_score["mean.test_averages.Score"])

    gridsearch_best_parameters_by_sfella_score_row_indexes = (
        gridsearch_averages_per_parameter_combination.groupby(
            gridsearch_environment_grouping_dims
        )["mean.sfella_score_average"].idxmax()
    )
    gridsearch_best_parameters_by_sfella_score = (
        gridsearch_averages_per_parameter_combination.loc[
            gridsearch_best_parameters_by_sfella_score_row_indexes
        ]
    )

    # select rows from full dataset which match the gridsearch best parameters
    rows = []
    with RobustProgressBar(
        max_value=len(averages_per_parameter_combination), granularity=10
    ) as bar:
        for index, (pd_index, row) in enumerate(
            averages_per_parameter_combination.iterrows()
        ):
            mask = (
                gridsearch_best_parameters_by_sfella_score[
                    gridsearch_parameter_grouping_cols
                ]
                == row[gridsearch_parameter_grouping_cols]
            ).all(
                axis=1
            )  # all() over columns
            if mask.sum() > 0:
                assert mask.sum() == 1
                # assert row["mean.sfella_score_average"] == gridsearch_best_parameters_by_sfella_score[mask]["mean.sfella_score_average"].item()
                rows.append(row)
            bar.update(index + 1)
    if use_separate_parameters_for_each_experiment:
        assert len(rows) == len(gridsearch_best_parameters_by_sfella_score)
    best_parameters_by_sfella_score = pd.concat(rows, axis=1).transpose()
    # best_parameters_by_sfella_score = averages_per_parameter_combination.loc[best_parameters_by_sfella_score_row_indexes]

    # if best_parameters_by_sfella_score contain -inf values then ignore these rows
    infinities = best_parameters_by_sfella_score["mean.sfella_score_average"] == -np.inf
    best_parameters_by_sfella_score = best_parameters_by_sfella_score.drop(
        best_parameters_by_sfella_score[infinities].index
    )
    # assert list(test["mean.sfella_score_average"]) == list(best_parameters_by_sfella_score["mean.sfella_score_average"])

    gridsearch_best_parameters_by_score_minus_std_row_indexes = (
        gridsearch_averages_per_parameter_combination.groupby(
            gridsearch_environment_grouping_dims
        )["score_mean_minus_std"].idxmax()
    )
    gridsearch_best_parameters_by_score_minus_std = (
        gridsearch_averages_per_parameter_combination.loc[
            gridsearch_best_parameters_by_score_minus_std_row_indexes
        ]
    )

    # select rows from full dataset which match the gridsearch best parameters
    rows = []
    with RobustProgressBar(
        max_value=len(averages_per_parameter_combination), granularity=10
    ) as bar:
        for index, (pd_index, row) in enumerate(
            averages_per_parameter_combination.iterrows()
        ):
            mask = (
                gridsearch_best_parameters_by_score_minus_std[
                    gridsearch_parameter_grouping_cols
                ]
                == row[gridsearch_parameter_grouping_cols]
            ).all(
                axis=1
            )  # all() over columns
            if mask.sum() > 0:
                assert mask.sum() == 1
                # assert row["mean.sfella_score_average"] == gridsearch_best_parameters_by_sfella_score[mask]["mean.sfella_score_average"].item()
                rows.append(row)
            bar.update(index + 1)
    if use_separate_parameters_for_each_experiment:
        assert len(rows) == len(gridsearch_best_parameters_by_score_minus_std)
    best_parameters_by_score_minus_std = pd.concat(rows, axis=1).transpose()
    # best_parameters_by_score_minus_std = averages_per_parameter_combination.loc[best_parameters_by_score_minus_std_row_indexes]
    # assert list(test["score_mean_minus_std"]) == list(best_parameters_by_score_minus_std_row_indexes["score_mean_minus_std"])

    total_score_dims = [
        "mean.test_averages.Score",
        "std.test_averages.Score",
        "mean.sfella_score_average",
        "std.sfella_score_average",
        "score_mean_minus_std",
    ]
    reward_dims = [
        "mean.test_averages.Reward",
        "std.test_averages.Reward",
        "mean.test_sfella_averages.Reward",
        "std.test_sfella_averages.Reward",
    ]
    average_score_subdims = [
        col
        for col in averages_per_parameter_combination.columns
        if (
            col.startswith("mean.test_averages.") or col == "score_mean_minus_std"
        )  # or col.startswith("std.test_averages."))
        and col not in total_score_dims
        and col not in reward_dims
    ]
    sfella_average_score_subdims = [
        col
        for col in averages_per_parameter_combination.columns
        if (
            col.startswith("mean.test_sfella_averages.")
        )  # or col.startswith("std.test_sfella_averages."))
        and col not in total_score_dims
        and col not in reward_dims
    ]

    gridsearch_cols = list(gridsearch_cols)
    gridsearch_cols.sort()
    gridsearch_cols = [
        "gridsearch_params." + col
        for col in gridsearch_cols
        if not col.endswith(".gridsearch_trial_no")  # this one is averaged out
        and ("gridsearch_params." + col)
        not in environment_grouping_dims  # map_max, experiment_name
    ]

    non_gridsearch_cols = list(flattened_config_without_gridsearch_keys.keys())
    non_gridsearch_cols.sort()
    non_gridsearch_cols = [
        "gridsearch_params." + col
        for col in non_gridsearch_cols
        if ("gridsearch_params." + col)
        not in environment_grouping_dims  # exclude map_max, experiment_name here as well, even if they happen to be not part of gridsearch, else we get duplicate columns below
    ]

    best_parameters_by_score = best_parameters_by_score[
        environment_grouping_dims
        + gridsearch_cols
        + ["count"]
        + total_score_dims
        + reward_dims
        + sfella_average_score_subdims
        + non_gridsearch_cols
    ]

    best_parameters_by_sfella_score = best_parameters_by_sfella_score[
        environment_grouping_dims
        + gridsearch_cols
        + ["count"]
        + total_score_dims
        + reward_dims
        + sfella_average_score_subdims
        + non_gridsearch_cols
    ]

    best_parameters_by_score_minus_std = best_parameters_by_score_minus_std[
        environment_grouping_dims
        + gridsearch_cols
        + ["count"]
        + total_score_dims
        + reward_dims
        + sfella_average_score_subdims
        + non_gridsearch_cols
    ]

    best_parameters_by_score = best_parameters_by_score.sort_values(
        by=environment_grouping_dims
    )
    best_parameters_by_sfella_score = best_parameters_by_sfella_score.sort_values(
        by=environment_grouping_dims
    )
    best_parameters_by_score_minus_std = best_parameters_by_score_minus_std.sort_values(
        by=environment_grouping_dims
    )

    if (
        True
    ):  # keep only config parameter names, remove their path to make column names shorter in the CSV file   # TODO: config option
        renames = {
            col: col.split(".")[-1]
            for col in best_parameters_by_score.columns
            if col.startswith("gridsearch_params.")
        }
        best_parameters_by_score_shortened = best_parameters_by_score.rename(
            columns=renames, inplace=False
        )

        renames = {
            col: col.split(".")[-1]
            for col in best_parameters_by_sfella_score.columns
            if col.startswith("gridsearch_params.")
        }
        best_parameters_by_sfella_score_shortened = (
            best_parameters_by_sfella_score.rename(columns=renames, inplace=False)
        )
        best_parameters_by_score_minus_std_shortened = (
            best_parameters_by_score_minus_std.rename(columns=renames, inplace=False)
        )
    else:
        best_parameters_by_score_shortened = best_parameters_by_score
        best_parameters_by_sfella_score_shortened = best_parameters_by_sfella_score
        best_parameters_by_score_minus_std_shortened = (
            best_parameters_by_score_minus_std
        )

    filepath = cfg.hparams.params_set_title + "_best_parameters_by_score.csv"
    print(f"\nWriting to {filepath}")
    try_df_to_csv_write(
        best_parameters_by_score_shortened, filepath, index=False, mode="w", header=True
    )

    filepath = cfg.hparams.params_set_title + "_best_parameters_by_sfella_score.csv"
    print(f"\nWriting to {filepath}")
    try_df_to_csv_write(
        best_parameters_by_sfella_score_shortened,
        filepath,
        index=False,
        mode="w",
        header=True,
    )

    filepath = cfg.hparams.params_set_title + "_best_parameters_by_score_minus_std.csv"
    print(f"\nWriting to {filepath}")
    try_df_to_csv_write(
        best_parameters_by_score_minus_std_shortened,
        filepath,
        index=False,
        mode="w",
        header=True,
    )

    # create a new specialised pipeline config file based on the best parameters per experiment

    # TODO: ensure to do not use special pipeline config when doing initial gridsearch
    pipeline_config_file = os.environ.get("PIPELINE_CONFIG")
    if pipeline_config_file is None:
        pipeline_config_file = "config_pipeline.yaml"
        parts = os.path.splitext(pipeline_config_file)
        specialised_pipeline_config_file = (
            parts[0]
            + "_"
            + cfg.hparams.params_set_title
            + ("_common" if not use_separate_parameters_for_each_experiment else "")
            + parts[1]
        )
    else:
        specialised_pipeline_config_file = pipeline_config_file
    pipeline_config = OmegaConf.load(
        os.path.join("aintelope/config", pipeline_config_file)
    )

    specialised_config_params_source = best_parameters_by_score  # TODO: config for choosing between best_parameters_by_score, best_parameters_by_sfella_score, best_parameters_by_score_minus_std

    specialised_pipeline_config = copy.deepcopy(pipeline_config)
    for map_size in [
        7
    ]:  # TODO: config, or select all available map sizes from gridsearch config
        for env_conf_name in pipeline_config:
            result_row_selector = (
                specialised_config_params_source[
                    "gridsearch_params.hparams.env_params.map_max"
                ]
                == map_size
            )
            if "experiment_name" in parameter_grouping_cols:
                result_row_selector &= (
                    specialised_config_params_source["experiment_name"] == env_conf_name
                )

            result_row = specialised_config_params_source[result_row_selector]

            if (
                len(result_row) > 0
            ):  # check whether the grid search produced data for this map_max and experiment_name pair
                key = env_conf_name + ".env_params.map_max"
                value = map_size
                OmegaConf.update(
                    specialised_pipeline_config, key, value, force_add=True
                )

                for gridsearch_col in gridsearch_cols:
                    key = (
                        env_conf_name
                        + "."
                        + gridsearch_col[len("gridsearch_params.hparams.") :]
                    )
                    value = result_row[gridsearch_col].item()
                    OmegaConf.update(
                        specialised_pipeline_config, key, value, force_add=True
                    )

    # TODO: confirm overwriting existing file
    # print(f"\nWriting to {specialised_pipeline_config_file}")
    # OmegaConf.save(
    #    specialised_pipeline_config,
    #    os.path.join("aintelope/config", specialised_pipeline_config_file),
    #    resolve=False,
    # )

    wait_for_enter("\nAnalytics done. Press [enter] to continue.")
    qqq = True


# / def gridsearch_analytics():


if __name__ == "__main__":
    register_resolvers()

    use_same_parameters_for_all_pipeline_experiments = False
    gridsearch_analytics()
