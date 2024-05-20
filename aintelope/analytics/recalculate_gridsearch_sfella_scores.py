import os
import copy
import json
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten, unflatten
from flatten_dict.reducers import make_reducer

# this one is cross-platform
from filelock import FileLock

from aintelope.analytics import plotting, recording
from aintelope.config.config_utils import register_resolvers, get_score_dimensions
from aintelope.utils import wait_for_enter, try_df_to_csv_write, RobustProgressBar


# need to specify config_path since we are in a subfolder and hydra does not automatically pay attention to current working directory. By default, hydra uses the directory of current file instead.
@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "aintelope", "config"),
    config_name="config_experiment",
)
def recalculate_gridsearch_sfella_scores(cfg: DictConfig) -> None:
    # TODO: refactor into a shared method
    # TODO: automatically select correct gridsearch config file based on main cfg
    gridsearch_config_file = os.environ.get("GRIDSEARCH_CONFIG")
    # if gridsearch_config is None:
    #    gridsearch_config = "initial_config_gridsearch.yaml"
    initial_config_gridsearch = OmegaConf.load(
        os.path.join("aintelope/config", gridsearch_config_file)
    )

    OmegaConf.update(cfg, "hparams", initial_config_gridsearch.hparams, force_add=True)

    test_summaries = []
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)
        if os.path.exists(aggregated_results_file):
            aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
            with aggregated_results_file_lock:
                with open(aggregated_results_file, mode="r", encoding="utf-8") as fh:
                    data = fh.read()

            # reducer = make_reducer(delimiter='.')

            lines = data.split("\n")
            with RobustProgressBar(
                max_value=len(lines), granularity=10
            ) as bar:  # this takes a few moments of time
                for line_index, line in enumerate(lines):
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    test_summary = json.loads(line)
                    test_summaries.append(test_summary)

                    bar.update(line_index + 1)
                # / for line_index, line in enumerate(lines):

        else:  # / if os.path.exists(aggregated_results_file):
            raise Exception("Aggregated results file not found")
    else:
        raise Exception("Aggregated results file not configured")

    if len(test_summaries) == 0:
        raise Exception("Aggregated results file is empty")

    # test_summaries = test_summaries[345:]   # for debugging

    log_dir_root = os.path.normpath(cfg["log_dir_root"])

    recalculated_test_summaries = []
    with RobustProgressBar(max_value=len(test_summaries)) as bar:
        for index, test_summary in enumerate(test_summaries):
            if "timestamp_pid_uuid" in test_summary:
                timestamp_pid_uuid = test_summary["timestamp_pid_uuid"]
            else:  # compatibility with gridsearch old data. # TODO: remove this branch later.
                timestamp_pid_uuid = test_summary["timestamp"]

            experiment_name = test_summary["experiment_name"]

            log_dir = os.path.join(
                log_dir_root, timestamp_pid_uuid
            )  # TODO: read outputs folder name from configuration
            experiment_dir = os.path.join(log_dir, experiment_name)
            events_fname = cfg.events_fname
            events = recording.read_events(experiment_dir, events_fname)
            if (
                len(events) == 0
            ):  # removed log folder, lets assume we do not want this data in the output file
                print(
                    f"\nSkipping {experiment_dir} : {events_fname} since it is removed or empty"
                )
                print(test_summary)
                continue

            # here gridsearch_params includes pipeline config for experiment
            gridsearch_params = OmegaConf.create(test_summary["gridsearch_params"])
            experiment_cfg = copy.deepcopy(
                cfg
            )  # need to deepcopy in order to not accumulate keys that were present in previous experiment and are not present in next experiment
            OmegaConf.update(experiment_cfg, "experiment_name", experiment_name)
            OmegaConf.update(  # here gridsearch_params includes pipeline config for experiment
                experiment_cfg, "hparams", gridsearch_params, force_add=True
            )

            num_train_pipeline_cycles = experiment_cfg.hparams.num_pipeline_cycles
            # score_dimensions = get_score_dimensions(experiment_cfg)
            score_dimensions = test_summary["score_dimensions"]
            score_dimensions.remove("Score")
            score_dimensions.remove("Reward")
            group_by_pipeline_cycle = cfg.hparams.num_pipeline_cycles >= 1

            (
                test_totals,
                test_averages,
                test_variances,
                test_sfella_totals,
                test_sfella_averages,
                test_sfella_variances,
                sfella_score_total,
                sfella_score_average,
                sfella_score_variance,
                score_dimensions_out,
            ) = plotting.aggregate_scores(
                events,
                num_train_pipeline_cycles,
                score_dimensions,
                group_by_pipeline_cycle=group_by_pipeline_cycle,
            )

            recalculated_test_summary = {
                "timestamp": test_summary["timestamp"],
                "timestamp_pid_uuid": timestamp_pid_uuid,
                "experiment_name": experiment_name,
                "title": test_summary["title"],
                "params_set_title": test_summary["params_set_title"],
                "gridsearch_params": OmegaConf.to_container(
                    gridsearch_params, resolve=True
                )
                if gridsearch_params is not None
                else None,  # Object of type DictConfig is not JSON serializable, neither can yaml.dump in plotting.prettyprint digest it, so need to convert it to ordinary dictionary
                "num_train_pipeline_cycles": num_train_pipeline_cycles,
                "score_dimensions": score_dimensions_out,
                "group_by_pipeline_cycle": group_by_pipeline_cycle,
                "test_totals": test_totals,
                "test_averages": test_averages,
                "test_variances": test_variances,
                # per score dimension results
                "test_sfella_totals": test_sfella_totals,
                "test_sfella_averages": test_sfella_averages,
                "test_sfella_variances": test_sfella_variances,
                # over score dimensions results
                # TODO: rename to test_*
                "sfella_score_total": sfella_score_total,
                "sfella_score_average": sfella_score_average,
                "sfella_score_variance": sfella_score_variance,
            }

            recalculated_test_summaries.append(recalculated_test_summary)
            bar.update(index + 1)

        # / for test_summary in test_summaries:

    # / with RobustProgressBar(max_value=len(test_summaries)) as bar:

    # parts = os.path.splitext(aggregated_results_file)
    # aggregated_results_file2 = parts[0] + "_recalculated" + parts[1]
    aggregated_results_file2 = aggregated_results_file
    print(f"\nWriting to {aggregated_results_file2}")
    aggregated_results_file2_lock = FileLock(aggregated_results_file2 + ".lock")
    with aggregated_results_file2_lock:
        with open(aggregated_results_file2, mode="w", encoding="utf-8") as fh:
            for recalculated_test_summary in recalculated_test_summaries:
                # Do not write directly to file. If JSON serialization error occurs during json.dump() then a broken line would be written into the file (I have verified this). Therefore using json.dumps() is safer.
                json_text = json.dumps(recalculated_test_summary)
                fh.write(
                    json_text + "\n"
                )  # \n : Prepare the file for appending new lines upon subsequent append. The last character in the JSONL file is allowed to be a line separator, and it will be treated the same as if there was no line separator present.
            fh.flush()

    wait_for_enter("\nRecalculation done. Press [enter] to continue.")
    qqq = True


# / def gridsearch_analytics():


if __name__ == "__main__":
    register_resolvers()

    use_same_parameters_for_all_pipeline_experiments = False
    recalculate_gridsearch_sfella_scores()
