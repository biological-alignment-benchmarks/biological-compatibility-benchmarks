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

from aintelope.config.config_utils import register_resolvers
from aintelope.utils import wait_for_enter, try_df_to_csv_write, RobustProgressBar


# need to specify config_path since we are in a subfolder and hydra does not automatically pay attention to current working directory. By default, hydra uses the directory of current file instead.
@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "aintelope", "config"),
    config_name="config_experiment",
)
def remove_trials_from_results(cfg: DictConfig) -> None:
    # TODO: refactor into a shared method
    # TODO: automatically select correct gridsearch config file based on main cfg
    gridsearch_config_file = os.environ.get("GRIDSEARCH_CONFIG")
    # if gridsearch_config is None:
    #    gridsearch_config = "initial_config_gridsearch.yaml"
    initial_config_gridsearch = OmegaConf.load(
        os.path.join("aintelope/config", gridsearch_config_file)
    )

    OmegaConf.update(cfg, "hparams", initial_config_gridsearch.hparams, force_add=True)

    lines_out = []
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)

        parts = os.path.splitext(aggregated_results_file)
        aggregated_results_file2 = parts[0] + "_recalculated" + parts[1]
        if False and os.path.exists(aggregated_results_file2):
            aggregated_results_file = aggregated_results_file2
            print(f"Using recalculated results file: {aggregated_results_file2}")
        else:
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

                    trial_no = flattened_test_summary[
                        "gridsearch_params.hparams.gridsearch_trial_no"
                    ]
                    if trial_no < 4:  # TODO: function argument
                        lines_out.append(line)

                    bar.update(line_index + 1)
                # / for line_index, line in enumerate(lines):

        else:  # / if os.path.exists(aggregated_results_file):
            raise Exception("Aggregated results file not found")
    else:
        raise Exception("Aggregated results file not configured")

    aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
    with aggregated_results_file_lock:
        with open(aggregated_results_file, mode="w", encoding="utf-8") as fh:
            for line in lines_out:
                fh.write(
                    line + "\n"
                )  # \n : Prepare the file for appending new lines upon subsequent append. The last character in the JSONL file is allowed to be a line separator, and it will be treated the same as if there was no line separator present.
            fh.flush()

    wait_for_enter("\nTrials removal done. Press [enter] to continue.")
    qqq = True


# / def remove_trials_from_results():


if __name__ == "__main__":
    register_resolvers()

    use_same_parameters_for_all_pipeline_experiments = False
    remove_trials_from_results()
