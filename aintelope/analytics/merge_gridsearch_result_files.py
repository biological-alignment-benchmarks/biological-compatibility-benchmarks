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


# NB! Do not run this script while calculations are running. You may end up with duplicate entries.
# Locking the output files for the duration of the entire script would not help either, since some
# duplicate entries might be still in calculation in a concurrent process and therefore would not
# be visible in the results file yet.
def merge_gridsearch_result_files() -> None:
    aggregated_results_file1 = "outputs\mixed.jsonl"
    aggregated_results_file2 = "aws_outputs\mixed_predators.jsonl"
    aggregated_results_file_out = (
        "outputs\mixed.jsonl"  # "aws_outputs\score_cooperation_merged.jsonl"
    )

    test_summaries1 = []
    if aggregated_results_file1:
        aggregated_results_file1 = os.path.normpath(aggregated_results_file1)
        print(f"Using results file: {aggregated_results_file1}")

        if os.path.exists(aggregated_results_file1):
            aggregated_results_file_lock = FileLock(aggregated_results_file1 + ".lock")
            with aggregated_results_file_lock:
                with open(aggregated_results_file1, mode="r", encoding="utf-8") as fh:
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
                    test_summaries1.append(test_summary)

                    bar.update(line_index + 1)
                # / for line_index, line in enumerate(lines):

        else:  # / if os.path.exists(aggregated_results_file):
            raise Exception("Aggregated results file 1 not found")
    else:
        raise Exception("Aggregated results file 1 not configured")

    if len(test_summaries1) == 0:
        raise Exception("Aggregated results file 1 is empty")

    test_summaries2 = []
    if aggregated_results_file2:
        aggregated_results_file2 = os.path.normpath(aggregated_results_file2)
        print(f"Using results file: {aggregated_results_file2}")

        if os.path.exists(aggregated_results_file2):
            aggregated_results_file_lock = FileLock(aggregated_results_file2 + ".lock")
            with aggregated_results_file_lock:
                with open(aggregated_results_file2, mode="r", encoding="utf-8") as fh:
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
                    test_summaries2.append(test_summary)

                    bar.update(line_index + 1)
                # / for line_index, line in enumerate(lines):

        else:  # / if os.path.exists(aggregated_results_file):
            raise Exception("Aggregated results file 2 not found")
    else:
        raise Exception("Aggregated results file 2 not configured")

    if len(test_summaries2) == 0:
        raise Exception("Aggregated results file 2 is empty")

    test_summaries_out = []
    test_configs = (
        []
    )  # hashset does not support hashing dictionaries, so need to use list. The row counts are sufficiently small that this is okay for time being.
    added_rows_count = 0
    with RobustProgressBar(max_value=len(test_summaries1), granularity=10) as bar:
        for index, test_summary in enumerate(test_summaries1):
            test_config = (
                test_summary["experiment_name"],
                test_summary["params_set_title"],
                test_summary["gridsearch_params"],
            )
            test_configs.append(test_config)
            bar.update(index + 1)

    with RobustProgressBar(max_value=len(test_summaries2), granularity=10) as bar:
        for index, test_summary in enumerate(test_summaries2):
            test_config = (
                test_summary["experiment_name"],
                test_summary["params_set_title"],
                test_summary["gridsearch_params"],
            )
            if test_config not in test_configs:
                test_summaries_out.append(test_summary)
                test_configs.append(test_config)
                added_rows_count += 1
            else:
                qqq = True  # for debugging
            bar.update(index + 1)

    print(f"\nAdded {added_rows_count} rows")

    print(f"\nWriting to {aggregated_results_file_out}")
    aggregated_results_file_lock = FileLock(aggregated_results_file_out + ".lock")
    with aggregated_results_file_lock:
        with open(
            aggregated_results_file_out, mode="a", encoding="utf-8"
        ) as fh:  # NB! append mode
            for test_summary in test_summaries_out:
                # Do not write directly to file. If JSON serialization error occurs during json.dump() then a broken line would be written into the file (I have verified this). Therefore using json.dumps() is safer.
                json_text = json.dumps(test_summary)
                fh.write(
                    json_text + "\n"
                )  # \n : Prepare the file for appending new lines upon subsequent append. The last character in the JSONL file is allowed to be a line separator, and it will be treated the same as if there was no line separator present.
            fh.flush()

    wait_for_enter("\nResults merge done. Press [enter] to continue.")
    qqq = True


# / def merge_gridsearch_result_files():


if __name__ == "__main__":
    register_resolvers()

    use_same_parameters_for_all_pipeline_experiments = False
    merge_gridsearch_result_files()
