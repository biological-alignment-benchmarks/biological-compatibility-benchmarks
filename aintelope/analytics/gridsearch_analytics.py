import os
import copy
import json
import pandas as pd
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
from flatten_dict import flatten, unflatten
from flatten_dict.reducers import make_reducer

# import mutex. This one is cross-platform
from filelock import FileLock

from aintelope.config.config_utils import register_resolvers

from progressbar import ProgressBar


# need to specify config_path since we are in a subfolder and hydra does not automatically pay attention to current working directory. By default, hydra uses the directory of current file instead.
@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), "aintelope", "config"), config_name="config_experiment")
def gridsearch_analytics(cfg: DictConfig) -> None: 


    # TODO: refactor into a shared method
    # TODO: automatically select correct gridsearch config file based on main cfg
    gridsearch_config_file = os.environ.get("GRIDSEARCH_CONFIG")
    #if gridsearch_config is None:
    #    gridsearch_config = "aintelope/config/config_gridsearch.yaml"
    config_gridsearch = OmegaConf.load(gridsearch_config_file)

    OmegaConf.update(
        cfg, "hparams", config_gridsearch.hparams, force_add=True
    )

    use_separate_models_for_each_experiment = cfg.hparams.use_separate_models_for_each_experiment


    # extract list parameters and compute cross product over their values
    # dict_config = OmegaConf.to_container(config_gridsearch, resolve=True) # convert DictConfig to dict # NB! DO resolve references here since we DO want to handle references to lists as lists
    dict_config = OmegaConf.to_container(config_gridsearch, resolve=False) # convert DictConfig to dict # NB! do NOT resolve references here since we do NOT want to handle references to lists as lists. Gridsearch should loop over each list only once.
    flattened_config = flatten(dict_config, reducer=make_reducer(delimiter='.'))    # convert to format {'a': 1, 'c.a': 2, 'c.b.x': 5, 'c.b.y': 10, 'd': [1, 2, 3]}
    list_entries = {key: value for key, value in flattened_config.items() if isinstance(value, list)}   # select only entries of list type
    list_entries["hparams.gridsearch_trial_no"] = config_gridsearch.hparams.gridsearch_trial_no   # this is a resolver that generates a list


    # preserve only the config parameters that were NOT searched over
    # TODO: is this needed?
    flattened_config_without_gridsearch_keys = dict(flattened_config)    # Omegaconf does not have entry removal method, so we need to work with dict
    gridsearch_cols = set()
    non_gridsearch_cols = set()
    for key, value in list_entries.items():
        if len(value) > 1:
            del flattened_config_without_gridsearch_keys[key]
            gridsearch_cols.add(key)
        else:
            flattened_config_without_gridsearch_keys[key] = value[0]

    unflattened_config_without_gridsearch_keys = unflatten(flattened_config_without_gridsearch_keys, splitter='dot')
    config_gridsearch_without_lists = OmegaConf.create(unflattened_config_without_gridsearch_keys)


    test_summaries = []
    if cfg.hparams.aggregated_results_file:
        aggregated_results_file = os.path.normpath(cfg.hparams.aggregated_results_file)

        parts = os.path.splitext(aggregated_results_file)
        aggregated_results_file2 = parts[0] + "_recalculated" + parts[1]
        if os.path.exists(aggregated_results_file2):
            aggregated_results_file = aggregated_results_file2
            print(f"Using recalculated results file: {aggregated_results_file2}")
        else:
            print(f"Using results file: {aggregated_results_file}")

        if os.path.exists(aggregated_results_file):
            aggregated_results_file_lock = FileLock(aggregated_results_file + ".lock")
            with aggregated_results_file_lock:                                    
                with open(aggregated_results_file, mode='r', encoding='utf-8') as fh:
                    data = fh.read()

            lines = data.split("\n")
            with ProgressBar(max_value=len(lines)) as bar:    # this takes a few moments of time
                for line_index, line in enumerate(lines): 
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    test_summary = json.loads(line)
                    # make nested dictionaries into one level
                    flattened_test_summary = flatten(test_summary, reducer=make_reducer(delimiter='.'))
                    for key, value in flattened_test_summary.items():
                        if isinstance(value, list):
                            flattened_test_summary[key] = str(value)    # pandas cannot handle list types when they are used as groupby keys
                        elif not isinstance(value, str) and np.isnan(value):   # in current setup the values can overflow only in -inf direction    # TODO: remove nan-s in the main program
                            flattened_test_summary[key] = -np.inf

                    test_summaries.append(flattened_test_summary)

                    if (line_index + 1) % 10 == 0:
                        bar.update(line_index + 1)
                #/ for line_index, line in enumerate(lines):
                bar.update(len(lines))

        else: #/ if os.path.exists(aggregated_results_file):
            raise Exception("Aggregated results file not found")
    else:
        raise Exception("Aggregated results file not configured")

    if len(test_summaries) == 0:
        raise Exception("Aggregated results file is empty")

    df = pd.DataFrame.from_records(test_summaries)


    score_cols = [
        col for col in df.columns 
        if 

        col.startswith("test_totals.")
        or col.startswith("test_averages.")
        or col.startswith("test_variances.")

        # per score dimension results
        or col.startswith("test_sfella_totals.")
        or col.startswith("test_sfella_averages.")
        or col.startswith("test_sfella_variances.")

        # TODO: rename to test_*
        or col in ["sfella_score_total", "sfella_score_average", "sfella_score_variance"]        
    ]

    df[score_cols] = df[score_cols].fillna(0)   # fill missing score dimensions with zeros


    groupby_cols = [
        col for col in df.columns 
        if 
        (use_separate_models_for_each_experiment and col == "experiment_name")
        or (
            col.startswith("gridsearch_params.") 
            and not col.endswith(".gridsearch_trial_no")    # calculate mean over gridsearch_trial_no
        )     
    ]

    df_params_and_results = df[[
        col for col in df.columns 
        if col in groupby_cols or col in score_cols       
    ]]

    # preserve the groupby cols after aggregation by mean(). See https://stackoverflow.com/questions/40139184/keeping-key-column-when-using-groupby-with-transform-in-pandas
    averages_per_parameter_combination = pd.concat([
        df[groupby_cols],   
        df_params_and_results.groupby(groupby_cols).transform("mean")
    ], axis=1)
    averages_per_parameter_combination = averages_per_parameter_combination.drop_duplicates()

    # group by columns: experiment, map

    # TODO: in case multiple rows have same best value, take all of them

    grouping_dims = ["gridsearch_params.hparams.env_params.map_max"]
    if use_separate_models_for_each_experiment:
        grouping_dims.append("experiment_name")        

    best_parameters_by_score = averages_per_parameter_combination.loc[averages_per_parameter_combination.groupby(grouping_dims)["test_averages.Score"].idxmax()]
    best_parameters_by_sfella_score = averages_per_parameter_combination.loc[averages_per_parameter_combination.groupby(grouping_dims)["sfella_score_average"].idxmax()]

    # if best_parameters_by_sfella_score contain -inf values then ignore these rows
    infinities = best_parameters_by_sfella_score["sfella_score_average"] == -np.inf
    best_parameters_by_sfella_score = best_parameters_by_sfella_score.drop(best_parameters_by_sfella_score[infinities].index)


    total_score_dims = ["test_averages.Score", "sfella_score_average"]
    reward_dims = ["test_averages.Reward", "test_sfella_averages.Reward"]
    average_score_dims = [col for col in averages_per_parameter_combination.columns if col.startswith("test_averages.") and col not in total_score_dims and col not in reward_dims]
    sfella_average_score_dims = [col for col in averages_per_parameter_combination.columns if col.startswith("test_sfella_averages.") and col not in total_score_dims and col not in reward_dims]
    

    gridsearch_cols = list(gridsearch_cols)
    gridsearch_cols.sort()
    gridsearch_cols = ["gridsearch_params." + col for col in gridsearch_cols 
                       if not col.endswith(".gridsearch_trial_no")    # this one is averaged out
                       and ("gridsearch_params." + col) not in grouping_dims]

    non_gridsearch_cols = list(flattened_config_without_gridsearch_keys.keys())
    non_gridsearch_cols.sort()
    non_gridsearch_cols = ["gridsearch_params." + col for col in non_gridsearch_cols]

    best_parameters_by_score = best_parameters_by_score[grouping_dims + gridsearch_cols + total_score_dims + reward_dims + sfella_average_score_dims + non_gridsearch_cols]

    best_parameters_by_sfella_score = best_parameters_by_sfella_score[grouping_dims + gridsearch_cols + total_score_dims + reward_dims + sfella_average_score_dims + non_gridsearch_cols]


    if True:    # keep only config parameter names, remove their path to make column names shorter in the CSV file   # TODO: config option
        renames = {col: col.split(".")[-1] for col in best_parameters_by_score.columns if col.startswith("gridsearch_params.")}
        best_parameters_by_score.rename(columns=renames, inplace=True)

        renames = {col: col.split(".")[-1] for col in best_parameters_by_sfella_score.columns if col.startswith("gridsearch_params.")}
        best_parameters_by_sfella_score.rename(columns=renames, inplace=True)


    filepath = cfg.hparams.params_set_title + "_best_parameters_by_score.csv"
    best_parameters_by_score.to_csv(filepath, index=False, mode='w', header=True)

    filepath = cfg.hparams.params_set_title + "_best_parameters_by_sfella_score.csv"
    best_parameters_by_sfella_score.to_csv(filepath, index=False, mode='w', header=True)

    
    input("\nAnalytics done. Press [enter] to continue.")
    qqq = True

#/ def gridsearch_analytics():



if __name__ == "__main__":

    register_resolvers()

    use_same_parameters_for_all_pipeline_experiments = False
    gridsearch_analytics()  
