from io import StringIO
import pandas as pd
import numpy as np
import scipy

from aintelope.utils import wait_for_enter


def compute_significance():
    mixed = """
map_max	num_iters	conv_size	gamma	num_conv_layers	num_episodes	count	mean.test_averages.Score	std.test_averages.Score										
7	300	3	0.7	2	30	4	-17.82592258	49.59143484										
9	400	2	0.9	2	30	4	-24.85383816	62.56987367
    """

    score = """
map_max	num_iters	conv_size	gamma	num_conv_layers	num_episodes	count	mean.test_averages.Score	std.test_averages.Score										
7	300	2	0.9	2	30	4	-61.33033835	79.46394627										
9	300	2	0.9	2	30	4	-66.2625402	79.63265602					
    """

    mixed = pd.read_csv(StringIO(mixed), sep="\t")
    score = pd.read_csv(StringIO(score), sep="\t")

    # print(f"map_max experiment_name pvalue statistic")
    results_dicts = []

    for index, (pd_index, mixed_row) in enumerate(mixed.iterrows()):
        score_row = score.iloc[index]

        map_max = mixed_row["map_max"]
        assert map_max == score_row["map_max"]

        if "experiment_name" in mixed.columns:
            experiment_name = mixed_row["experiment_name"]
            assert experiment_name == score_row["experiment_name"]
        else:
            experiment_name = "*"
            assert "experiment_name" not in score.columns

        mixed_n = mixed_row["count"]
        mixed_mean = mixed_row["mean.test_averages.Score"]
        mixed_std = mixed_row["std.test_averages.Score"]

        score_n = score_row["count"]
        score_mean = score_row["mean.test_averages.Score"]
        score_std = score_row["std.test_averages.Score"]

        result = scipy.stats.ttest_ind_from_stats(
            mixed_mean,
            mixed_std,
            mixed_n,
            score_mean,
            score_std,
            score_n,
            equal_var=True,
            alternative="two-sided",
        )
        statistic = result.statistic
        pvalue = result.pvalue

        # print(f"{map_max} {experiment_name} {pvalue} {statistic}")
        results_dicts.append(
            {
                "map_max": map_max,
                "experiment_name": experiment_name,
                "pvalue": pvalue,
                "statistic": statistic,
            }
        )

    # / for index, (pd_index, mixed_row) in enumerate(mixed.iterrows()):

    results_df = pd.DataFrame.from_records(results_dicts)
    print(results_df)

    wait_for_enter("\nDone. Press [enter] to continue.")
    qqq = True


# / def compute_significance():


if __name__ == "__main__":
    compute_significance()
