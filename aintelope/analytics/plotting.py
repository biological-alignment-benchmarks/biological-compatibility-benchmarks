import os

from typing import Optional

import dateutil.parser as dparser
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import math
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
import yaml

"""
Create and return plots for various analytics.
"""


def plot_history(events):
    """
    Plot the events from a history.
    args:
        events: pandas DataFrame
    return:
        plot: matplotlib.axes.Axes
    """
    plot = "NYI"

    return plot


def plot_groupby(all_events, group_keys, score_dimensions):
    keys = group_keys + score_dimensions
    data = all_events[keys]
    plot_data = data.groupby(group_keys).mean()
    return plot_data


def filter_train_and_test_events(all_events, num_train_pipeline_cycles, score_dimensions, group_by_pipeline_cycle):

    events = pd.concat(all_events)

    if all_events[0].columns[-1] == "Score":  # old AIntelope environment
        score_dimensions = ["Score"]

    if score_dimensions != ["Score"]:
        events["Score"] = events[score_dimensions].sum(axis=1)
        score_dimensions = ["Score"] + score_dimensions
    score_dimensions = ["Reward"] + score_dimensions
    events[score_dimensions] = events[score_dimensions].astype(float)
  
    if group_by_pipeline_cycle:
        train_events = events[events["Pipeline cycle"] < num_train_pipeline_cycles]
        test_events = events[events["Pipeline cycle"] >= num_train_pipeline_cycles]
    else:
        train_events = events[events["IsTest"] == 0]
        test_events = events[events["IsTest"] == 1]

    return (events, train_events, test_events, score_dimensions)
   

def calc_sfellas(df):
    """Applies pre-aggregation transformation of values using a formula developed 
    in "Using soft maximin for risk averse multi-objective decision-making"
    https://link.springer.com/article/10.1007/s10458-022-09586-2
    """

    result = pd.DataFrame().reindex_like(df).astype(df.dtypes)    # create copy of structure without data
    log_result = pd.DataFrame().reindex_like(df).astype(df.dtypes)    # create copy of structure without data

    negatives = (df <= 0)
    positives = np.logical_not(negatives)

    # use alternative exp and log base to avoid infinities
    if False:
        result[negatives] = 1 - np.exp(-df[negatives])
        result[positives] = np.log(df[positives] + 1) 
    
    else:
        base = 1.05    # TODO: config

        # for plotting the functions, use:
        # negatives(x)=((1)/(ln(1.05)))-1.05^(-x-log(1.05,ln(1.05)))
        # positives(x)=log(1.05,x+((1)/(ln(1.05))))+log(1.05,ln(1.05))

        # Align zero point of x and y, and ensure that the derivative of both of the functions is 1 at that zero point.
        # That way the ends of exp and log functions meet smoothly at 45 degree angle.
        split_point = math.log(base)
        log_split_point = math.log(base, split_point)   # yes, log of log here

        result[negatives] = (1 / split_point) - np.power(base, -df[negatives] - log_split_point)
        result[positives] = np.emath.logn(base, df[positives] + (1 / split_point)) + log_split_point


    log_result[negatives] = df[negatives]   # NB! log(sfella(-x)) is offset by log(-1) # these values are always negative
    log_result[positives] = np.log(np.log(df[positives] + 1) + 1)   # NB! log(sfella(+x)) is offset by log(+1)   # these values are always positive

    return result, log_result
    

def aggregate_test_scores(all_events, num_train_pipeline_cycles, score_dimensions, group_by_pipeline_cycle: bool = False):
    """In case of multi-agent environments, the scores are aggregated 
    over both agents without grouping by agent. Right now the agents use
    same model, so that is okay. If the agents use different models in the 
    future then maybe different approach will be needed.
    """

    (all_events, train_events, test_events, score_dimensions) = filter_train_and_test_events(all_events, num_train_pipeline_cycles, score_dimensions, group_by_pipeline_cycle)
    test_events = test_events[score_dimensions]


    totals = test_events.sum(axis=0).to_dict()
    averages = test_events.mean(axis=0).to_dict()   # sum over rows 
    # TODO: If, however, ddof is specified, the divisor N - ddof is used instead. In standard statistical practice, ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
    variances = test_events.var(axis=0, ddof=0).to_dict()

    sfellas, log_sfellas = calc_sfellas(test_events)		# TODO: aggregation of log_sfellas using formula from https://www.mathworks.com/matlabcentral/fileexchange/25273-methods-for-calculating-precise-logarithm-of-a-sum-and-subtraction

    # TODO: sum calculation for log_sfellas
    sfella_totals = sfellas.sum(axis=0).to_dict()   # sum over rows 
    sfella_averages = sfellas.mean(axis=0).to_dict()
    # TODO: If, however, ddof is specified, the divisor N - ddof is used instead. In standard statistical practice, ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
    sfella_variances = sfellas.var(axis=0, ddof=0).to_dict()
    #for key, value in sfella_variances.items():
    #    if np.isnan(value):
    #        sfella_variances[key] = 0


    score_subdimensions = list(score_dimensions)  # clone
    score_subdimensions.remove("Score")   # this is aggregated before sfella tranformation, so cannot be used for sfella score
    score_subdimensions.remove("Reward")

    if len(score_subdimensions) > 0:  # new multi-objective Gridworlds environments
        # sum over score dimensions AFTER SFELLA transformation
        sfella_scores = sfellas[score_subdimensions].sum(axis=1)      # sum over cols

        # aggregate over iterations
        sfella_score_total = sfella_scores.sum(axis=0).item()   # sum over rows
        sfella_score_average = sfella_scores.mean(axis=0).item()
        # TODO: If, however, ddof is specified, the divisor N - ddof is used instead. In standard statistical practice, ddof=1 provides an unbiased estimator of the variance of a hypothetical infinite population. ddof=0 provides a maximum likelihood estimate of the variance for normally distributed variables.
        sfella_score_variance = sfella_scores.var(axis=0, ddof=0).item()
    else: # compatibility with old AIntelope environments
        sfella_score_total = sfella_totals["Score"]
        sfella_score_average = sfella_averages["Score"]
        sfella_score_variance = sfella_variances["Score"]


    return (totals, averages, variances, sfella_totals, sfella_averages, sfella_variances, sfella_score_total, sfella_score_average, sfella_score_variance, score_dimensions)


def plot_performance(all_events, num_train_episodes, num_train_pipeline_cycles, score_dimensions, save_path: Optional[str], title: Optional[str] = "", group_by_pipeline_cycle: bool = False, do_not_show_plot: bool = False):
    """
    Plot performance between rewards and scores.
    Accepts a list of event records from which a boxplot is done.
    TODO: further consideration should be had on *what* to average over.
    """

    (all_events, train_events, test_events, score_dimensions) = filter_train_and_test_events(all_events, num_train_pipeline_cycles, score_dimensions, group_by_pipeline_cycle)

    if group_by_pipeline_cycle:
        plot_data1 = (
            "Pipeline cycle",
            plot_groupby(all_events, ["Run_id", "Pipeline cycle", "Agent_id"], score_dimensions),
        )
    else:
        plot_data1 = (
            "Episode",
            plot_groupby(all_events, ["Run_id", "Episode", "Agent_id"], score_dimensions),
        )

    plot_data2 = (
        "Train Step",
        plot_groupby(train_events, ["Run_id", "Step", "Agent_id"], score_dimensions),
    )
    plot_data3 = (
        "Test Step",
        plot_groupby(test_events, ["Run_id", "Step", "Agent_id"], score_dimensions),
    )
    plot_datas = [plot_data1, plot_data2, plot_data3]

    # fig = plt.figure()
    fig, subplots = plt.subplots(3)

    linewidth = 0.75    # TODO: config

    for index, subplot in enumerate(subplots):
        (plot_label, plot_data) = plot_datas[index]

        subplot.plot(plot_data["Reward"].to_numpy(), label="Reward", linewidth=linewidth)
        subplot.plot(plot_data["Score"].to_numpy(), label="Score", linewidth=linewidth)
        for score_dimension in score_dimensions:
            subplot.plot(plot_data[score_dimension].to_numpy(), label=score_dimension, linewidth=linewidth)

        subplot.set_title((title + " by " + plot_label).strip())
        subplot.set(xlabel=plot_label, ylabel="Mean Reward")
        subplot.legend()

    if save_path:
        save_plot(fig, save_path)

    if not do_not_show_plot:
        # enable this code if you want the plot to open automatically
        plt.ion()
        fig.show()
        plt.draw()
        # TODO: use multithreading for rendering the plot
        plt.pause(60)  # render the plot. Usually the plot is rendered quickly but sometimes it may require up to 60 sec. Else you get just a blank window

    return fig


def plot_heatmap(agent, env):
    """
    Plot how the agent sees the values in an environment.
    """
    plot = "NYI"
    return plot


def save_plot(plot, save_path):
    """
    Save plot to file. Will get deprecated if nothing else comes here.
    """
    # save in two formats. SVG is good for resizing during viewing
    plot.savefig(save_path + ".png", dpi=200)
    plot.savefig(save_path + ".svg", dpi=200)


def prettyprint(data):
		print(yaml.dump(data, allow_unicode=True, default_flow_style=False))  # default_flow_style=False moves each entry into its own line