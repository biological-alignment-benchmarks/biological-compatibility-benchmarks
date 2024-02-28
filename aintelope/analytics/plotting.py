from typing import Optional

import dateutil.parser as dparser
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt

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


def plot_groupby(all_events, group_keys, labels):
    keys = group_keys + ["Reward"] + labels
    data = pd.DataFrame(columns=keys)
    for events in all_events:
        if len(data) == 0:
            data = events[
                keys
            ].copy()  # needed to avoid Pandas complaining about empty dataframe
        else:
            data = pd.concat([data, events[keys]])

    data["Reward"] = data["Reward"].astype(float)
    data[labels] = data[labels].astype(float)
    data["Score"] = data[labels].sum(axis=1)

    plot_data = data.groupby(group_keys).mean()

    return plot_data


def open_plot(fig):
    # run this code if you want the plot to open automatically
    # NOT TESTED
    plt.ion()
    fig.show()
    plt.draw()
    plt.pause(0.1)
    input("Press [enter] to continue.")


def plot_performance(events, column_labels, intervals, save_path: Optional[str]):
    """
    Plot multiples into a single image by adding them into this function.
    Choose the labels you want to see from the dataset, example:
    labels = ["Score"]+score_dimensions
    Choose intervals for the different plots:
    intervals = ["Episode","Step"]

    """
    if len(intervals) > 1:
        print("Needs multiple entries in intervals.")
        return plt.figure()

    plot_datas = []
    for interval in intervals:
        plot_datas.append(
            (
                interval,
                plot_groupby(
                    events, ["Run_id", "Agent_id"] + [interval], column_labels
                ),
            )
        )

    fig, subplots = plt.subplots(len(intervals), 1)

    for index, subplot in enumerate(subplots):
        (plot_label, plot_data) = plot_datas[index]
        subplot.plot(plot_data["Reward"].to_numpy(), label="Reward")
        subplot.plot(plot_data["Score"].to_numpy(), label="Score")
        for label in column_labels:
            subplot.plot(plot_data[label].to_numpy(), label=label)

        subplot.set_title("By " + plot_label)
        subplot.set(xlabel=plot_label, ylabel="Mean Reward")
        # subplot.legend(intervals[index]) # Currently this clips TODO prettify

    if save_path:
        save_plot(fig, save_path)

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
    plot.savefig(save_path)
