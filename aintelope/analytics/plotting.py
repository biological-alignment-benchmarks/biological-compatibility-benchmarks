import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
import matplotlib.collections as mcoll
import matplotlib.path as mpath
import dateutil.parser as dparser
import seaborn as sns

from typing import Optional

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


def plot_performance(all_events, save_path: Optional[str]):
    """
    Plot performance between rewards and scores.
    Accepts a list of event records from which a boxplot is done.
    """
    keys = ["Run_id", "Agent_id", "Reward", "Score"]
    data = pd.DataFrame(columns=keys)
    for events in all_events:
        data.concat(events[keys])
    data.groupby(["Agent_id"])["Reward", "Score"].mean()
    plot = plt.boxplot(data)

    if save_path:
        save_plot(plot, save_path)
    return plot


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
