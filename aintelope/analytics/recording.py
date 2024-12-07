# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import csv
import logging
import os
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

# this one is cross-platform
from filelock import FileLock

from aintelope.utils import try_df_to_csv_write

# Library for handling saving to file

logger = logging.getLogger("aintelope.analytics.recording")

"""
Uses config_experiment.yaml's fields:
experiment_dir: ${log_dir_root}/${timestamp}/${experiment_name}
events_dir: events.csv
checkpoint_dir: checkpoints/

HOWTO:
One test_conf will form one folder in outputs, that contains 
1..n same runs for significance
1..n different runs for pipeline
1 agent.
You need to change the agent params in the main config_experiment
and run the same again in order to have a comparison point.
"""


def record_events(record_path, events):
    """
    Record events of the training to given path.
    """

    # speed up CSV generation by not saving arrays
    # TODO: save arrays to separate pickle files
    del events["State"]
    del events["Next_state"]

    logger.info(f"Saving training records to disk at {record_path}")
    record_path.parent.mkdir(exist_ok=True, parents=True)

    try_df_to_csv_write(
        events,
        record_path,
        index=False,
        mode="a",
        header=not os.path.exists(record_path),
    )


def read_events(record_path, events_filename):
    """
    Read the events saved in record_events.
    """
    events = []

    for path in Path(record_path).rglob(events_filename):
        with FileLock(
            str(path) + ".lock"
        ):  # lock for better robustness against other processes writing to it concurrently
            events.append(pd.read_csv(path))

    return events


def read_checkpoints(checkpoint_dir):
    """
    Read models from a checkpoint.
    """
    model_paths = []
    for path in Path(checkpoint_dir).rglob("*"):
        model_paths.append(path)
    model_paths.sort(key=lambda x: os.path.getmtime(x))

    return model_paths


### Old stuff, not in use, but should belong here:

def plot_events(agent, style: str = "thickness", color: str = "viridis") -> Figure:
    """
    Docstring missing, these are old functions I'm unsure are in use atm.
    """
    events_df = agent.get_events()
    agent_df, food_df, water_df = agent.process_events(events_df)

    plt.rcParams[
        "figure.constrained_layout.use"
    ] = True  # ensure that plot labels fit to the image and do not overlap

    fig, ax = plt.subplots(figsize=(8, 8))

    if style == "thickness":
        ax.plot(agent_df["x"], agent_df["y"], ".r-")
    elif style == "colormap":
        cmap = matplotlib.colormaps[color]

        agent_arr = agent_df.to_numpy()  # coordinates x y
        # coordinates are ordered in x1 y1 x2 y2
        step_pairs = np.concatenate([agent_arr[:-1], agent_arr[1:]], axis=1)
        unique_steps, step_freq = np.unique(step_pairs, axis=0, return_counts=True)

        for line_segment, col in zip(unique_steps, step_freq / step_freq.max()):
            if (line_segment[:2] == line_segment[2:]).all():  # agent did not move
                im = ax.scatter(
                    line_segment[0],
                    line_segment[1],
                    s=70,
                    marker="o",
                    color=cmap(col),
                )
            else:
                ax.plot(line_segment[[0, 2]], line_segment[[1, 3]], color=cmap(col))

        cbar = fig.colorbar(im)
        cbar.set_label("Relative Frequency Agent")
    else:
        raise NotImplementedError(f"{style} is not a valid plot style!")

    ax.plot(food_df["x"], food_df["y"], "xg", markersize=8, label="Food")
    ax.plot(water_df["x"], water_df["y"], "xb", markersize=8, label="Water")
    ax.legend()
    plt.tight_layout()
    return fig
