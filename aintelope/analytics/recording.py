# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Repository: https://github.com/aintelope/biological-compatibility-benchmarks

import csv
import gzip
import lzma
import shutil
import logging
import os
import sys
from pathlib import Path
from io import BytesIO

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
1..n different runs for pipeline
n agents.
You need to change the agent params in the main config_experiment
and run the same again in order to have a comparison point.
"""


def to_xz(src_path, dst_path, dict_mb=16, nice_len=273, verify=True):
    """
    Compress src_path -> dst_path (.xz) using liblzma LZMA2 filter.
    Emulates 7z parameters: -txz -mm=LZMA2 -md=16m -mfb=273 -mx9
    """

    filters = [
        {
            "id": lzma.FILTER_LZMA2,
            "dict_size": dict_mb * 1024 * 1024,  # -md=16m
            "lc": 3,  # -mx=9
            "lp": 0,  # -mx=9
            "pb": 2,  # -mx=9
            "mode": lzma.MODE_NORMAL,  # -mx=9
            "nice_len": nice_len,  # -mfb=273
            "mf": lzma.MF_BT4,  # -mx=9: strong match finder
            "depth": 0,  # 0 = auto (liblzma chooses based on other filter options)
        }
    ]

    # XZ container with CRC64 (explicit)
    with open(src_path, "rb") as fhin, lzma.open(
        dst_path, "wb", format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, filters=filters
    ) as fhout:
        shutil.copyfileobj(fhin, fhout, length=1024 * 1024)  # 1 MB chunks

    if verify:
        with open(src_path, "rb", 1024 * 1024) as ofh:
            orig_data = ofh.read()

        with lzma.open(dst_path, "rb") as lfh:
            xz_data = lfh.read()

        return orig_data == xz_data

    else:
        return True


# / def to_xz(src_path, dst_path, dict_mb=16, nice_len=273, verify=True):


class EventLog(object):
    default_gzip_compresslevel = 6  # 6 is default level for gzip: https://linux.die.net/man/1/gzip and https://github.com/ebiggers/libdeflate

    def __init__(
        self,
        experiment_dir,
        events_fname,
        headers,
        gzip_log=False,
        gzip_compresslevel=None,
        lzma_log=False,
    ):
        self.gzip_log = gzip_log
        self.lzma_log = lzma_log
        self.record_path = Path(os.path.join(experiment_dir, events_fname))
        logger.info(f"Saving training records to disk at {self.record_path}")
        self.record_path.parent.mkdir(exist_ok=True, parents=True)

        # speed up CSV generation by not saving arrays
        # TODO: save arrays to separate pickle files
        self.state_col_index = headers.index("State")
        self.next_state_col_index = headers.index("Next_state")
        headers = [x for x in headers if x != "State" and x != "Next_state"]

        if gzip_log:
            if gzip_compresslevel is None:
                gzip_compresslevel = self.default_gzip_compresslevel
            write_header = not os.path.exists(self.record_path + ".gz")
            self.file = gzip.open(
                self.record_path + ".gz",
                mode="at",
                newline="",
                encoding="utf-8",
                compresslevel=gzip_compresslevel,
            )  # csv writer creates its own newlines therefore need to set newline to empty string here     # TODO: buffering for gzip
        else:
            write_header = not os.path.exists(self.record_path)
            self.file = open(
                self.record_path,
                mode="at",
                buffering=1024 * 1024,
                newline="",
                encoding="utf-8",
            )  # csv writer creates its own newlines therefore need to set newline to empty string here

        self.writer = csv.writer(
            self.file, quoting=csv.QUOTE_MINIMAL, delimiter=","
        )  # TODO: use TSV format instead

        if (
            write_header
        ):  # TODO: if the file already exists then assert that the header is same
            self.writer.writerow(headers)
            # self.file.flush()

    def log_event(self, event):
        transformed_cols = []
        for index, col in enumerate(event):
            # speed up CSV generation by not saving arrays
            # TODO: save arrays to separate pickle files
            if index == self.state_col_index or index == self.next_state_col_index:
                continue

            # if type(col) == datetime.datetime:
            #    col = datetime.datetime.strftime(col, '%Y.%m.%d-%H.%M.%S')

            if isinstance(col, str):
                col = (
                    col.strip()
                    .replace("\r", "\\r")
                    .replace("\n", "\\n")
                    .replace("\t", "\\t")
                )  # CSV/TSV format does not support these characters
                # col = re.sub(r"[\n\r\t]", " ", col.strip())   # CSV/TSV format does not support these characters

            transformed_cols.append(col)

        self.writer.writerow(transformed_cols)
        # self.file.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()

        if self.lzma_log and not self.gzip_log:
            xz_path = Path(str(self.record_path) + ".xz")
            if to_xz(self.record_path, xz_path, verify=True):
                os.remove(self.record_path)


# / class EventLog(object):


# def record_events(record_path, events):
#    """
#    Record events of the training to given path.
#    """

#    # speed up CSV generation by not saving arrays
#    # TODO: save arrays to separate pickle files
#    del events["State"]
#    del events["Next_state"]

#    logger.info(f"Saving training records to disk at {record_path}")
#    record_path.parent.mkdir(exist_ok=True, parents=True)

#    try_df_to_csv_write(
#        events,
#        record_path,
#        index=False,
#        mode="a",
#        header=not os.path.exists(record_path),
#    )


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

    for path in Path(record_path).rglob(events_filename + ".xz"):
        with lzma.open(path, "rb") as lfh:
            xz_data = lfh.read()
        events.append(pd.read_csv(BytesIO(xz_data)))

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
