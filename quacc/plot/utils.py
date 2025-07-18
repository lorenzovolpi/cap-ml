import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

DPI = 300


def get_ref_limits(true_accs: np.ndarray, estim_accs: np.ndarray):
    """get lmits of reference line"""

    _edges = (
        np.min([np.min(true_accs), np.min(estim_accs)]),
        np.max([np.max(true_accs), np.max(estim_accs)]),
    )
    _lims = np.array([[_edges[0], _edges[1]], [_edges[0], _edges[1]]])
    return _lims


def get_binned_values(df, val_name, n_bins):
    # sh_min, sh_max = np.min(df.loc[:, "shifts"]), np.max(df.loc[:, "shifts"])
    val_min, val_max = 0, 1
    bins = np.linspace(val_min, val_max, n_bins + 1)
    binwidth = (val_max - val_min) / n_bins
    vals_bin_idx = np.digitize(df.loc[:, val_name], bins, right=True)
    bins[1:] = bins[1:] - binwidth / 2
    bins = np.hstack([bins, [val_max]])
    return bins[vals_bin_idx]


def save_figure(plot: Axes, basedir, filename):
    exts = [
        # "svg",
        "pdf",
        "png",
    ]
    files = [os.path.join(basedir, f"{filename}.{ext}") for ext in exts]
    for f in files:
        plot.figure.savefig(f, bbox_inches="tight", dpi=DPI)
    plot.figure.clear()
    plt.close(plot.figure)
