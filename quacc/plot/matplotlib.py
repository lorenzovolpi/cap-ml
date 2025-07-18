import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure

from quacc.plot.utils import get_ref_limits
from quacc.utils.commons import get_plots_path


def _get_markers(num: int):
    ls = "ovx+sDph*^1234X><.Pd"
    if num > len(ls):
        ls = ls * (num / len(ls) + 1)
    return list(ls)[:num]


def _get_cycler(num):
    cm = plt.get_cmap("tab20") if num > 10 else plt.get_cmap("tab10")
    return cycler(color=[cm(i) for i in range(num)])


def _save_or_return(fig: Figure, basedir, cls_name, acc_name, dataset_name, plot_type) -> Figure | None:
    if basedir is None:
        return fig

    plotsubdir = "all" if dataset_name == "*" else dataset_name
    file = get_plots_path(basedir, cls_name, acc_name, plotsubdir, plot_type)
    os.makedirs(Path(file).parent, exist_ok=True)
    fig.savefig(file)


def plot_diagonal(
    method_names: list[str],
    true_accs: np.ndarray,
    estim_accs: np.ndarray,
    cls_name,
    acc_name,
    dataset_name,
    *,
    basedir=None,
) -> Figure | None:
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_aspect("equal")

    cy = _get_cycler(len(method_names))

    for name, x, estim, _cy in zip(method_names, true_accs, estim_accs, cy):
        ax.plot(
            x,
            estim,
            label=name,
            color=_cy["color"],
            linestyle="None",
            marker="o",
            markersize=3,
            zorder=2,
            alpha=0.25,
        )

    # ensure limits are equal for both axes
    _lims = get_ref_limits(true_accs, estim_accs)
    ax.set(xlim=_lims[0], ylim=_lims[1])

    # draw polyfit line per method
    # for name, x, estim, _cy in zip(method_names, true_accs, estim_accs, cy):
    #     slope, interc = np.polyfit(x, estim, 1)
    #     y_lr = np.array([slope * x + interc for x in _lims])
    #     ax.plot(
    #         _lims,
    #         y_lr,
    #         label=name,
    #         color=_cy["color"],
    #         linestyle="-",
    #         markersize="0",
    #         zorder=1,
    #     )

    # plot reference line
    ax.plot(
        _lims,
        _lims,
        color="black",
        linestyle="--",
        markersize=0,
        zorder=1,
    )

    ax.set(xlabel=f"True {acc_name}", ylabel=f"Estimated {acc_name}")

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    file_name = "diagonal"
    return _save_or_return(fig, basedir, cls_name, acc_name, dataset_name, file_name)


def plot_delta(
    method_names: list[str],
    prevs: np.ndarray,
    acc_errs: np.ndarray,
    cls_name,
    acc_name,
    dataset_name,
    *,
    stdevs: np.ndarray | None = None,
    basedir=None,
) -> Figure | None:
    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    ax.grid()

    cy = _get_cycler(len(method_names))

    if stdevs is None:
        stdevs = [None] * len(method_names)
    for name, x, delta, stdev, _cy in zip(method_names, prevs, acc_errs, stdevs, cy):
        x = [str(p) for p in x]
        ax.plot(
            x,
            delta,
            label=name,
            color=_cy["color"],
            linestyle="-",
            marker="",
            markersize=3,
            zorder=2,
        )
        if stdev is not None:
            ax.fill_between(
                prevs,
                delta - stdev,
                delta + stdev,
                color=_cy["color"],
                alpha=0.25,
            )

    ax.set(
        xlabel="Test Prevalence",
        ylabel=f"Prediction Error for {acc_name}",
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    file_name = "delta" if stdevs is None else "stdev"
    return _save_or_return(fig, basedir, cls_name, acc_name, dataset_name, file_name)


def plot_shift(
    method_names: list[str],
    prevs: np.ndarray,
    acc_errs: np.ndarray,
    cls_name,
    acc_name,
    dataset_name,
    *,
    basedir=None,
) -> Figure | None:
    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    ax.grid()

    cy = _get_cycler(len(method_names))

    for name, x, delta, _cy in zip(method_names, prevs, acc_errs, cy):
        x = [str(p) for p in x]
        ax.plot(
            x,
            delta,
            label=name,
            color=_cy["color"],
            linestyle="-",
            marker="",
            markersize=3,
            zorder=2,
        )

    ax.set(
        xlabel="Amount of Prior Probability Shift",
        ylabel=f"Prediction Error for {acc_name}",
    )

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    file_name = "shift"
    return _save_or_return(fig, basedir, cls_name, acc_name, dataset_name, file_name)
