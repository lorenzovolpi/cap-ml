from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from quacc.environment import env

matplotlib.use("agg")


def _get_markers(n: int):
    ls = "ovx+sDph*^1234X><.Pd"
    if n > len(ls):
        ls = ls * (n / len(ls) + 1)
    return list(ls)[:n]


def plot_delta(
    base_prevs,
    columns,
    data,
    *,
    stdevs=None,
    pos_class=1,
    metric="acc",
    name="default",
    train_prev=None,
    fit_scores=None,
    legend=True,
    avg=None,
) -> Path:
    _base_title = "delta_stdev" if stdevs is not None else "delta"
    if train_prev is not None:
        t_prev_pos = int(round(train_prev[pos_class] * 100))
        title = f"{_base_title}_{name}_{t_prev_pos}_{metric}"
    else:
        title = f"{_base_title}_{name}_avg_{avg}_{metric}"

    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    ax.grid()

    NUM_COLORS = len(data)
    cm = plt.get_cmap("tab10")
    if NUM_COLORS > 10:
        cm = plt.get_cmap("tab20")
    cy = cycler(color=[cm(i) for i in range(NUM_COLORS)])

    base_prevs = base_prevs[:, pos_class]
    for method, deltas, _cy in zip(columns, data, cy):
        ax.plot(
            base_prevs,
            deltas,
            label=method,
            color=_cy["color"],
            linestyle="-",
            marker="o",
            markersize=3,
            zorder=2,
        )
        if stdevs is not None:
            _col_idx = np.where(columns == method)[0]
            stdev = stdevs[_col_idx].flatten()
            nn_idx = np.intersect1d(
                np.where(deltas != np.nan)[0],
                np.where(stdev != np.nan)[0],
            )
            _bps, _ds, _st = base_prevs[nn_idx], deltas[nn_idx], stdev[nn_idx]
            ax.fill_between(
                _bps,
                _ds - _st,
                _ds + _st,
                color=_cy["color"],
                alpha=0.25,
            )
        if fit_scores is not None and method in fit_scores:
            ax.plot(
                base_prevs,
                np.repeat(fit_scores[method], base_prevs.shape[0]),
                color=_cy["color"],
                linestyle="--",
                markersize=0,
            )

    x_label = "test" if avg is None or avg == "train" else "train"
    ax.set(
        xlabel=f"{x_label} prevalence",
        ylabel=metric,
        title=title,
    )

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    output_path = env.PLOT_OUT_DIR / f"{title}.png"
    fig.savefig(output_path, bbox_inches="tight")

    return output_path


def plot_diagonal(
    reference,
    columns,
    data,
    *,
    pos_class=1,
    metric="acc",
    name="default",
    train_prev=None,
    legend=True,
):
    if train_prev is not None:
        t_prev_pos = int(round(train_prev[pos_class] * 100))
        title = f"diagonal_{name}_{t_prev_pos}_{metric}"
    else:
        title = f"diagonal_{name}_{metric}"

    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    ax.grid()
    ax.set_aspect("equal")

    NUM_COLORS = len(data)
    cm = plt.get_cmap("tab10")
    if NUM_COLORS > 10:
        cm = plt.get_cmap("tab20")
    cy = cycler(
        color=[cm(i) for i in range(NUM_COLORS)],
        marker=_get_markers(NUM_COLORS),
    )

    reference = np.array(reference)
    x_ticks = np.unique(reference)
    x_ticks.sort()

    for deltas, _cy in zip(data, cy):
        ax.plot(
            reference,
            deltas,
            color=_cy["color"],
            linestyle="None",
            marker=_cy["marker"],
            markersize=3,
            zorder=2,
            alpha=0.25,
        )

    # ensure limits are equal for both axes
    _alims = np.stack(((ax.get_xlim(), ax.get_ylim())), axis=-1)
    _lims = np.array([f(ls) for f, ls in zip([np.min, np.max], _alims)])
    ax.set(xlim=tuple(_lims), ylim=tuple(_lims))

    for method, deltas, _cy in zip(columns, data, cy):
        slope, interc = np.polyfit(reference, deltas, 1)
        y_lr = np.array([slope * x + interc for x in _lims])
        ax.plot(
            _lims,
            y_lr,
            label=method,
            color=_cy["color"],
            linestyle="-",
            markersize="0",
            zorder=1,
        )

    # plot reference line
    ax.plot(
        _lims,
        _lims,
        color="black",
        linestyle="--",
        markersize=0,
        zorder=1,
    )

    ax.set(xlabel=f"true {metric}", ylabel=f"estim. {metric}", title=title)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    output_path = env.PLOT_OUT_DIR / f"{title}.png"
    fig.savefig(output_path, bbox_inches="tight")
    return output_path


def plot_shift(
    shift_prevs,
    columns,
    data,
    *,
    pos_class=1,
    metric="acc",
    name="default",
    train_prev=None,
    fit_scores=None,
    legend=True,
) -> Path:
    if train_prev is not None:
        t_prev_pos = int(round(train_prev[pos_class] * 100))
        title = f"shift_{name}_{t_prev_pos}_{metric}"
    else:
        title = f"shift_{name}_avg_{metric}"

    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    ax.grid()

    NUM_COLORS = len(data)
    cm = plt.get_cmap("tab10")
    if NUM_COLORS > 10:
        cm = plt.get_cmap("tab20")
    cy = cycler(color=[cm(i) for i in range(NUM_COLORS)])

    shift_prevs = shift_prevs[:, pos_class]
    for method, shifts, _cy in zip(columns, data, cy):
        ax.plot(
            shift_prevs,
            shifts,
            label=method,
            color=_cy["color"],
            linestyle="-",
            marker="o",
            markersize=3,
            zorder=2,
        )

        if fit_scores is not None and method in fit_scores:
            ax.plot(
                shift_prevs,
                np.repeat(fit_scores[method], shift_prevs.shape[0]),
                color=_cy["color"],
                linestyle="--",
                markersize=0,
            )

    ax.set(xlabel="dataset shift", ylabel=metric, title=title)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    output_path = env.PLOT_OUT_DIR / f"{title}.png"
    fig.savefig(output_path, bbox_inches="tight")

    return output_path
