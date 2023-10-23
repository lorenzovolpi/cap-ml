from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from quacc.environ import env


def _get_markers(n: int):
    ls = [
        "o",
        "v",
        "x",
        "+",
        "s",
        "D",
        "p",
        "h",
        "*",
        "^",
        "1",
        "2",
        "3",
        "4",
        "X",
        ">",
        "<",
        ".",
        "P",
        "d",
    ]
    if n > len(ls):
        ls = ls * (n / len(ls) + 1)
    return ls[:n]


def plot_delta(
    base_prevs,
    dict_vals,
    *,
    pos_class=1,
    metric="acc",
    name="default",
    train_prev=None,
    legend=True,
) -> Path:
    if train_prev is not None:
        t_prev_pos = int(round(train_prev[pos_class] * 100))
        title = f"delta_{name}_{t_prev_pos}_{metric}"
    else:
        title = f"delta_{name}_{metric}"

    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    ax.grid()

    NUM_COLORS = len(dict_vals)
    cm = plt.get_cmap("tab10")
    if NUM_COLORS > 10:
        cm = plt.get_cmap("tab20")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)],
    )

    base_prevs = [bp[pos_class] for bp in base_prevs]
    for method, deltas in dict_vals.items():
        avg = np.array([np.mean(d, axis=-1) for d in deltas])
        # std = np.array([np.std(d, axis=-1) for d in deltas])
        ax.plot(
            base_prevs,
            avg,
            label=method,
            linestyle="-",
            marker="o",
            markersize=3,
            zorder=2,
        )
        # ax.fill_between(base_prevs, avg - std, avg + std, alpha=0.25)

    ax.set(xlabel="test prevalence", ylabel=metric, title=title)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    output_path = env.PLOT_OUT_DIR / f"{title}.png"
    fig.savefig(output_path, bbox_inches="tight")

    return output_path


def plot_diagonal(
    reference,
    dict_vals,
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

    NUM_COLORS = len(dict_vals)
    cm = plt.get_cmap("tab10")
    ax.set_prop_cycle(
        marker=_get_markers(NUM_COLORS) * 2,
        color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)] * 2,
    )

    reference = np.array(reference)
    x_ticks = np.unique(reference)
    x_ticks.sort()

    for _, deltas in dict_vals.items():
        deltas = np.array(deltas)
        ax.plot(
            reference,
            deltas,
            linestyle="None",
            markersize=3,
            zorder=2,
        )

    for method, deltas in dict_vals.items():
        deltas = np.array(deltas)
        x_interp = x_ticks[[0, -1]]
        y_interp = np.interp(x_interp, reference, deltas)
        ax.plot(
            x_interp,
            y_interp,
            label=method,
            linestyle="-",
            markersize="0",
            zorder=1,
        )

    ax.set(xlabel="test prevalence", ylabel=metric, title=title)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    output_path = env.PLOT_OUT_DIR / f"{title}.png"
    fig.savefig(output_path, bbox_inches="tight")
    return output_path


def plot_shift(
    base_prevs,
    dict_vals,
    *,
    pos_class=1,
    metric="acc",
    name="default",
    train_prev=None,
    legend=True,
) -> Path:
    if train_prev is None:
        raise AttributeError("train_prev cannot be None.")

    train_prev = train_prev[pos_class]
    t_prev_pos = int(round(train_prev * 100))
    title = f"shift_{name}_{t_prev_pos}_{metric}"

    fig, ax = plt.subplots()
    ax.set_aspect("auto")
    ax.grid()

    NUM_COLORS = len(dict_vals)
    cm = plt.get_cmap("tab10")
    if NUM_COLORS > 10:
        cm = plt.get_cmap("tab20")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / NUM_COLORS) for i in range(NUM_COLORS)],
    )

    base_prevs = np.around(
        [abs(bp[pos_class] - train_prev) for bp in base_prevs], decimals=2
    )
    for method, deltas in dict_vals.items():
        delta_bins = {}
        for bp, delta in zip(base_prevs, deltas):
            if bp not in delta_bins:
                delta_bins[bp] = []
            delta_bins[bp].append(delta)

        bp_unique, delta_avg = zip(
            *sorted(
                {k: np.mean(v) for k, v in delta_bins.items()}.items(),
                key=lambda db: db[0],
            )
        )

        ax.plot(
            bp_unique,
            delta_avg,
            label=method,
            linestyle="-",
            marker="o",
            markersize=3,
            zorder=2,
        )

    ax.set(xlabel="test prevalence", ylabel=metric, title=title)

    if legend:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    output_path = env.PLOT_OUT_DIR / f"{title}.png"
    fig.savefig(output_path, bbox_inches="tight")

    return output_path
