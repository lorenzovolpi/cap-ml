import matplotlib.pyplot as plt

from quacc.environ import env


def plot_delta(base_prevs, dict_vals, metric, title):
    fig, ax = plt.subplots()

    base_prevs = [f for f, p in base_prevs]
    for method, deltas in dict_vals.items():
        ax.plot(
            base_prevs,
            deltas,
            label=method,
            linestyle="-",
            marker="o",
            markersize=3,
            zorder=2,
        )

    ax.set(xlabel="test prevalence", ylabel=metric, title=title)
    # ax.set_ylim(0, 1)
    # ax.set_xlim(0, 1)
    ax.legend()
    output_path = env.PLOT_OUT_DIR / f"{title}.png"
    plt.savefig(output_path)
