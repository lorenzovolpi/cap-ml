import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go

from quacc.plot.utils import get_binned_values, get_ref_limits

MODE = "lines"
L_WIDTH = 3
LEGEND = {
    "title": "",
    "orientation": "h",
    "yanchor": "bottom",
    "xanchor": "right",
    "x": 1,
    "y": 1.02,
    "font": {
        # "family": "DejaVu Sans",
        "size": 14,
    },
}
FONT = {"size": 14}
TEMPLATE = "seaborn"


def _update_layout(fig, x_label, y_label, **kwargs):
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=TEMPLATE,
        font=FONT,
        legend=LEGEND,
        **kwargs,
    )


def _hex_to_rgb(hex: str, t: float | None = None):
    hex = hex.lstrip("#")
    rgb = [int(hex[i : i + 2], 16) for i in [0, 2, 4]]
    if t is not None:
        rgb.append(t)
    return f"{'rgb' if t is None else 'rgba'}{str(tuple(rgb))}"


def _get_colors(num):
    match num:
        case v if v > 10:
            __colors = plotly.colors.qualitative.Light24
        case _:
            __colors = plotly.colors.qualitative.G10

    def __generator(cs):
        while True:
            for c in cs:
                yield c

    return __generator(__colors)


def plot_diagonal(df: pd.DataFrame, cls_name, acc_name, dataset_name):
    fig = px.scatter(df, x="true_accs", y="estim_accs", color="method", opacity=0.5, hover_data=["prev"])

    lims = get_ref_limits(df["true_accs"].to_numpy(), df["estim_accs"].to_numpy())
    fig.add_scatter(x=lims[0], y=lims[1], line=dict(color="black", dash="dash"), mode="lines", showlegend=False)

    _update_layout(
        fig,
        x_label="True Accuracy",
        y_label="Estimated Accuracy",
        yaxis_scaleanchor="x",
        yaxis_scaleratio=1.0,
    )

    return fig


def plot_shift(
    df: pd.DataFrame,
    cls_name,
    acc_name,
    dataset_name,
    *,
    n_bins=20,
    basedir=None,
    file_name=None,
) -> go.Figure:
    # binning on shift values
    df.loc[:, "shifts_bin"] = get_binned_values(df, "shifts", n_bins)

    fig = px.line(
        df.groupby(["shifts_bin", "method"]).mean().reset_index(),
        x="shifts_bin",
        y="acc_err",
        color="method",
    )
    fig.update_traces(line=dict(width=L_WIDTH))

    _update_layout(fig, "Amount of PPS", "Accuracy Prediction Error")
    return fig
