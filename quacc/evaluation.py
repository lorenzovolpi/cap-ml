from quapy.protocol import (
    OnLabelledCollectionProtocol,
    AbstractStochasticSeededProtocol,
)
import quapy as qp
from typing import Iterable, Callable, Union

from .estimator import AccuracyEstimator
import pandas as pd
import quacc.error as error


def estimate(estimator: AccuracyEstimator, protocol: AbstractStochasticSeededProtocol):
    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    base_prevs, true_prevs, estim_prevs = [], [], []
    for sample in protocol():
        e_sample = estimator.extend(sample)
        estim_prev = estimator.estimate(e_sample.X, ext=True)
        base_prevs.append(sample.prevalence())
        true_prevs.append(e_sample.prevalence())
        estim_prevs.append(estim_prev)

    return base_prevs, true_prevs, estim_prevs


def evaluation_report(
    estimator: AccuracyEstimator,
    protocol: AbstractStochasticSeededProtocol,
    error_metrics: Iterable[Union[str, Callable]] = "all",
):
    base_prevs, true_prevs, estim_prevs = estimate(estimator, protocol)

    if error_metrics == "all":
        error_metrics = ["mae", "rae", "mrae", "kld", "nkld", "f1e"]

    error_funcs = [
        error.from_name(e) if isinstance(e, str) else e for e in error_metrics
    ]
    assert all(hasattr(e, "__call__") for e in error_funcs), "invalid error function"
    error_names = [e.__name__ for e in error_funcs]

    df_cols = ["base_prev", "true_prev", "estim_prev"] + error_names
    if "f1e" in df_cols:
        df_cols.remove("f1e")
        df_cols.extend(["f1e_true", "f1e_estim"])
    lst = []
    for base_prev, true_prev, estim_prev in zip(base_prevs, true_prevs, estim_prevs):
        series = {
            "base_prev": base_prev,
            "true_prev": true_prev,
            "estim_prev": estim_prev,
        }
        for error_name, error_metric in zip(error_names, error_funcs):
            if error_name == "f1e":
                series["f1e_true"] = error_metric(true_prev)
                series["f1e_estim"] = error_metric(estim_prev)
                continue

            score = error_metric(true_prev, estim_prev)
            series[error_name] = score

        lst.append(series)

    df = pd.DataFrame(lst, columns=df_cols)
    return df
