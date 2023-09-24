import itertools
from quapy.protocol import (
    OnLabelledCollectionProtocol,
    AbstractStochasticSeededProtocol,
)
from typing import Iterable, Callable, Union

from .estimator import AccuracyEstimator
import pandas as pd
import numpy as np
import quacc.error as error
import statistics as stats


def estimate(
    estimator: AccuracyEstimator,
    protocol: AbstractStochasticSeededProtocol,
):
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


def avg_groupby_distribution(lst, error_names):
    def _bprev(s):
        return (s[("base", "F")], s[("base", "T")])

    def _normalize_prev(r):
        for prev_name in ["true", "estim"]:
            raw_prev = [v for ((k0, k1), v) in r.items() if k0 == prev_name]
            norm_prev = [v / sum(raw_prev) for v in raw_prev]
            for n, v in zip(
                itertools.product([prev_name], ["TN", "FP", "FN", "TP"]), norm_prev
            ):
                r[n] = v
        return r

    current_bprev = _bprev(lst[0])
    bprev_cnt = 0
    g_lst = [[]]
    for s in lst:
        if _bprev(s) == current_bprev:
            g_lst[bprev_cnt].append(s)
        else:
            g_lst.append([])
            bprev_cnt += 1
            current_bprev = _bprev(s)
            g_lst[bprev_cnt].append(s)

    r_lst = []
    for gs in g_lst:
        assert len(gs) > 0
        r = {}
        r[("base", "F")], r[("base", "T")] = _bprev(gs[0])

        for pn in [(n1, n2) for ((n1, n2), _) in gs[0].items() if n1 != "base"]:
            r[pn] = stats.mean(map(lambda s: s[pn], gs))

        r = _normalize_prev(r)

        for en in itertools.product(["errors"], error_names):
            r[en] = stats.mean(map(lambda s: s[en], gs))

        r_lst.append(r)

    return r_lst


def evaluation_report(
    estimator: AccuracyEstimator,
    protocol: AbstractStochasticSeededProtocol,
    error_metrics: Iterable[Union[str, Callable]] = "all",
    aggregate: bool = True,
):
    def _report_columns(err_names):
        base_cols = list(itertools.product(["base"], ["F", "T"]))
        prev_cols = list(itertools.product(["true", "estim"], ["TN", "FP", "FN", "TP"]))
        err_cols = list(itertools.product(["errors"], err_names))
        return base_cols + prev_cols, err_cols

    base_prevs, true_prevs, estim_prevs = estimate(estimator, protocol)

    if error_metrics == "all":
        error_metrics = ["ae", "f1"]

    error_funcs = [
        error.from_name(e) if isinstance(e, str) else e for e in error_metrics
    ]
    assert all(hasattr(e, "__call__") for e in error_funcs), "invalid error function"
    error_names = [e.__name__ for e in error_funcs]
    error_cols = error_names.copy()
    if "f1" in error_cols:
        error_cols.remove("f1")
        error_cols.extend(["f1_true", "f1_estim"])
    if "f1e" in error_cols:
        error_cols.remove("f1e")
        error_cols.extend(["f1e_true", "f1e_estim"])

    # df_cols = ["base_prev", "true_prev", "estim_prev"] + error_names
    prev_cols, err_cols = _report_columns(error_cols)

    lst = []
    for base_prev, true_prev, estim_prev in zip(base_prevs, true_prevs, estim_prevs):
        series = {
            k: v
            for (k, v) in zip(
                prev_cols, np.concatenate((base_prev, true_prev, estim_prev), axis=0)
            )
        }
        for error_name, error_metric in zip(error_names, error_funcs):
            if error_name == "f1e":
                series[("errors", "f1e_true")] = error_metric(true_prev)
                series[("errors", "f1e_estim")] = error_metric(estim_prev)
                continue
            if error_name == "f1":
                f1_true, f1_estim = error_metric(true_prev), error_metric(estim_prev)
                series[("errors", "f1_true")] = f1_true
                series[("errors", "f1_estim")] = f1_estim
                continue

            score = error_metric(true_prev, estim_prev)
            series[("errors", error_name)] = score

        lst.append(series)

    lst = avg_groupby_distribution(lst, error_cols) if aggregate else lst

    df = pd.DataFrame(
        lst,
        columns=pd.MultiIndex.from_tuples(prev_cols + err_cols),
    )
    return df
