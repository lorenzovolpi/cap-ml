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
    prevalence: bool = True,
):
    def _report_columns(err_names):
        base_cols = list(itertools.product(["base"], ["F", "T"]))
        prev_cols = list(itertools.product(["true", "estim"], ["TN", "FP", "FN", "TP"]))
        err_cols = list(itertools.product(["errors"], err_names))
        return base_cols, prev_cols, err_cols

    base_prevs, true_prevs, estim_prevs = estimate(estimator, protocol)

    if error_metrics == "all":
        error_metrics = ["mae", "f1"]

    error_funcs = [
        error.from_name(e) if isinstance(e, str) else e for e in error_metrics
    ]
    assert all(hasattr(e, "__call__") for e in error_funcs), "invalid error function"
    error_names = [e.__name__ for e in error_funcs]
    error_cols = []
    for err in error_names:
        if err == "mae":
            error_cols.extend(["mae estim", "mae true"])
        elif err == "f1":
            error_cols.extend(["f1 estim", "f1 true"])
        elif err == "f1e":
            error_cols.extend(["f1e estim", "f1e true"])
        else:
            error_cols.append(err)

    # df_cols = ["base_prev", "true_prev", "estim_prev"] + error_names
    base_cols, prev_cols, err_cols = _report_columns(error_cols)

    lst = []
    for base_prev, true_prev, estim_prev in zip(base_prevs, true_prevs, estim_prevs):
        if prevalence:
            series = {
                k: v
                for (k, v) in zip(
                    base_cols + prev_cols,
                    np.concatenate((base_prev, true_prev, estim_prev), axis=0),
                )
            }
            df_cols = base_cols + prev_cols + err_cols
        else:
            series = {k: v for (k, v) in zip(base_cols, base_prev)}
            df_cols = base_cols + err_cols

        for err in error_cols:
            error_funcs = {
                "mae true": lambda: error.mae(true_prev),
                "mae estim": lambda: error.mae(estim_prev),
                "f1 true": lambda: error.f1(true_prev),
                "f1 estim": lambda: error.f1(estim_prev),
                "f1e true": lambda: error.f1e(true_prev),
                "f1e estim": lambda: error.f1e(estim_prev),
            }
            series[("errors", err)] = error_funcs[err]()

        lst.append(series)

    lst = avg_groupby_distribution(lst, error_cols) if aggregate else lst

    df = pd.DataFrame(
        lst,
        columns=pd.MultiIndex.from_tuples(df_cols),
    )
    return df
