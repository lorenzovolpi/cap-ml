import multiprocessing
import time

import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import (
    APP,
    AbstractStochasticSeededProtocol,
    OnLabelledCollectionProtocol,
)
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

import quacc.error as error
import quacc.evaluation.baseline as baseline
from quacc.dataset import get_imdb, get_rcv1, get_spambase
from quacc.evaluation.report import EvaluationReport

from ..estimator import (
    AccuracyEstimator,
    BinaryQuantifierAccuracyEstimator,
    MulticlassAccuracyEstimator,
)

qp.environ["SAMPLE_SIZE"] = 100

pd.set_option("display.float_format", "{:.4f}".format)

n_prevalences = 21
repreats = 100


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


def evaluation_report(
    estimator: AccuracyEstimator,
    protocol: AbstractStochasticSeededProtocol,
    method: str,
) -> EvaluationReport:
    base_prevs, true_prevs, estim_prevs = estimate(estimator, protocol)
    report = EvaluationReport(prefix=method)

    for base_prev, true_prev, estim_prev in zip(base_prevs, true_prevs, estim_prevs):
        acc_score = error.acc(estim_prev)
        f1_score = error.f1(estim_prev)
        report.append_row(
            base_prev, 
            acc_score=1. - acc_score,
            acc = abs(error.acc(true_prev) - acc_score),
            f1_score=f1_score,
            f1=abs(error.f1(true_prev) - f1_score)
        )

    return report


def evaluate(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    method: str,
):
    estimator : AccuracyEstimator = {
        "bin": BinaryQuantifierAccuracyEstimator,
        "mul": MulticlassAccuracyEstimator,
    }[method](c_model)
    estimator.fit(validation)
    return evaluation_report(estimator, protocol, method)


def evaluate_binary(model, validation, protocol):
    return evaluate(model, validation, protocol, "bin")


def evaluate_multiclass(model, validation, protocol):
    return evaluate(model, validation, protocol, "mul")


def fit_and_estimate(_estimate, train, validation, test):
    model = LogisticRegression()

    model.fit(*train.Xy)
    protocol = APP(test, n_prevalences=n_prevalences, repeats=repreats)
    start = time.time()
    result = _estimate(model, validation, protocol)
    end = time.time()

    return {
        "name": _estimate.__name__,
        "result": result,
        "time": end - start,
    }


def evaluate_comparison(dataset: str, **kwargs) -> EvaluationReport:
    train, validation, test = {
        "spambase": get_spambase,
        "imdb": get_imdb,
        "rcv1": get_rcv1,
    }[dataset](**kwargs)

    for k,v in kwargs.items():
        print(k, ":", v)

    prevs = {
        "train": train.prevalence(),
        "validation": validation.prevalence(),
    }

    start = time.time()
    with multiprocessing.Pool(8) as pool:
        estimators = [
            evaluate_binary,
            evaluate_multiclass,
            baseline.kfcv,
            baseline.atc_mc,
            baseline.atc_ne,
            baseline.doc_feat,
            baseline.rca_score,
            baseline.rca_star_score,
        ]
        tasks = [(estim, train, validation, test) for estim in estimators]
        results = [pool.apply_async(fit_and_estimate, t) for t in tasks]
        results = list(map(lambda r: r.get(), results))
        er = EvaluationReport.combine_reports(*list(map(lambda r: r["result"], results)))
        times = {r["name"]:r["time"] for r in results}
    end = time.time()
    times["tot"] = end - start
    er.times = times
    er.prevs = prevs

    return er
