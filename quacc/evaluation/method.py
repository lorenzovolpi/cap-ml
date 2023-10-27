from functools import wraps

import numpy as np
import sklearn.metrics as metrics
from quapy.data import LabelledCollection
from quapy.protocol import AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator

import quacc.error as error
from quacc.evaluation.report import EvaluationReport

from ..estimator import (
    AccuracyEstimator,
    BinaryQuantifierAccuracyEstimator,
    MulticlassAccuracyEstimator,
)

_methods = {}


def method(func):
    @wraps(func)
    def wrapper(c_model, validation, protocol):
        return func(c_model, validation, protocol)

    _methods[func.__name__] = wrapper

    return wrapper


def estimate(
    estimator: AccuracyEstimator,
    protocol: AbstractStochasticSeededProtocol,
):
    base_prevs, true_prevs, estim_prevs, pred_probas, labels = [], [], [], [], []
    for sample in protocol():
        e_sample, pred_proba = estimator.extend(sample)
        estim_prev = estimator.estimate(e_sample.X, ext=True)
        base_prevs.append(sample.prevalence())
        true_prevs.append(e_sample.prevalence())
        estim_prevs.append(estim_prev)
        pred_probas.append(pred_proba)
        labels.append(sample.y)

    return base_prevs, true_prevs, estim_prevs, pred_probas, labels


def evaluation_report(
    estimator: AccuracyEstimator,
    protocol: AbstractStochasticSeededProtocol,
    method: str,
) -> EvaluationReport:
    base_prevs, true_prevs, estim_prevs, pred_probas, labels = estimate(
        estimator, protocol
    )
    report = EvaluationReport(name=method)

    for base_prev, true_prev, estim_prev, pred_proba, label in zip(
        base_prevs, true_prevs, estim_prevs, pred_probas, labels
    ):
        pred = np.argmax(pred_proba, axis=-1)
        acc_score = error.acc(estim_prev)
        f1_score = error.f1(estim_prev)
        report.append_row(
            base_prev,
            acc_score=acc_score,
            acc=abs(metrics.accuracy_score(label, pred) - acc_score),
            f1_score=f1_score,
            f1=abs(error.f1(true_prev) - f1_score),
        )

    report.fit_score = estimator.fit_score

    return report


def evaluate(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    method: str,
    q_model: str,
    **kwargs,
):
    estimator: AccuracyEstimator = {
        "bin": BinaryQuantifierAccuracyEstimator,
        "mul": MulticlassAccuracyEstimator,
    }[method](c_model, q_model=q_model.upper(), **kwargs)
    estimator.fit(validation)
    _method = f"{method}_{q_model}"
    if "recalib" in kwargs:
        _method += f"_{kwargs['recalib']}"
    if ("gs", True) in kwargs.items():
        _method += "_gs"
    return evaluation_report(estimator, protocol, _method)


@method
def bin_sld(c_model, validation, protocol) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "sld")


@method
def mul_sld(c_model, validation, protocol) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "sld")


@method
def bin_sld_bcts(c_model, validation, protocol) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "sld", recalib="bcts")


@method
def mul_sld_bcts(c_model, validation, protocol) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "sld", recalib="bcts")


@method
def bin_sld_gs(c_model, validation, protocol) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "sld", gs=True)


@method
def mul_sld_gs(c_model, validation, protocol) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "sld", gs=True)


@method
def bin_cc(c_model, validation, protocol) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "cc")


@method
def mul_cc(c_model, validation, protocol) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "cc")
