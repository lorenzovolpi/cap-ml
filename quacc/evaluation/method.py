import numpy as np
import sklearn.metrics as metrics
from quapy.data import LabelledCollection
from quapy.protocol import (
    AbstractStochasticSeededProtocol,
    OnLabelledCollectionProtocol,
)
from sklearn.base import BaseEstimator

import quacc.error as error
from quacc.evaluation.report import EvaluationReport

from ..estimator import (
    AccuracyEstimator,
    BinaryQuantifierAccuracyEstimator,
    MulticlassAccuracyEstimator,
)


def estimate(
    estimator: AccuracyEstimator,
    protocol: AbstractStochasticSeededProtocol,
):
    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

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
    }[method](c_model, q_model=q_model, **kwargs)
    estimator.fit(validation)
    _method = f"{method}_{q_model}"
    for k, v in kwargs.items():
        _method += f"_{v}"
    return evaluation_report(estimator, protocol, _method)


def evaluate_bin_sld(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "SLD")


def evaluate_mul_sld(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "SLD")


def evaluate_bin_sld_nbvs(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "SLD", recalib="nbvs")


def evaluate_mul_sld_nbvs(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "SLD", recalib="nbvs")


def evaluate_bin_sld_bcts(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "SLD", recalib="bcts")


def evaluate_mul_sld_bcts(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "SLD", recalib="bcts")


def evaluate_bin_sld_ts(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "SLD", recalib="ts")


def evaluate_mul_sld_ts(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "SLD", recalib="ts")


def evaluate_bin_sld_vs(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "SLD", recalib="vs")


def evaluate_mul_sld_vs(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "SLD", recalib="vs")


def evaluate_bin_cc(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin", "CC")


def evaluate_mul_cc(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul", "CC")
