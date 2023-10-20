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
            acc_score=1.0 - acc_score,
            acc=abs(error.acc(true_prev) - acc_score),
            f1_score=f1_score,
            f1=abs(error.f1(true_prev) - f1_score),
        )

    return report


def evaluate(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    method: str,
):
    estimator: AccuracyEstimator = {
        "bin": BinaryQuantifierAccuracyEstimator,
        "mul": MulticlassAccuracyEstimator,
    }[method](c_model)
    estimator.fit(validation)
    return evaluation_report(estimator, protocol, method)


def evaluate_bin_sld(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "bin")


def evaluate_mul_sld(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
) -> EvaluationReport:
    return evaluate(c_model, validation, protocol, "mul")
