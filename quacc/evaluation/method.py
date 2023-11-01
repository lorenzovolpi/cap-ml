from functools import wraps
from typing import Callable, Union

import numpy as np
from quapy.method.aggregative import SLD
from quapy.protocol import UPP, AbstractProtocol, OnLabelledCollectionProtocol
from sklearn.linear_model import LogisticRegression

import quacc as qc
from quacc.evaluation.report import EvaluationReport
from quacc.method.model_selection import GridSearchAE

from ..method.base import BQAE, MCAE, BaseAccuracyEstimator

_methods = {}


def method(func):
    @wraps(func)
    def wrapper(c_model, validation, protocol):
        return func(c_model, validation, protocol)

    _methods[func.__name__] = wrapper

    return wrapper


def evaluate(
    estimator: BaseAccuracyEstimator,
    protocol: AbstractProtocol,
    error_metric: Union[Callable | str],
) -> float:
    if isinstance(error_metric, str):
        error_metric = qc.error.from_name(error_metric)

    collator_bck_ = protocol.collator
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    estim_prevs, true_prevs = [], []
    for sample in protocol():
        e_sample = estimator.extend(sample)
        estim_prev = estimator.estimate(e_sample.X, ext=True)
        estim_prevs.append(estim_prev)
        true_prevs.append(e_sample.prevalence())

    protocol.collator = collator_bck_

    true_prevs = np.array(true_prevs)
    estim_prevs = np.array(estim_prevs)

    return error_metric(true_prevs, estim_prevs)


def evaluation_report(
    estimator: BaseAccuracyEstimator,
    protocol: AbstractProtocol,
    method: str,
) -> EvaluationReport:
    report = EvaluationReport(name=method)
    for sample in protocol():
        e_sample = estimator.extend(sample)
        estim_prev = estimator.estimate(e_sample.X, ext=True)
        acc_score = qc.error.acc(estim_prev)
        f1_score = qc.error.f1(estim_prev)
        report.append_row(
            sample.prevalence(),
            acc_score=acc_score,
            acc=abs(qc.error.acc(e_sample.prevalence()) - acc_score),
            f1_score=f1_score,
            f1=abs(qc.error.f1(e_sample.prevalence()) - f1_score),
        )

    return report


@method
def bin_sld(c_model, validation, protocol) -> EvaluationReport:
    est = BQAE(c_model, SLD(LogisticRegression()))
    est.fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
        method="bin_sld",
    )


@method
def mul_sld(c_model, validation, protocol) -> EvaluationReport:
    est = MCAE(c_model, SLD(LogisticRegression()))
    est.fit(validation)
    return evaluation_report(
        estimator=est,
        protocor=protocol,
        method="mul_sld",
    )


@method
def bin_sld_bcts(c_model, validation, protocol) -> EvaluationReport:
    est = BQAE(c_model, SLD(LogisticRegression(), recalib="bcts"))
    est.fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
        method="bin_sld_bcts",
    )


@method
def mul_sld_bcts(c_model, validation, protocol) -> EvaluationReport:
    est = MCAE(c_model, SLD(LogisticRegression(), recalib="bcts"))
    est.fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
        method="mul_sld_bcts",
    )


@method
def mul_sld_gs(c_model, validation, protocol) -> EvaluationReport:
    v_train, v_val = validation.split_stratified(0.6, random_state=0)
    model = SLD(LogisticRegression())
    est = GridSearchAE(
        model=model,
        param_grid={
            "q__classifier__C": np.logspace(-3, 3, 7),
            "q__classifier__class_weight": [None, "balanced"],
            "q__recalib": [None, "bcts", "vs"],
        },
        refit=False,
        protocol=UPP(v_val, repeats=100),
        verbose=True,
    ).fit(v_train)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
        method="mul_sld_gs",
    )
