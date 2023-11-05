import inspect
from functools import wraps

import numpy as np
from quapy.method.aggregative import PACC, SLD
from quapy.protocol import UPP, AbstractProtocol
from sklearn.linear_model import LogisticRegression

import quacc as qc
from quacc.evaluation.report import EvaluationReport
from quacc.method.model_selection import BQAEgsq, GridSearchAE, MCAEgsq

from ..method.base import BQAE, MCAE, BaseAccuracyEstimator

_methods = {}


def method(func):
    @wraps(func)
    def wrapper(c_model, validation, protocol):
        return func(c_model, validation, protocol)

    _methods[func.__name__] = wrapper

    return wrapper


def evaluation_report(
    estimator: BaseAccuracyEstimator,
    protocol: AbstractProtocol,
) -> EvaluationReport:
    method_name = inspect.stack()[1].function
    report = EvaluationReport(name=method_name)
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
    est = BQAE(c_model, SLD(LogisticRegression())).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def mul_sld(c_model, validation, protocol) -> EvaluationReport:
    est = MCAE(c_model, SLD(LogisticRegression())).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def binmc_sld(c_model, validation, protocol) -> EvaluationReport:
    est = BQAE(
        c_model,
        SLD(LogisticRegression()),
        confidence="max_conf",
    ).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def mulmc_sld(c_model, validation, protocol) -> EvaluationReport:
    est = MCAE(
        c_model,
        SLD(LogisticRegression()),
        confidence="max_conf",
    ).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def binne_sld(c_model, validation, protocol) -> EvaluationReport:
    est = BQAE(
        c_model,
        SLD(LogisticRegression()),
        confidence="entropy",
    ).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def mulne_sld(c_model, validation, protocol) -> EvaluationReport:
    est = MCAE(
        c_model,
        SLD(LogisticRegression()),
        confidence="entropy",
    ).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def bin_sld_gs(c_model, validation, protocol) -> EvaluationReport:
    v_train, v_val = validation.split_stratified(0.6, random_state=0)
    model = BQAE(c_model, SLD(LogisticRegression()))
    est = GridSearchAE(
        model=model,
        param_grid={
            "q__classifier__C": np.logspace(-3, 3, 7),
            "q__classifier__class_weight": [None, "balanced"],
            "q__recalib": [None, "bcts", "vs"],
            "confidence": [None, "max_conf", "entropy"],
        },
        refit=False,
        protocol=UPP(v_val, repeats=100),
        verbose=True,
    ).fit(v_train)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def mul_sld_gs(c_model, validation, protocol) -> EvaluationReport:
    v_train, v_val = validation.split_stratified(0.6, random_state=0)
    model = MCAE(c_model, SLD(LogisticRegression()))
    est = GridSearchAE(
        model=model,
        param_grid={
            "q__classifier__C": np.logspace(-3, 3, 7),
            "q__classifier__class_weight": [None, "balanced"],
            "q__recalib": [None, "bcts", "vs"],
            "confidence": [None, "max_conf", "entropy"],
        },
        refit=False,
        protocol=UPP(v_val, repeats=100),
        verbose=True,
    ).fit(v_train)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def bin_sld_gsq(c_model, validation, protocol) -> EvaluationReport:
    est = BQAEgsq(
        c_model,
        SLD(LogisticRegression()),
        param_grid={
            "classifier__C": np.logspace(-3, 3, 7),
            "classifier__class_weight": [None, "balanced"],
            "recalib": [None, "bcts", "vs"],
        },
        refit=False,
        verbose=False,
    ).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def mul_sld_gsq(c_model, validation, protocol) -> EvaluationReport:
    est = MCAEgsq(
        c_model,
        SLD(LogisticRegression()),
        param_grid={
            "classifier__C": np.logspace(-3, 3, 7),
            "classifier__class_weight": [None, "balanced"],
            "recalib": [None, "bcts", "vs"],
        },
        refit=False,
        verbose=False,
    ).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def bin_pacc(c_model, validation, protocol) -> EvaluationReport:
    est = BQAE(c_model, PACC(LogisticRegression())).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def mul_pacc(c_model, validation, protocol) -> EvaluationReport:
    est = MCAE(c_model, PACC(LogisticRegression())).fit(validation)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def bin_pacc_gs(c_model, validation, protocol) -> EvaluationReport:
    v_train, v_val = validation.split_stratified(0.6, random_state=0)
    model = BQAE(c_model, PACC(LogisticRegression()))
    est = GridSearchAE(
        model=model,
        param_grid={
            "q__classifier__C": np.logspace(-3, 3, 7),
            "q__classifier__class_weight": [None, "balanced"],
            "confidence": [None, "max_conf", "entropy"],
        },
        refit=False,
        protocol=UPP(v_val, repeats=100),
        verbose=False,
    ).fit(v_train)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )


@method
def mul_pacc_gs(c_model, validation, protocol) -> EvaluationReport:
    v_train, v_val = validation.split_stratified(0.6, random_state=0)
    model = MCAE(c_model, PACC(LogisticRegression()))
    est = GridSearchAE(
        model=model,
        param_grid={
            "q__classifier__C": np.logspace(-3, 3, 7),
            "q__classifier__class_weight": [None, "balanced"],
            "confidence": [None, "max_conf", "entropy"],
        },
        refit=False,
        protocol=UPP(v_val, repeats=100),
        verbose=False,
    ).fit(v_train)
    return evaluation_report(
        estimator=est,
        protocol=protocol,
    )
