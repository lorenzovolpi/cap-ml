from statistics import mean

import numpy as np
import sklearn.metrics as metrics
from quapy.data import LabelledCollection
from quapy.protocol import (
    AbstractStochasticSeededProtocol,
    OnLabelledCollectionProtocol,
)
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate

import elsahar19_rca.rca as rca
import garg22_ATC.ATC_helper as atc
import guillory21_doc.doc as doc
import jiang18_trustscore.trustscore as trustscore

from .report import EvaluationReport


def kfcv(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)

    scoring = ["accuracy", "f1_macro"]
    scores = cross_validate(c_model, validation.X, validation.y, scoring=scoring)
    acc_score = mean(scores["test_accuracy"])
    f1_score = mean(scores["test_f1_macro"])

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    report = EvaluationReport(name="kfcv")
    for test in protocol():
        test_preds = c_model_predict(test.X)
        meta_acc = abs(acc_score - metrics.accuracy_score(test.y, test_preds))
        meta_f1 = abs(f1_score - metrics.f1_score(test.y, test_preds))
        report.append_row(
            test.prevalence(),
            acc_score=acc_score,
            f1_score=f1_score,
            acc=meta_acc,
            f1=meta_f1,
        )

    return report


def reference(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
):
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")
    c_model_predict = getattr(c_model, "predict_proba")
    report = EvaluationReport(name="ref")
    for test in protocol():
        test_probs = c_model_predict(test.X)
        test_preds = np.argmax(test_probs, axis=-1)
        report.append_row(
            test.prevalence(),
            acc_score=metrics.accuracy_score(test.y, test_preds),
            f1_score=metrics.f1_score(test.y, test_preds),
        )

    return report


def atc_mc(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)

    ## Load ID validation data probs and labels
    val_probs, val_labels = c_model_predict(validation.X), validation.y

    ## score function, e.g., negative entropy or argmax confidence
    val_scores = atc.get_max_conf(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)
    _, atc_thres = atc.find_ATC_threshold(val_scores, val_labels == val_preds)

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    report = EvaluationReport(name="atc_mc")
    for test in protocol():
        ## Load OOD test data probs
        test_probs = c_model_predict(test.X)
        test_preds = np.argmax(test_probs, axis=-1)
        test_scores = atc.get_max_conf(test_probs)
        atc_accuracy = atc.get_ATC_acc(atc_thres, test_scores)
        meta_acc = abs(atc_accuracy - metrics.accuracy_score(test.y, test_preds))
        f1_score = atc.get_ATC_f1(atc_thres, test_scores, test_probs)
        meta_f1 = abs(f1_score - metrics.f1_score(test.y, test_preds))
        report.append_row(
            test.prevalence(),
            acc=meta_acc,
            acc_score=atc_accuracy,
            f1_score=f1_score,
            f1=meta_f1,
        )

    return report


def atc_ne(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)

    ## Load ID validation data probs and labels
    val_probs, val_labels = c_model_predict(validation.X), validation.y

    ## score function, e.g., negative entropy or argmax confidence
    val_scores = atc.get_entropy(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)
    _, atc_thres = atc.find_ATC_threshold(val_scores, val_labels == val_preds)

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    report = EvaluationReport(name="atc_ne")
    for test in protocol():
        ## Load OOD test data probs
        test_probs = c_model_predict(test.X)
        test_preds = np.argmax(test_probs, axis=-1)
        test_scores = atc.get_entropy(test_probs)
        atc_accuracy = atc.get_ATC_acc(atc_thres, test_scores)
        meta_acc = abs(atc_accuracy - metrics.accuracy_score(test.y, test_preds))
        f1_score = atc.get_ATC_f1(atc_thres, test_scores, test_probs)
        meta_f1 = abs(f1_score - metrics.f1_score(test.y, test_preds))
        report.append_row(
            test.prevalence(),
            acc=meta_acc,
            acc_score=atc_accuracy,
            f1_score=f1_score,
            f1=meta_f1,
        )

    return report


def trust_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    test: LabelledCollection,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)

    test_pred = c_model_predict(test.X)

    trust_model = trustscore.TrustScore()
    trust_model.fit(validation.X, validation.y)

    return trust_model.get_score(test.X, test_pred)


def doc_feat(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)

    val_probs, val_labels = c_model_predict(validation.X), validation.y
    val_scores = np.max(val_probs, axis=-1)
    val_preds = np.argmax(val_probs, axis=-1)
    v1acc = np.mean(val_preds == val_labels) * 100

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    report = EvaluationReport(name="doc_feat")
    for test in protocol():
        test_probs = c_model_predict(test.X)
        test_preds = np.argmax(test_probs, axis=-1)
        test_scores = np.max(test_probs, axis=-1)
        score = (v1acc + doc.get_doc(val_scores, test_scores)) / 100.0
        meta_acc = abs(score - metrics.accuracy_score(test.y, test_preds))
        report.append_row(test.prevalence(), acc=meta_acc, acc_score=score)

    return report


def rca_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)
    val_pred1 = c_model_predict(validation.X)

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    report = EvaluationReport(name="rca")
    for test in protocol():
        try:
            test_pred = c_model_predict(test.X)
            c_model2 = rca.clone_fit(c_model, test.X, test_pred)
            c_model2_predict = getattr(c_model2, predict_method)
            val_pred2 = c_model2_predict(validation.X)
            rca_score = 1.0 - rca.get_score(val_pred1, val_pred2, validation.y)
            meta_score = abs(rca_score - metrics.accuracy_score(test.y, test_pred))
            report.append_row(test.prevalence(), acc=meta_score, acc_score=rca_score)
        except ValueError:
            report.append_row(
                test.prevalence(), acc=float("nan"), acc_score=float("nan")
            )

    return report


def rca_star_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)
    validation1, validation2 = validation.split_stratified(
        train_prop=0.5, random_state=0
    )
    val1_pred = c_model_predict(validation1.X)
    c_model1 = rca.clone_fit(c_model, validation1.X, val1_pred)
    c_model1_predict = getattr(c_model1, predict_method)
    val2_pred1 = c_model1_predict(validation2.X)

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    report = EvaluationReport(name="rca_star")
    for test in protocol():
        try:
            test_pred = c_model_predict(test.X)
            c_model2 = rca.clone_fit(c_model, test.X, test_pred)
            c_model2_predict = getattr(c_model2, predict_method)
            val2_pred2 = c_model2_predict(validation2.X)
            rca_star_score = 1.0 - rca.get_score(val2_pred1, val2_pred2, validation2.y)
            meta_score = abs(rca_star_score - metrics.accuracy_score(test.y, test_pred))
            report.append_row(
                test.prevalence(), acc=meta_score, acc_score=rca_star_score
            )
        except ValueError:
            report.append_row(
                test.prevalence(), acc=float("nan"), acc_score=float("nan")
            )

    return report
