from statistics import mean
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from quapy.data import LabelledCollection
from garg22_ATC.ATC_helper import (
    find_ATC_threshold,
    get_ATC_acc,
    get_entropy,
    get_max_conf,
)
import numpy as np
from jiang18_trustscore.trustscore import TrustScore


def kfcv(c_model: BaseEstimator, validation: LabelledCollection) -> Dict:
    scoring = ["f1_macro"]
    scores = cross_validate(c_model, validation.X, validation.y, scoring=scoring)
    return {"f1_score": mean(scores["test_f1_macro"])}


def ATC_MC(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    test: LabelledCollection,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)

    ## Load ID validation data probs and labels
    val_probs, val_labels = c_model_predict(validation.X), validation.y

    ## Load OOD test data probs
    test_probs = c_model_predict(test.X)

    ## score function, e.g., negative entropy or argmax confidence
    val_scores = get_max_conf(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)

    test_scores = get_max_conf(test_probs)

    _, ATC_thres = find_ATC_threshold(val_scores, val_labels == val_preds)
    ATC_accuracy = get_ATC_acc(ATC_thres, test_scores)

    return {
        "true_acc": 100 * np.mean(np.argmax(test_probs, axis=-1) == test.y),
        "pred_acc": ATC_accuracy,
    }


def ATC_NE(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    test: LabelledCollection,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)

    ## Load ID validation data probs and labels
    val_probs, val_labels = c_model_predict(validation.X), validation.y

    ## Load OOD test data probs
    test_probs = c_model_predict(test.X)

    ## score function, e.g., negative entropy or argmax confidence
    val_scores = get_entropy(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)

    test_scores = get_entropy(test_probs)

    _, ATC_thres = find_ATC_threshold(val_scores, val_labels == val_preds)
    ATC_accuracy = get_ATC_acc(ATC_thres, test_scores)

    return {
        "true_acc": 100 * np.mean(np.argmax(test_probs, axis=-1) == test.y),
        "pred_acc": ATC_accuracy,
    }


def trust_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    test: LabelledCollection,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)

    test_pred = c_model_predict(test.X)

    trust_model = TrustScore()
    trust_model.fit(validation.X, validation.y)

    return trust_model.get_score(test.X, test_pred)

