from ast import get_docstring
from statistics import mean
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from quapy.data import LabelledCollection
import garg22_ATC.ATC_helper as atc
import numpy as np
import jiang18_trustscore.trustscore as trustscore
import guillory21_doc.doc as doc


def kfcv(c_model: BaseEstimator, validation: LabelledCollection) -> Dict:
    scoring = ["f1_macro"]
    scores = cross_validate(c_model, validation.X, validation.y, scoring=scoring)
    return {"f1_score": mean(scores["test_f1_macro"])}


def atc_mc(
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
    val_scores = atc.get_max_conf(val_probs)
   #pred_idxv1           #calib_probsv1/probsv1
    val_preds = np.argmax(val_probs, axis=-1)
   #pred_probs_new            #probs_new
    test_scores = atc.get_max_conf(test_probs)
                                     #pred_probsv1  #labelsv1     #pred_idxv1
    _, atc_thres = atc.find_ATC_threshold(val_scores,    val_labels == val_preds)
                              #calib_thres_balance #pred_probs_new
    atc_accuracy = atc.get_ATC_acc(atc_thres,           test_scores)

    return {
        "true_acc": 100 * np.mean(np.argmax(test_probs, axis=-1) == test.y),
        "pred_acc": atc_accuracy,
    }


def atc_ne(
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
    val_scores = atc.get_entropy(val_probs)
    val_preds = np.argmax(val_probs, axis=-1)

    test_scores = atc.get_entropy(test_probs)

    _, atc_thres = atc.find_ATC_threshold(val_scores, val_labels == val_preds)
    atc_accuracy = atc.get_ATC_acc(atc_thres, test_scores)

    return {
        "true_acc": 100 * np.mean(np.argmax(test_probs, axis=-1) == test.y),
        "pred_acc": atc_accuracy,
    }


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
    test: LabelledCollection,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)

    val_probs, val_labels = c_model_predict(validation.X), validation.y
    test_probs = c_model_predict(test.X)
    val_scores = np.max(val_probs, axis=-1)
    test_scores = np.max(test_probs, axis=-1)
    val_preds = np.argmax(val_probs, axis=-1)

    v1acc = np.mean(val_preds == val_labels)*100
    return v1acc + doc.get_doc(val_scores, test_scores)
