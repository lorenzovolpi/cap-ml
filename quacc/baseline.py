from statistics import mean
from typing import Dict

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate

import elsahar19_rca.rca as rca
import garg22_ATC.ATC_helper as atc
import guillory21_doc.doc as doc
import jiang18_trustscore.trustscore as trustscore
import lipton_bbse.labelshift as bbse


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
    val_preds = np.argmax(val_probs, axis=-1)
    test_scores = atc.get_max_conf(test_probs)

    _, atc_thres = atc.find_ATC_threshold(val_scores, val_labels == val_preds)
    atc_accuracy = atc.get_ATC_acc(atc_thres, test_scores)

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

    v1acc = np.mean(val_preds == val_labels) * 100
    return v1acc + doc.get_doc(val_scores, test_scores)


def rca_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    test: LabelledCollection,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)
    test_pred = c_model_predict(test.X)
    c_model2 = rca.clone_fit(test.X, test_pred)
    c_model2_predict = getattr(c_model2, predict_method)

    val_pred1 = c_model_predict(validation.X)
    val_pred2 = c_model2_predict(validation.X)

    return rca.get_score(val_pred1, val_pred2, validation.y)

def rca_star_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    test: LabelledCollection,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)
    validation1, validation2 = validation.split_stratified(train_prop=0.5)
    test_pred = c_model_predict(test.X)
    val1_pred = c_model_predict(validation1.X)
    c_model1 = rca.clone_fit(validation1.X, val1_pred)
    c_model2 = rca.clone_fit(test.X, test_pred)
    c_model1_predict = getattr(c_model1, predict_method)
    c_model2_predict = getattr(c_model2, predict_method)

    val2_pred1 = c_model1_predict(validation2.X)
    val2_pred2 = c_model2_predict(validation2.X)

    return rca.get_score(val2_pred1, val2_pred2, validation2.y)

    
def bbse_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    test: LabelledCollection,
    predict_method="predict_proba",
):

    c_model_predict = getattr(c_model, predict_method)
    val_probs, val_labels = c_model_predict(validation.X), validation.y
    test_probs = c_model_predict(test.X)

    wt = bbse.estimate_labelshift_ratio(val_labels, val_probs, test_probs, 2)
    estim_prev = bbse.estimate_target_dist(wt, val_labels, 2)
    true_prev = test.prevalence()
    return qp.error.ae(true_prev, estim_prev)
