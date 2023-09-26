from statistics import mean
from typing import Dict

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from quapy.protocol import (
    AbstractStochasticSeededProtocol,
    OnLabelledCollectionProtocol,
)

import elsahar19_rca.rca as rca
import garg22_ATC.ATC_helper as atc
import guillory21_doc.doc as doc
import jiang18_trustscore.trustscore as trustscore
import lipton_bbse.labelshift as bbse
import pandas as pd
import statistics as stats


def kfcv(c_model: BaseEstimator, validation: LabelledCollection) -> Dict:
    scoring = ["f1_macro"]
    scores = cross_validate(c_model, validation.X, validation.y, scoring=scoring)
    return {"f1_score": mean(scores["test_f1_macro"])}


def avg_groupby_distribution(results):
    def base_prev(s):
        return (s[("base", "F")], s[("base", "T")])

    grouped_list = {}
    for r in results:
        bp = base_prev(r)
        if bp in grouped_list.keys():
            grouped_list[bp].append(r)
        else:
            grouped_list[bp] = [r]

    series = []
    for (fp, tp), r_list in grouped_list.items():
        assert len(r_list) > 0
        r_avg = {}
        r_avg[("base", "F")], r_avg[("base", "T")] = fp, tp
        for pn in [(n1, n2) for ((n1, n2), _) in r_list[0].items() if n1 != "base"]:
            r_avg[pn] = stats.mean(map(lambda r: r[pn], r_list))
        series.append(r_avg)

    return series


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

    cols = [
        ("base", "F"),
        ("base", "T"),
        ("atc_mc", "accuracy"),
    ]
    results = []
    for test in protocol():
        ## Load OOD test data probs
        test_probs = c_model_predict(test.X)
        test_scores = atc.get_max_conf(test_probs)
        atc_accuracy = 1.0 - (atc.get_ATC_acc(atc_thres, test_scores) / 100.0)
        [f_prev, t_prev] = test.prevalence()
        results.append({k: v for k, v in zip(cols, [f_prev, t_prev, atc_accuracy])})

    series = avg_groupby_distribution(results)
    return pd.DataFrame(
        series,
        columns=pd.MultiIndex.from_tuples(cols),
    )


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

    cols = [
        ("base", "F"),
        ("base", "T"),
        ("atc_ne", "accuracy"),
    ]
    results = []
    for test in protocol():
        ## Load OOD test data probs
        test_probs = c_model_predict(test.X)
        test_scores = atc.get_entropy(test_probs)
        atc_accuracy = 1.0 - (atc.get_ATC_acc(atc_thres, test_scores) / 100.0)
        [f_prev, t_prev] = test.prevalence()
        results.append({k: v for k, v in zip(cols, [f_prev, t_prev, atc_accuracy])})

    series = avg_groupby_distribution(results)
    return pd.DataFrame(
        series,
        columns=pd.MultiIndex.from_tuples(cols),
    )


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

    cols = [
        ("base", "F"),
        ("base", "T"),
        ("doc_feat", "score"),
    ]
    results = []
    for test in protocol():
        test_probs = c_model_predict(test.X)
        test_scores = np.max(test_probs, axis=-1)
        score = 1.0 - ((v1acc + doc.get_doc(val_scores, test_scores)) / 100.0)
        [f_prev, t_prev] = test.prevalence()
        results.append({k: v for k, v in zip(cols, [f_prev, t_prev, score])})

    series = avg_groupby_distribution(results)
    return pd.DataFrame(
        series,
        columns=pd.MultiIndex.from_tuples(cols),
    )


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

    cols = [
        ("base", "F"),
        ("base", "T"),
        ("rca", "score"),
    ]
    results = []
    for test in protocol():
        try: 
            [f_prev, t_prev] = test.prevalence()
            test_pred = c_model_predict(test.X)
            c_model2 = rca.clone_fit(c_model, test.X, test_pred)
            c_model2_predict = getattr(c_model2, predict_method)
            val_pred2 = c_model2_predict(validation.X)
            rca_score = rca.get_score(val_pred1, val_pred2, validation.y)
            results.append({k: v for k, v in zip(cols, [f_prev, t_prev, rca_score])})
        except ValueError:
            results.append({k: v for k, v in zip(cols, [f_prev, t_prev, float("nan")])})

    series = avg_groupby_distribution(results)
    return pd.DataFrame(
        series,
        columns=pd.MultiIndex.from_tuples(cols),
    )


def rca_star_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict",
):
    c_model_predict = getattr(c_model, predict_method)
    validation1, validation2 = validation.split_stratified(train_prop=0.5)
    val1_pred = c_model_predict(validation1.X)
    c_model1 = rca.clone_fit(c_model, validation1.X, val1_pred)
    c_model1_predict = getattr(c_model1, predict_method)
    val2_pred1 = c_model1_predict(validation2.X)

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    cols = [
        ("base", "F"),
        ("base", "T"),
        ("rca*", "score"),
    ]
    results = []
    for test in protocol():
        [f_prev, t_prev] = test.prevalence()
        try:
            test_pred = c_model_predict(test.X)
            c_model2 = rca.clone_fit(c_model, test.X, test_pred)
            c_model2_predict = getattr(c_model2, predict_method)
            val2_pred2 = c_model2_predict(validation2.X)
            rca_star_score = rca.get_score(val2_pred1, val2_pred2, validation2.y)
            results.append(
                {k: v for k, v in zip(cols, [f_prev, t_prev, rca_star_score])}
            )
        except ValueError:
            results.append({k: v for k, v in zip(cols, [f_prev, t_prev, float("nan")])})

    series = avg_groupby_distribution(results)
    return pd.DataFrame(
        series,
        columns=pd.MultiIndex.from_tuples(cols),
    )


def bbse_score(
    c_model: BaseEstimator,
    validation: LabelledCollection,
    protocol: AbstractStochasticSeededProtocol,
    predict_method="predict_proba",
):
    c_model_predict = getattr(c_model, predict_method)
    val_probs, val_labels = c_model_predict(validation.X), validation.y

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    cols = [
        ("base", "F"),
        ("base", "T"),
        ("bbse", "score"),
    ]
    results = []
    for test in protocol():
        test_probs = c_model_predict(test.X)
        wt = bbse.estimate_labelshift_ratio(val_labels, val_probs, test_probs, 2)
        estim_prev = bbse.estimate_target_dist(wt, val_labels, 2)[1]
        true_prev = test.prevalence()
        [f_prev, t_prev] = true_prev
        acc = qp.error.ae(true_prev, estim_prev)
        results.append({k: v for k, v in zip(cols, [f_prev, t_prev, acc])})

    series = avg_groupby_distribution(results)
    return pd.DataFrame(
        series,
        columns=pd.MultiIndex.from_tuples(cols),
    )
