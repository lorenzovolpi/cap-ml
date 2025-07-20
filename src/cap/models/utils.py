from collections import defaultdict
from typing import List, override

import numpy as np
import scipy
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from sklearn.base import BaseEstimator


def get_posteriors_from_h(h, X):
    if hasattr(h, "predict_proba"):
        P = h.predict_proba(X)
    else:
        n_classes = len(h.classes_)
        dec_scores = h.decision_function(X)
        if n_classes == 1:
            dec_scores = np.vstack([-dec_scores, dec_scores]).T
        P = scipy.special.softmax(dec_scores, axis=1)
    return P


def max_conf(P, keepdims=False):
    mc = P.max(axis=1, keepdims=keepdims)
    return mc


def neg_entropy(P, keepdims=False):
    ne = scipy.stats.entropy(P, axis=1)
    if keepdims:
        ne = ne.reshape(-1, 1)
    return ne


def max_inverse_softmax(P, keepdims=False):
    P = smooth(P, epsilon=1e-12, axis=1)
    lgP = np.log(P)
    mis = np.max(lgP - lgP.mean(axis=1, keepdims=True), axis=1, keepdims=keepdims)
    return mis


def smooth(prevalences, epsilon=1e-5, axis=None):
    """
    Smooths a prevalence vector.

    :param prevalences: np.ndarray
    :param epsilon: float, a small quantity (default 1E-5)
    :return: smoothed prevalence vector
    """
    prevalences = prevalences + epsilon
    prevalences /= prevalences.sum(axis=axis, keepdims=axis is not None)
    return prevalences


class OracleQuantifier(AggregativeQuantifier):
    def __init__(self, ldata: List[LabelledCollection], classifier: BaseEstimator = None):
        self.hash_K = len(ldata)
        self.ldata = defaultdict(lambda: [])
        for ui in ldata:
            self.ldata[self._get_hash(ui.X)].append(ui)
        self.classifier = classifier

    def _get_hash(self, X):
        return int(np.around(np.abs(X).sum() * self.hash_K, decimals=0))

    @override
    def fit(self, data: LabelledCollection, fit_classifier=True, val_split=None):
        pass

    @override
    def classifier_fit_predict(self, data: LabelledCollection, fit_classifier=True, predict_on=None):
        return []

    @override
    def aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        pass

    @override
    def aggregate(self, classif_predictions: np.ndarray):
        pass

    @override
    def quantify(self, instances):
        _hash = self._get_hash(instances)
        lcs = self.ldata[_hash]
        eq_idx = [np.all(instances == lc.X) for lc in lcs].index(True)
        return lcs[eq_idx].prevalence()
