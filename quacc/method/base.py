import math
from abc import abstractmethod
from copy import deepcopy
from typing import List

import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import BaseQuantifier
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator

from quacc.data import ExtendedCollection


class BaseAccuracyEstimator(BaseQuantifier):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseQuantifier,
    ):
        self.__check_classifier(classifier)
        self.quantifier = quantifier

    def __check_classifier(self, classifier):
        if not hasattr(classifier, "predict_proba"):
            raise ValueError(
                f"Passed classifier {classifier.__class__.__name__} cannot predict probabilities."
            )
        self.classifier = classifier

    def extend(self, coll: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        if not pred_proba:
            pred_proba = self.classifier.predict_proba(coll.X)
        return ExtendedCollection.extend_collection(coll, pred_proba)

    @abstractmethod
    def fit(self, train: LabelledCollection | ExtendedCollection):
        ...

    @abstractmethod
    def estimate(self, instances, ext=False) -> np.ndarray:
        ...


class MultiClassAccuracyEstimator(BaseAccuracyEstimator):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseQuantifier,
    ):
        super().__init__(classifier, quantifier)
        self.e_train = None

    def fit(self, train: LabelledCollection):
        pred_probs = self.classifier.predict_proba(train.X)
        self.e_train = ExtendedCollection.extend_collection(train, pred_probs)

        self.quantifier.fit(self.e_train)

        return self

    def estimate(self, instances, ext=False) -> np.ndarray:
        e_inst = instances
        if not ext:
            pred_prob = self.classifier.predict_proba(instances)
            e_inst = ExtendedCollection.extend_instances(instances, pred_prob)

        estim_prev = self.quantifier.quantify(e_inst)
        return self._check_prevalence_classes(estim_prev)

    def _check_prevalence_classes(self, estim_prev) -> np.ndarray:
        estim_classes = self.quantifier.classes_
        true_classes = self.e_train.classes_
        for _cls in true_classes:
            if _cls not in estim_classes:
                estim_prev = np.insert(estim_prev, _cls, [0.0], axis=0)
        return estim_prev


class BinaryQuantifierAccuracyEstimator(BaseAccuracyEstimator):
    def __init__(self, classifier: BaseEstimator, quantifier: BaseAccuracyEstimator):
        super().__init__(classifier, quantifier)
        self.quantifiers = []
        self.e_trains = []

    def fit(self, train: LabelledCollection | ExtendedCollection):
        pred_probs = self.classifier.predict_proba(train.X)
        self.e_train = ExtendedCollection.extend_collection(train, pred_probs)

        self.n_classes = self.e_train.n_classes
        self.e_trains = self.e_train.split_by_pred()
        self.quantifiers = [deepcopy(self.quantifier) for _ in self.e_trains]

        self.quantifiers = []
        for train in self.e_trains:
            quant = deepcopy(self.quantifier)
            quant.fit(train)
            self.quantifiers.append(quant)

    def estimate(self, instances, ext=False):
        # TODO: test
        e_inst = instances
        if not ext:
            pred_prob = self.classifier.predict_proba(instances)
            e_inst = ExtendedCollection.extend_instances(instances, pred_prob)

        _ncl = int(math.sqrt(self.n_classes))
        s_inst, norms = ExtendedCollection.split_inst_by_pred(_ncl, e_inst)
        estim_prevs = self._quantify_helper(s_inst, norms)

        estim_prev = np.array([prev_row for prev_row in zip(*estim_prevs)]).flatten()
        return estim_prev

    def _quantify_helper(
        self,
        s_inst: List[np.ndarray | csr_matrix],
        norms: List[float],
    ):
        estim_prevs = []
        for quant, inst, norm in zip(self.quantifiers, s_inst, norms):
            if inst.shape[0] > 0:
                estim_prevs.append(quant.quantify(inst) * norm)
            else:
                estim_prevs.append(np.asarray([0.0, 0.0]))

        return estim_prevs


BAE = BaseAccuracyEstimator
MCAE = MultiClassAccuracyEstimator
BQAE = BinaryQuantifierAccuracyEstimator
