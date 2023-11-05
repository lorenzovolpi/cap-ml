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
        confidence=None,
    ):
        self.__check_classifier(classifier)
        self.quantifier = quantifier
        self.confidence = confidence

    def __check_classifier(self, classifier):
        if not hasattr(classifier, "predict_proba"):
            raise ValueError(
                f"Passed classifier {classifier.__class__.__name__} cannot predict probabilities."
            )
        self.classifier = classifier

    def __get_confidence(self):
        def max_conf(probas):
            _mc = np.max(probas, axis=-1)
            _min = 1.0 / probas.shape[1]
            _norm_mc = (_mc - _min) / (1.0 - _min)
            return _norm_mc

        def entropy(probas):
            _ent = np.sum(np.multiply(probas, np.log(probas + 1e-20)), axis=1)
            return _ent

        if self.confidence is None:
            return None

        __confs = {
            "max_conf": max_conf,
            "entropy": entropy,
        }
        return __confs.get(self.confidence, None)

    def __get_ext(self, pred_proba):
        _ext = pred_proba
        _f_conf = self.__get_confidence()
        if _f_conf is not None:
            _confs = _f_conf(pred_proba).reshape((len(pred_proba), 1))
            _ext = np.concatenate((_confs, pred_proba), axis=1)

        return _ext

    def extend(self, coll: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(coll.X)

        _ext = self.__get_ext(pred_proba)
        return ExtendedCollection.extend_collection(coll, pred_proba=_ext)

    def _extend_instances(self, instances: np.ndarray | csr_matrix, pred_proba=None):
        if pred_proba is None:
            pred_proba = self.classifier.predict_proba(instances)

        _ext = self.__get_ext(pred_proba)
        return ExtendedCollection.extend_instances(instances, _ext)

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
        confidence: str = None,
    ):
        super().__init__(
            classifier=classifier,
            quantifier=quantifier,
            confidence=confidence,
        )
        self.e_train = None

    def fit(self, train: LabelledCollection):
        self.e_train = self.extend(train)

        self.quantifier.fit(self.e_train)

        return self

    def estimate(self, instances, ext=False) -> np.ndarray:
        e_inst = instances if ext else self._extend_instances(instances)

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
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseAccuracyEstimator,
        confidence: str = None,
    ):
        super().__init__(
            classifier=classifier,
            quantifier=quantifier,
            confidence=confidence,
        )
        self.quantifiers = []
        self.e_trains = []

    def fit(self, train: LabelledCollection | ExtendedCollection):
        self.e_train = self.extend(train)

        self.n_classes = self.e_train.n_classes
        self.e_trains = self.e_train.split_by_pred()

        self.quantifiers = []
        for train in self.e_trains:
            quant = deepcopy(self.quantifier)
            quant.fit(train)
            self.quantifiers.append(quant)

        return self

    def estimate(self, instances, ext=False):
        # TODO: test
        e_inst = instances if ext else self._extend_instances(instances)

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
