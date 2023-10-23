import math
from abc import abstractmethod

import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import CC, SLD
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from quacc.data import ExtendedCollection


class AccuracyEstimator:
    def extend(self, base: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        if not pred_proba:
            pred_proba = self.c_model.predict_proba(base.X)
        return ExtendedCollection.extend_collection(base, pred_proba), pred_proba

    @abstractmethod
    def fit(self, train: LabelledCollection | ExtendedCollection):
        ...

    @abstractmethod
    def estimate(self, instances, ext=False):
        ...


class MulticlassAccuracyEstimator(AccuracyEstimator):
    def __init__(self, c_model: BaseEstimator, q_model="SLD", **kwargs):
        self.c_model = c_model
        if q_model == "SLD":
            available_args = ["recalib"]
            sld_args = {k: v for k, v in kwargs.items() if k in available_args}
            self.q_model = SLD(LogisticRegression(), **sld_args)
        elif q_model == "CC":
            self.q_model = CC(LogisticRegression())

        self.e_train = None

    def fit(self, train: LabelledCollection | ExtendedCollection):
        # check if model is fit
        # self.model.fit(*train.Xy)
        if isinstance(train, LabelledCollection):
            pred_prob_train = cross_val_predict(
                self.c_model, *train.Xy, method="predict_proba"
            )

            self.e_train = ExtendedCollection.extend_collection(train, pred_prob_train)
        else:
            self.e_train = train

        self.q_model.fit(self.e_train)

    def estimate(self, instances, ext=False):
        if not ext:
            pred_prob = self.c_model.predict_proba(instances)
            e_inst = ExtendedCollection.extend_instances(instances, pred_prob)
        else:
            e_inst = instances

        estim_prev = self.q_model.quantify(e_inst)

        return self._check_prevalence_classes(
            self.e_train.classes_, self.q_model.classes_, estim_prev
        )

    def _check_prevalence_classes(self, true_classes, estim_classes, estim_prev):
        for _cls in true_classes:
            if _cls not in estim_classes:
                estim_prev = np.insert(estim_prev, _cls, [0.0], axis=0)
        return estim_prev


class BinaryQuantifierAccuracyEstimator(AccuracyEstimator):
    def __init__(self, c_model: BaseEstimator, q_model="SLD", **kwargs):
        self.c_model = c_model
        if q_model == "SLD":
            available_args = ["recalib"]
            sld_args = {k: v for k, v in kwargs.items() if k in available_args}
            self.q_model_0 = SLD(LogisticRegression(), **sld_args)
            self.q_model_1 = SLD(LogisticRegression(), **sld_args)
        elif q_model == "CC":
            self.q_model_0 = CC(LogisticRegression())
            self.q_model_1 = CC(LogisticRegression())

        self.e_train = None

    def fit(self, train: LabelledCollection | ExtendedCollection):
        # check if model is fit
        # self.model.fit(*train.Xy)
        if isinstance(train, LabelledCollection):
            pred_prob_train = cross_val_predict(
                self.c_model, *train.Xy, method="predict_proba"
            )

            self.e_train = ExtendedCollection.extend_collection(train, pred_prob_train)
        elif isinstance(train, ExtendedCollection):
            self.e_train = train

        self.n_classes = self.e_train.n_classes
        [e_train_0, e_train_1] = self.e_train.split_by_pred()

        self.q_model_0.fit(e_train_0)
        self.q_model_1.fit(e_train_1)

    def estimate(self, instances, ext=False):
        # TODO: test
        if not ext:
            pred_prob = self.c_model.predict_proba(instances)
            e_inst = ExtendedCollection.extend_instances(instances, pred_prob)
        else:
            e_inst = instances

        _ncl = int(math.sqrt(self.n_classes))
        s_inst, norms = ExtendedCollection.split_inst_by_pred(_ncl, e_inst)
        [estim_prev_0, estim_prev_1] = [
            self._quantify_helper(inst, norm, q_model)
            for (inst, norm, q_model) in zip(
                s_inst, norms, [self.q_model_0, self.q_model_1]
            )
        ]

        estim_prev = []
        for prev_row in zip(estim_prev_0, estim_prev_1):
            for prev in prev_row:
                estim_prev.append(prev)

        return np.asarray(estim_prev)

    def _quantify_helper(self, inst, norm, q_model):
        if inst.shape[0] > 0:
            return np.asarray(list(map(lambda p: p * norm, q_model.quantify(inst))))
        else:
            return np.asarray([0.0, 0.0])
