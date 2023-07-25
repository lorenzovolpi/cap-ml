from abc import abstractmethod
import math

import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import SLD
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from quacc.data import ExtendedCollection as EC


def _check_prevalence_classes(true_classes, estim_classes, estim_prev):
    for _cls in true_classes:
        if _cls not in estim_classes:
            estim_prev = np.insert(estim_prev, _cls, [0.0], axis=0)
    return estim_prev


class AccuracyEstimator:
    def extend(self, base: LabelledCollection, pred_proba=None) -> EC:
        if not pred_proba:
            pred_proba = self.model.predict_proba(base.X)
        return EC.extend_collection(base, pred_proba)

    @abstractmethod
    def fit(self, train: LabelledCollection | EC):
        ...

    @abstractmethod
    def estimate(self, instances, ext=False):
        ...


class MulticlassAccuracyEstimator(AccuracyEstimator):
    def __init__(self, c_model: BaseEstimator):
        self.c_model = c_model
        self.q_model = SLD(LogisticRegression())
        self.e_train = None

    def fit(self, train: LabelledCollection | EC):
        # check if model is fit
        # self.model.fit(*train.Xy)
        if isinstance(train, LabelledCollection):
            pred_prob_train = cross_val_predict(
                self.c_model, *train.Xy, method="predict_proba"
            )

            self.e_train = EC.extend_collection(train, pred_prob_train)
        else:
            self.e_train = train

        self.q_model.fit(self.e_train)

    def estimate(self, instances, ext=False):
        if not ext:
            pred_prob = self.c_model.predict_proba(instances)
            e_inst = EC.extend_instances(instances, pred_prob)
        else:
            e_inst = instances

        estim_prev = self.q_model.quantify(e_inst)

        return _check_prevalence_classes(
            self.e_train.classes_, self.q_model.classes_, estim_prev
        )


class BinaryQuantifierAccuracyEstimator(AccuracyEstimator):
    def __init__(self, c_model: BaseEstimator):
        self.c_model = c_model
        self.q_model_0 = SLD(LogisticRegression())
        self.q_model_1 = SLD(LogisticRegression())
        self.e_train: EC = None

    def fit(self, train: LabelledCollection | EC):
        # check if model is fit
        # self.model.fit(*train.Xy)
        if isinstance(train, LabelledCollection):
            pred_prob_train = cross_val_predict(
                self.c_model, *train.Xy, method="predict_proba"
            )

            self.e_train = EC.extend_collection(train, pred_prob_train)
        else:
            self.e_train = train

        [e_train_0, e_train_1] = self.e_train.split_by_pred()

        self.q_model_0.fit(self.e_train_0)
        self.q_model_1.fit(self.e_train_1)

    def estimate(self, instances, ext=False):
        # TODO: test
        if not ext:
            pred_prob = self.c_model.predict_proba(instances)
            e_inst = EC.extend_instances(instances, pred_prob)
        else:
            e_inst = instances

        _ncl = int(math.sqrt(self.e_train.n_classes))
        [e_inst_0, e_inst_1] = [
            e_inst[ind] for ind in EC.split_index_by_pred(_ncl, e_inst)
        ]
        estim_prev_0 = self.q_model_0.quantify(e_inst_0)
        estim_prev_1 = self.q_model_1.quantify(e_inst_1)

        estim_prev = []
        for prev_row in zip(estim_prev_0, estim_prev_1):
            for prev in prev_row:
                estim_prev.append(prev)

        return estim_prev

