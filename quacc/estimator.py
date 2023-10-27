import math
from abc import abstractmethod

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import CC, SLD
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from quacc.data import ExtendedCollection


class AccuracyEstimator:
    def __init__(self):
        self.fit_score = None

    def _gs_params(self, t_val: LabelledCollection):
        return {
            "param_grid": {
                "classifier__C": np.logspace(-3, 3, 7),
                "classifier__class_weight": [None, "balanced"],
                "recalib": [None, "bcts"],
            },
            "protocol": UPP(t_val, repeats=1000),
            "error": qp.error.mae,
            "refit": False,
            "timeout": -1,
            "n_jobs": None,
            "verbose": True,
        }

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


AE = AccuracyEstimator


class MulticlassAccuracyEstimator(AccuracyEstimator):
    def __init__(self, c_model: BaseEstimator, q_model="SLD", gs=False, recalib=None):
        super().__init__()
        self.c_model = c_model
        self._q_model_name = q_model.upper()
        self.e_train = None
        self.gs = gs
        self.recalib = recalib

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

        if self._q_model_name == "SLD":
            if self.gs:
                t_train, t_val = self.e_train.split_stratified(0.6, random_state=0)
                gs_params = self._gs_params(t_val)
                self.q_model = GridSearchQ(
                    SLD(LogisticRegression()),
                    **gs_params,
                )
                self.q_model.fit(t_train)
                self.fit_score = self.q_model.best_score_
            else:
                self.q_model = SLD(LogisticRegression(), recalib=self.recalib)
                self.q_model.fit(self.e_train)
        elif self._q_model_name == "CC":
            self.q_model = CC(LogisticRegression())
            self.q_model.fit(self.e_train)

    def estimate(self, instances, ext=False):
        if not ext:
            pred_prob = self.c_model.predict_proba(instances)
            e_inst = ExtendedCollection.extend_instances(instances, pred_prob)
        else:
            e_inst = instances

        estim_prev = self.q_model.quantify(e_inst)

        return self._check_prevalence_classes(
            self.e_train.classes_, self.q_model, estim_prev
        )

    def _check_prevalence_classes(self, true_classes, q_model, estim_prev):
        if isinstance(q_model, GridSearchQ):
            estim_classes = q_model.best_model().classes_
        else:
            estim_classes = q_model.classes_
        for _cls in true_classes:
            if _cls not in estim_classes:
                estim_prev = np.insert(estim_prev, _cls, [0.0], axis=0)
        return estim_prev


class BinaryQuantifierAccuracyEstimator(AccuracyEstimator):
    def __init__(self, c_model: BaseEstimator, q_model="SLD", gs=False, recalib=None):
        super().__init__()
        self.c_model = c_model
        self._q_model_name = q_model.upper()
        self.q_models = []
        self.gs = gs
        self.recalib = recalib
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
        e_trains = self.e_train.split_by_pred()

        if self._q_model_name == "SLD":
            fit_scores = []
            for e_train in e_trains:
                if self.gs:
                    t_train, t_val = e_train.split_stratified(0.6, random_state=0)
                    gs_params = self._gs_params(t_val)
                    q_model = GridSearchQ(
                        SLD(LogisticRegression()),
                        **gs_params,
                    )
                    q_model.fit(t_train)
                    fit_scores.append(q_model.best_score_)
                    self.q_models.append(q_model)
                else:
                    q_model = SLD(LogisticRegression(), recalib=self.recalib)
                    q_model.fit(e_train)
                    self.q_models.append(q_model)

            if self.gs:
                self.fit_score = np.mean(fit_scores)

        elif self._q_model_name == "CC":
            for e_train in e_trains:
                q_model = CC(LogisticRegression())
                q_model.fit(e_train)
                self.q_models.append(q_model)

    def estimate(self, instances, ext=False):
        # TODO: test
        if not ext:
            pred_prob = self.c_model.predict_proba(instances)
            e_inst = ExtendedCollection.extend_instances(instances, pred_prob)
        else:
            e_inst = instances

        _ncl = int(math.sqrt(self.n_classes))
        s_inst, norms = ExtendedCollection.split_inst_by_pred(_ncl, e_inst)
        estim_prevs = [
            self._quantify_helper(inst, norm, q_model)
            for (inst, norm, q_model) in zip(s_inst, norms, self.q_models)
        ]

        estim_prev = []
        for prev_row in zip(*estim_prevs):
            for prev in prev_row:
                estim_prev.append(prev)

        return np.asarray(estim_prev)

    def _quantify_helper(self, inst, norm, q_model):
        if inst.shape[0] > 0:
            return np.asarray(list(map(lambda p: p * norm, q_model.quantify(inst))))
        else:
            return np.asarray([0.0, 0.0])
