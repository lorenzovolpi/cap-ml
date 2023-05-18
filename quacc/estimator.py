import numpy as np
import scipy.sparse as sp
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict

from .data import ExtendedCollection


def _check_prevalence_classes(true_classes, estim_classes, estim_prev):
    for _cls in true_classes:
        if _cls not in estim_classes:
            estim_prev = np.insert(estim_prev, _cls, [0.0], axis=0)
    return estim_prev


def _get_ex_class(classes, true_class, pred_class):
    return true_class * classes + pred_class


def _extend_instances(instances, pred_proba):
    if isinstance(instances, sp.csr_matrix):
        _pred_proba = sp.csr_matrix(pred_proba)
        n_x = sp.hstack([instances, _pred_proba])
    elif isinstance(instances, np.ndarray):
        n_x = np.concatenate((instances, pred_proba), axis=1)
    else:
        raise ValueError("Unsupported matrix format")

    return n_x


def _extend_collection(base: LabelledCollection, pred_proba) -> ExtendedCollection:
    n_classes = base.n_classes

    # n_X = [ X | predicted probs. ]
    n_x = _extend_instances(base.X, pred_proba)

    # n_y = (exptected y, predicted y)
    pred = np.asarray([prob.argmax(axis=0) for prob in pred_proba])
    n_y = np.asarray(
        [
            _get_ex_class(n_classes, true_class, pred_class)
            for (true_class, pred_class) in zip(base.y, pred)
        ]
    )

    return ExtendedCollection(n_x, n_y, [*range(0, n_classes * n_classes)])


class AccuracyEstimator:
    def __init__(self, model: BaseEstimator, q_model: BaseQuantifier):
        self.model = model
        self.q_model = q_model
        self.e_train = None

    def extend(self, base: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        if not pred_proba:
            pred_proba = self.model.predict_proba(base.X)
        return _extend_collection(base, pred_proba)

    def fit(self, train: LabelledCollection | ExtendedCollection):
        # check if model is fit
        # self.model.fit(*train.Xy)
        if isinstance(train, LabelledCollection):
            pred_prob_train = cross_val_predict(
                self.model, train.Xy, method="predict_proba"
            )

            self.e_train = _extend_collection(train, pred_prob_train)
        else:
            self.e_train = train

        self.q_model.fit(self.e_train)

    def estimate(self, instances, ext=False):
        if not ext:
            pred_prob = self.model.predict_proba(instances)
            e_inst = _extend_instances(instances, pred_prob)
        else:
            e_inst = instances

        estim_prev = self.q_model.quantify(e_inst)

        return _check_prevalence_classes(
            e_inst.classes_, self.q_model.classes_, estim_prev
        )
