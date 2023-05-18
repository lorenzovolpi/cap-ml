import numpy as np
import scipy.sparse as sp
from quapy.data import LabelledCollection
from quapy.method.base import BaseQuantifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict


def _get_ex_class(classes, true_class, pred_class):
    return true_class * classes + pred_class


def _extend_collection(coll, pred_prob):
    n_classes = coll.n_classes

    # n_X = [ X | predicted probs. ]
    if isinstance(coll.X, sp.csr_matrix):
        pred_prob_csr = sp.csr_matrix(pred_prob)
        n_x = sp.hstack([coll.X, pred_prob_csr])
    elif isinstance(coll.X, np.ndarray):
        n_x = np.concatenate((coll.X, pred_prob), axis=1)
    else:
        raise ValueError("Unsupported matrix format")

    # n_y = (exptected y, predicted y)
    n_y = []
    for i, true_class in enumerate(coll.y):
        pred_class = pred_prob[i].argmax(axis=0)
        n_y.append(_get_ex_class(n_classes, true_class, pred_class))

    return LabelledCollection(n_x, np.asarray(n_y), [*range(0, n_classes * n_classes)])


class AccuracyQuantifier:
    def __init__(self, model: BaseEstimator, q_model: BaseQuantifier):
        self.model = model
        self.q_model = q_model

    def fit(self, train: LabelledCollection):
        self._train = train
        self.model.fit(*self._train.Xy)
        self._pred_prob_train = cross_val_predict(
            self.model, *self._train.Xy, method="predict_proba"
        )
        self._e_train = _extend_collection(self._train, self._pred_prob_train)

        self.q_model.fit(self._e_train)
