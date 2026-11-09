"""Protocol tests for PrediQuant (SQS-S): split of V and consistency of the prevalence estimates."""

import numpy as np
from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from cap.models.confidence import PrediQuant

N_CLASSES = 3
SAMPLE_SIZE = 50


def _data(n, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(N_CLASSES, size=n)
    X = rng.normal(size=(n, 2)) + 2.0 * np.eye(N_CLASSES, 2)[y]
    return X, y


class RecordingPACC(PACC):
    """PACC that records the instances it is trained on and the instances it classifies."""

    def fit(self, X, y):
        self.fit_X = np.asarray(X).copy()
        self.classified_X = []
        return super().fit(X, y)

    def classify(self, X):
        self.classified_X.append(np.asarray(X).copy())
        return super().classify(X)


def _fitted_method():
    X_tr, y_tr = _data(300, seed=0)
    h = LogisticRegression().fit(X_tr, y_tr)
    X_val, y_val = _data(400, seed=1)
    val = LabelledCollection(X_val, y_val, classes=np.arange(N_CLASSES))

    def acc(ct):
        return np.trace(ct) / ct.sum()

    method = PrediQuant(
        acc,
        RecordingPACC(LogisticRegression()),
        alpha=0.5,
        num_samples=20,
        sample_size=SAMPLE_SIZE,
        random_state=0,
    )
    method.fit(val, h.predict_proba(X_val))
    return method, val, h


def _rows(X):
    return {tuple(row) for row in np.asarray(X)}


def test_quantifier_never_sees_the_bag_sampling_subset():
    method, val, _ = _fitted_method()
    fit_rows = _rows(method.q.fit_X)
    # the only classify call made by fit is on V2, the bag-sampling subset
    assert len(method.q.classified_X) == 1
    bag_rows = _rows(method.q.classified_X[0])
    assert fit_rows.isdisjoint(bag_rows)
    assert fit_rows | bag_rows == _rows(val.X)
    assert abs(len(fit_rows) - len(bag_rows)) <= N_CLASSES


def test_bags_and_target_go_through_the_same_estimator():
    method, val, _ = _fitted_method()
    # rebuild the split of V and the candidate bags with the same seeds used by fit
    _, V2_idx = train_test_split(np.arange(len(val)), test_size=0.5, random_state=0, stratify=val.y)
    V2 = val.sampling_from_index(V2_idx)
    bags = list(UPP(V2, sample_size=SAMPLE_SIZE, repeats=40, random_state=0, return_type="index")())
    assert len(bags) == len(method.sigma_pred_prevs)
    for idx, bag_prev in zip(bags[:10], method.sigma_pred_prevs[:10]):
        # a target equal to the candidate bag gets exactly the bag's estimate: same surrogate pipeline
        assert np.allclose(method._predict_test_prior(V2.X[idx]), bag_prev)
