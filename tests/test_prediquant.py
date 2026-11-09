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


def test_table_range_is_the_closest_bags():
    method, _, h = _fitted_method()
    X_test, _ = _data(SAMPLE_SIZE, seed=2)
    P = h.predict_proba(X_test)
    cts = method.predict_ct_range(X_test, P)
    closest = method._predict_closest_idx(method._predict_test_prior(X_test))[: method.num_samples]
    assert cts.shape == (method.num_samples, N_CLASSES, N_CLASSES)
    assert np.allclose(cts, np.asarray(method.sigma_ct)[closest])
    # the accuracy range is the accuracy of each table of the range
    assert np.allclose(method.predict_range(X_test, P), [method.acc_fn(ct) for ct in cts])
    assert np.allclose(method.predict_ct(X_test, P), cts.mean(axis=0))


def test_mean_table_gives_the_mean_vanilla_accuracy():
    method, _, h = _fitted_method()
    X_test, _ = _data(SAMPLE_SIZE, seed=3)
    P = h.predict_proba(X_test)
    # equal-size bags: the trace of the mean table is the mean of the traces
    assert np.isclose(method.predict(X_test, P), method.predict_range(X_test, P).mean())
    assert np.isclose(method.predict_with_confidence(X_test, P).point_estimate, method.predict(X_test, P))


def test_acc_alias_and_switch():
    method, _, _ = _fitted_method()
    assert method.acc is method.acc_fn

    def other(ct):
        return ct[0, 0]

    method.switch(other)
    assert method.acc is other
