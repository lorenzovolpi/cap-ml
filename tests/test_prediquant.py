"""Protocol tests for PrediQuant (SQS-S): split of V (also with rho), consistency of the prevalence estimates, stored
quantities after a prediction."""

import numpy as np
import pytest
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


def _fitted_method(rho=None, seed=0):
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
        random_state=seed,
        rho=rho,
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


RHOS = [0.1, 0.3, 0.5, 0.7, 0.9]


def test_default_split_is_train_test_split():
    method, val, _ = _fitted_method()
    Vq_idx, Vb_idx = train_test_split(np.arange(len(val)), test_size=0.5, random_state=0, stratify=val.y)
    assert np.array_equal(method.Vq_idx, Vq_idx)
    assert np.array_equal(method.Vb_idx, Vb_idx)


@pytest.mark.parametrize("rho", RHOS)
def test_rho_split_partitions_v_with_the_right_class_counts(rho):
    method, val, _ = _fitted_method(rho=rho)
    Vq, Vb = set(method.Vq_idx.tolist()), set(method.Vb_idx.tolist())
    assert Vq.isdisjoint(Vb)
    assert Vq | Vb == set(range(len(val)))
    for c in val.classes_:
        n_c = int((val.y == c).sum())
        assert int((val.y[method.Vq_idx] == c).sum()) == int(np.floor(rho * n_c + 0.5))


def test_rho_splits_are_nested():
    vqs = [set(_fitted_method(rho=rho)[0].Vq_idx.tolist()) for rho in RHOS]
    for small, large in zip(vqs[:-1], vqs[1:]):
        assert small < large


def test_rho_split_depends_on_the_seed():
    a = _fitted_method(rho=0.3, seed=0)[0].Vq_idx
    b = _fitted_method(rho=0.3, seed=1)[0].Vq_idx
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("rho", [0.0, 1.0, -0.2, 1.5])
def test_rho_out_of_range_is_rejected(rho):
    with pytest.raises(ValueError):
        PrediQuant(lambda ct: 0.0, PACC(LogisticRegression()), sample_size=SAMPLE_SIZE, rho=rho)


def test_rho_that_empties_a_class_is_rejected():
    X_val, y_val = _data(400, seed=1)
    y_val[:] = 0
    y_val[:2] = 1
    y_val[2:4] = 2
    val = LabelledCollection(X_val, y_val, classes=np.arange(N_CLASSES))
    method = PrediQuant(lambda ct: 0.0, PACC(LogisticRegression()), sample_size=SAMPLE_SIZE, random_state=0, rho=0.1)
    with pytest.raises(ValueError):
        method.fit(val, np.full((len(val), N_CLASSES), 1.0 / N_CLASSES))


@pytest.mark.parametrize("rho", RHOS)
def test_rho_quantifier_never_sees_vb_and_bags_share_the_estimator(rho):
    method, val, _ = _fitted_method(rho=rho)
    assert _rows(method.q.fit_X) == _rows(val.X[method.Vq_idx])
    assert _rows(method.q.classified_X[0]) == _rows(val.X[method.Vb_idx])
    Vb_X = val.X[method.Vb_idx]
    for idx, bag_prev in zip(method.sigma_indices[:5], method.sigma_pred_prevs[:5]):
        assert np.allclose(method._predict_test_prior(Vb_X[idx]), bag_prev)


def test_stored_prevalence_and_selected_bags():
    method, _, h = _fitted_method(rho=0.3)
    X_test, _ = _data(SAMPLE_SIZE, seed=4)
    cts = method.predict_ct_range(X_test, h.predict_proba(X_test))
    assert np.allclose(method.last_test_prev, method.q.predict(X_test))
    assert len(method.last_selected_bags) == method.num_samples
    assert np.allclose(cts, np.asarray(method.sigma_ct)[method.last_selected_bags])


def test_class_support_on_known_bags():
    method, _, _ = _fitted_method()
    Vb_y = np.array([0, 0, 1, 1, 2, 2, 2, 2])
    method.Vb_y = Vb_y
    method.classes_ = np.arange(N_CLASSES)
    method.sigma_indices = [np.array([0, 2, 4]), np.array([0, 0, 5]), np.array([1, 3, 6])]
    # bags 0 and 1: class 0 -> {0}, class 1 -> {2}, class 2 -> {4, 5}
    assert np.allclose(method.class_support([0, 1]), [1 / 2, 1 / 2, 2 / 4])
    # all bags: classes 0 and 1 fully covered, class 2 -> {4, 5, 6}
    assert np.allclose(method.class_support([0, 1, 2]), [1.0, 1.0, 3 / 4])
    method.last_selected_bags = np.array([2])
    assert np.allclose(method.class_support(), [1 / 2, 1 / 2, 1 / 4])
