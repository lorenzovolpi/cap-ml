"""Tests for RQBS (SQS-H): the two variants of the prevalence and the reproducibility of the range from ``random_state``."""

import numpy as np
import pytest
import quapy as qp
from joblib import Parallel, delayed
from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC
from sklearn.linear_model import LogisticRegression

import cap.models.utils
from cap.models.confidence import RQBS
from cap.utils.commons import contingency_table

N_CLASSES = 3
SAMPLE_SIZE = 50
NUM_SAMPLES = 20


def _data(n, seed):
    rng = np.random.RandomState(seed)
    y = rng.randint(N_CLASSES, size=n)
    X = rng.normal(size=(n, 2)) + 2.0 * np.eye(N_CLASSES, 2)[y]
    return X, y


def _acc(ct):
    return np.trace(ct) / ct.sum()


def _fitted(random_state=0, bootstrap=True):
    """RQBS fitted on V, the target instances and the posteriors of h on the target."""
    X_tr, y_tr = _data(300, seed=0)
    h = LogisticRegression().fit(X_tr, y_tr)
    X_val, y_val = _data(400, seed=1)
    val = LabelledCollection(X_val, y_val, classes=np.arange(N_CLASSES))
    method = RQBS(
        _acc,
        PACC(LogisticRegression()),
        num_samples=NUM_SAMPLES,
        sample_size=SAMPLE_SIZE,
        random_state=random_state,
        bootstrap=bootstrap,
    )
    method.fit(val, h.predict_proba(X_val))
    X_test, _ = _data(SAMPLE_SIZE, seed=2)
    return method, X_test, h.predict_proba(X_test)


def _range(random_state=0, global_seed=None, bootstrap=True):
    """The table range of a fixed target; ``global_seed`` sets NumPy's global state first."""
    if global_seed is not None:
        np.random.seed(global_seed)
    method, X_test, P_test = _fitted(random_state, bootstrap)
    return method.predict_ct_range(X_test, P_test)


def _class_counts(cts):
    """Number of instances of each true class in each bag (tables have true classes on rows)."""
    return np.round(cts.sum(axis=2) * SAMPLE_SIZE).astype(int)


def test_bootstrap_is_the_default():
    assert RQBS(_acc, PACC(LogisticRegression()), sample_size=SAMPLE_SIZE, random_state=0).bootstrap is True


@pytest.mark.parametrize("bootstrap", [True, False])
def test_range_shape_and_tables(bootstrap):
    cts = _range(bootstrap=bootstrap)
    assert cts.shape == (NUM_SAMPLES, N_CLASSES, N_CLASSES)
    # normalised tables of bags of SAMPLE_SIZE instances
    assert np.allclose(cts.sum(axis=(1, 2)), 1.0)
    assert np.allclose(cts * SAMPLE_SIZE, np.round(cts * SAMPLE_SIZE))


@pytest.mark.parametrize("bootstrap", [True, False])
def test_range_is_reproducible_and_ignores_the_global_state(bootstrap):
    assert np.array_equal(_range(global_seed=1, bootstrap=bootstrap), _range(global_seed=2, bootstrap=bootstrap))


@pytest.mark.parametrize("bootstrap", [True, False])
def test_range_depends_on_random_state(bootstrap):
    assert not np.array_equal(_range(random_state=0, bootstrap=bootstrap), _range(random_state=1, bootstrap=bootstrap))


def test_range_is_the_same_in_joblib_workers():
    for bootstrap in (True, False):
        in_process = _range(bootstrap=bootstrap)
        in_workers = Parallel(n_jobs=2, backend="loky")(delayed(_range)(bootstrap=bootstrap) for _ in range(2))
        for cts in in_workers:
            assert np.array_equal(cts, in_process)


def test_global_state_and_seed_are_left_untouched():
    for bootstrap in (True, False):
        qp.environ["_R_SEED"] = 0
        np.random.seed(5)
        _ = np.random.rand()
        state = np.random.get_state()
        _range(random_state=7, bootstrap=bootstrap)
        after = np.random.get_state()
        assert qp.environ["_R_SEED"] == 0
        assert after[0] == state[0] and np.array_equal(after[1], state[1]) and after[2:] == state[2:]


def test_point_estimate_draws_every_bag_at_the_prevalence_of_the_target():
    method, X_test, P_test = _fitted(bootstrap=False)
    cts = method.predict_ct_range(X_test, P_test)
    q_hat = cap.models.utils.smooth(method.q.aggregate(method.q.classify(X_test)))
    counts = _class_counts(cts)
    # sampling_index takes round(size * q_hat_j) instances of class j and spreads the rounding remainder at random
    assert np.all(np.abs(counts - SAMPLE_SIZE * q_hat) <= N_CLASSES)
    assert np.all(counts.max(axis=0) - counts.min(axis=0) <= N_CLASSES)


def test_bootstrap_spreads_the_prevalences_of_the_bags():
    spread_bootstrap = np.ptp(_class_counts(_range(bootstrap=True)), axis=0).max()
    spread_point = np.ptp(_class_counts(_range(bootstrap=False)), axis=0).max()
    assert spread_bootstrap > spread_point


def test_bootstrap_variant_keeps_the_draws_of_v0_1_1_11():
    """The range with ``bootstrap=True`` equals the one of v0.1.1.11, rebuilt here step by step."""
    method, X_test, P_test = _fitted(bootstrap=True)
    predictions = method.q.classify(X_test)
    n = len(predictions)
    state = np.random.get_state()
    np.random.seed(method.random_state)
    qhat_range = [
        cap.models.utils.smooth(method.q.aggregate(predictions[np.random.randint(0, n, n)])) for _ in range(NUM_SAMPLES)
    ]
    idxs = [method.val_post.sampling_index(SAMPLE_SIZE, *q_hat) for q_hat in qhat_range]
    np.random.set_state(state)
    expected = [
        contingency_table(method.val_post.y[idx], method.val_post.X[idx, :].argmax(axis=1), N_CLASSES) for idx in idxs
    ]
    assert np.array_equal(method.predict_ct_range(X_test, P_test), np.asarray(expected))
