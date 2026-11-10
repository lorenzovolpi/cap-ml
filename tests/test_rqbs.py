"""Tests for RQBS (SQS-H, bootstrap variant): reproducibility of the table range from ``random_state``."""

import numpy as np
import quapy as qp
from joblib import Parallel, delayed
from quapy.data import LabelledCollection
from quapy.method.aggregative import PACC
from sklearn.linear_model import LogisticRegression

from cap.models.confidence import RQBS

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


def _range(random_state=0, global_seed=None):
    """Fit RQBS on V and return the table range of a fixed target; ``global_seed`` sets NumPy's global state first."""
    if global_seed is not None:
        np.random.seed(global_seed)
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
    )
    method.fit(val, h.predict_proba(X_val))
    X_test, _ = _data(SAMPLE_SIZE, seed=2)
    return method.predict_ct_range(X_test, h.predict_proba(X_test))


def test_range_shape_and_tables():
    cts = _range()
    assert cts.shape == (NUM_SAMPLES, N_CLASSES, N_CLASSES)
    # normalised tables of bags of SAMPLE_SIZE instances
    assert np.allclose(cts.sum(axis=(1, 2)), 1.0)
    assert np.allclose(cts * SAMPLE_SIZE, np.round(cts * SAMPLE_SIZE))


def test_range_is_reproducible_and_ignores_the_global_state():
    assert np.array_equal(_range(global_seed=1), _range(global_seed=2))


def test_range_depends_on_random_state():
    assert not np.array_equal(_range(random_state=0), _range(random_state=1))


def test_range_is_the_same_in_joblib_workers():
    in_process = _range()
    in_workers = Parallel(n_jobs=2, backend="loky")(delayed(_range)() for _ in range(2))
    for cts in in_workers:
        assert np.array_equal(cts, in_process)


def test_global_state_and_seed_are_left_untouched():
    qp.environ["_R_SEED"] = 0
    np.random.seed(5)
    _ = np.random.rand()
    state = np.random.get_state()
    _range(random_state=7)
    after = np.random.get_state()
    assert qp.environ["_R_SEED"] == 0
    assert after[0] == state[0] and np.array_equal(after[1], state[1]) and after[2:] == state[2:]
