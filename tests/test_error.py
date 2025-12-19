import itertools as IT

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from cap.error import k_bin, k_macro


def simulate_ct(ct: np.ndarray):
    n_samples = 500000
    n = ct.shape[0]
    couples = [(i, j) for i in range(n) for j in range(n)]
    counts = np.around(ct.flatten() * n_samples, decimals=0).astype("int")
    counts[-1] += n_samples - counts.sum()  # adjust for rounding errors
    y_couples = np.repeat(couples, counts, axis=0)
    y, y_hat = y_couples[:, 0], y_couples[:, 1]

    return y, y_hat


def macro_bacc_score(y, y_hat, n):
    baccs = 0.0
    for i in range(n):
        yi = np.where(y == i, 1, 0)
        y_hati = np.where(y_hat == i, 1, 0)
        baccs += balanced_accuracy_score(yi, y_hati)

    return baccs / n


def k_bin_params():
    repeats, n = 25, 2
    eps = 1e-3
    cts = []
    for _ in range(repeats):
        ct = np.random.random(n**2).reshape(n, n)
        ct += eps
        ct /= ct.sum()
        cts.append(ct)

    return cts


def k_macro_params():
    repeats, ns = 25, [3, 7, 11, 27]
    eps = 1e-3
    cts = []
    for n, _ in IT.product(ns, range(repeats)):
        ct = np.random.random(n**2).reshape(n, n)
        ct += eps
        ct /= ct.sum()
        cts.append(ct)

    return cts


@pytest.fixture(params=k_bin_params())
def k_bin_fix(request):
    ct = request.param
    y, y_hat = simulate_ct(ct)
    bacc = balanced_accuracy_score(y, y_hat) * 2 - 1
    yield ct, bacc


@pytest.fixture(params=k_macro_params())
def k_macro_fix(request):
    ct = request.param
    n = ct.shape[0]
    y, y_hat = simulate_ct(ct)
    bacc = macro_bacc_score(y, y_hat, n) * 2 - 1
    yield ct, bacc


class TestError:
    def test_k_bin(self, k_bin_fix):
        ct, result = k_bin_fix
        kres = k_bin(ct)
        assert np.isclose(k_bin(ct), result, atol=1e-4)
        assert kres >= -1.0 and kres <= 1.0
        assert result >= -1.0 and result <= 1.0

    def test_k_macro(self, k_macro_fix):
        ct, result = k_macro_fix
        kres = k_macro(ct)
        assert np.isclose(k_macro(ct), result, atol=1e-4)
        assert kres >= -1.0 and kres <= 1.0
        assert result >= -1.0 and result <= 1.0
