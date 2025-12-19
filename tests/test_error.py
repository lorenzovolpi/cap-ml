import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score

from cap.error import K_bin, K_macro


def simulate_ct(ct: np.ndarray):
    n_samples = 100000
    n = ct.shape[0]
    couples = [(i, j) for i in range(n) for j in range(n)]
    counts = np.around(ct.flatten() * n_samples, decimals=0).astype("int")
    counts[-1] += n_samples - counts.sum()  # adjust for rounding errors
    y_couples = np.repeat(couples, counts, axis=0)
    y, y_hat = y_couples[:, 0], y_couples[:, 1]

    # y_count = np.around(ct.sum(axis=1) * n_samples, decimals=0).astype("int")
    # y_count[-1] += n_samples - y_count.sum()  # adjust for rounding errors
    # y_hat_count = np.around(ct.sum(axis=0) * n_samples, decimals=0).astype("int")
    # y_hat_count[-1] += n_samples - y_hat_count.sum()  # adjust for rounding errors
    #
    # y = np.repeat(np.arange(n), y_count)
    # y_hat = np.repeat(np.arange(n), y_hat_count)

    return y, y_hat


def k_bin_params():
    repeats, n = 20, 2
    eps = 1e-3
    cts = []
    for _ in range(repeats):
        ct = np.random.random(n**2).reshape(n, n)
        ct += eps
        ct /= ct.sum()
        cts.append(ct)
    return cts


def k_macro_params():
    repeats, n = 20, 3
    eps = 1e-3
    cts = []
    for _ in range(repeats):
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
    y, y_hat = simulate_ct(ct)
    bacc = balanced_accuracy_score(y, y_hat) * 2 - 1
    # bacc = balanced_accuracy_score(y, y_hat)
    yield ct, bacc


class TestError:
    def test_k_bin(self, k_bin_fix):
        ct, result = k_bin_fix
        assert np.isclose(K_bin(ct), result)

    # @pytest.mark.parametrize(
    #     "ct,result",
    #     [
    #         (np.array([[0.2, 0.07, 0.25], [0.08, 0.1, 0.03], [0.07, 0.15, 0.05]]), 0.02381612),
    #         # (np.array([[0.0, 0.0], [0.4, 0.6]]), 0.2),
    #         # (np.array([[0.4, 0.6], [0.0, 0.0]]), -0.2),
    #     ],
    # )
    def test_k_macro(self, k_macro_fix):
        ct, result = k_macro_fix
        n = ct.shape[0]
        print(f"{n=}")
        # for i in range(n):
        #     tp = ct[i, i]
        #     fp = np.sum(ct[:, i]) - tp
        #     fn = np.sum(ct[i, :]) - tp
        #     tn = 1.0 - tp - fp - fn
        #     tpr = tp / (tp + fn)
        #     tnr = tn / (tn + fp)
        #     print(f"case {i}")
        #     print(f"{tp=}, {fp=}, {fn=}, {tn=}")
        #     print(f"{tpr=}, {tnr=}")
        #     print(f"k{i}={tpr + tnr - 1}")
        #     print()
        res = K_macro(ct)
        print(res)
        assert np.isclose(res, result)
