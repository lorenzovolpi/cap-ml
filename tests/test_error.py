import numpy as np
import pytest

from cap.error import K_bin, K_macro


def simulate_ct(ct):
    y_count = np.around(ct.sum(axis=1) * 1000, decimals=0)
    y_hat_count = np.around(ct.sum(axis=0) * 1000, decimals=0)


@pytest.fixture
def k_macro_fix():
    eps = 1e-3
    for i in range(10):
        ct = np.random.random(3**2)
        ct += eps
        ct /= ct.sum()
        ct.reshape(3, 3)


@pytest.mark.err
class TestError:
    @pytest.mark.parametrize(
        "ct,result",
        [
            (np.array([[0.1, 0.3], [0.2, 0.4]]), -0.08333333333333337),
            (np.array([[0.0, 0.0], [0.4, 0.6]]), 0.2),
            (np.array([[0.4, 0.6], [0.0, 0.0]]), -0.2),
        ],
    )
    def test_k_bin(self, ct, result):
        assert np.isclose(K_bin(ct), result)

    @pytest.mark.parametrize(
        "ct,result",
        [
            (np.array([[0.2, 0.07, 0.25], [0.08, 0.1, 0.03], [0.07, 0.15, 0.05]]), 0.02381612),
            # (np.array([[0.0, 0.0], [0.4, 0.6]]), 0.2),
            # (np.array([[0.4, 0.6], [0.0, 0.0]]), -0.2),
        ],
    )
    def test_k_macro(self, ct, result):
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
