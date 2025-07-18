import numpy as np
import pytest

from quacc import error
from quacc.legacy.data import ExtendedPrev, ExtensionPolicy


@pytest.mark.err
class TestError:
    @pytest.mark.parametrize(
        "prev,result",
        [
            (np.array([[1, 4], [4, 4]]), 0.5),
            (np.array([[6, 2, 4], [2, 4, 2], [4, 2, 6]]), 0.5),
        ],
    )
    def test_f1(self, prev, result):
        ep = ExtendedPrev(prev.flatten(), prev.shape[0], extpol=ExtensionPolicy())
        assert error.f1(prev) == result
        assert error.f1(ep) == result

    @pytest.mark.parametrize(
        "prev,result",
        [
            (np.array([[4, 4], [4, 4]]), 0.5),
            (np.array([[2, 4, 2], [2, 2, 4], [4, 2, 2]]), 0.25),
        ],
    )
    def test_acc(self, prev, result):
        ep = ExtendedPrev(prev.flatten(), prev.shape[0], extpol=ExtensionPolicy())
        assert error.acc(prev) == result
        assert error.acc(ep) == result

    @pytest.mark.parametrize(
        "true_prev,estim_prev,nbcl,extpol,result",
        [
            (
                [
                    np.array([0.2, 0.4, 0.1, 0.3]),
                    np.array([0.1, 0.5, 0.1, 0.3]),
                ],
                [
                    np.array([0.3, 0.4, 0.2, 0.1]),
                    np.array([0.5, 0.3, 0.1, 0.1]),
                ],
                2,
                ExtensionPolicy(),
                np.array([0.1, 0.2]),
            ),
            (
                [
                    np.array([0.2, 0.4, 0.4]),
                    np.array([0.1, 0.5, 0.4]),
                ],
                [
                    np.array([0.3, 0.4, 0.3]),
                    np.array([0.5, 0.3, 0.2]),
                ],
                2,
                ExtensionPolicy(collapse_false=True),
                np.array([0.1, 0.2]),
            ),
            (
                [
                    np.array([0.02, 0.04, 0.16, 0.38, 0.1, 0.05, 0.15, 0.08, 0.02]),
                    np.array([0.04, 0.02, 0.14, 0.40, 0.1, 0.03, 0.17, 0.07, 0.03]),
                ],
                [
                    np.array([0.02, 0.04, 0.16, 0.48, 0.0, 0.05, 0.15, 0.08, 0.02]),
                    np.array([0.14, 0.02, 0.04, 0.30, 0.2, 0.03, 0.17, 0.07, 0.03]),
                ],
                3,
                ExtensionPolicy(),
                np.array([0.1, 0.2]),
            ),
            (
                [
                    np.array([0.2, 0.4, 0.2, 0.2]),
                    np.array([0.1, 0.3, 0.2, 0.4]),
                ],
                [
                    np.array([0.3, 0.3, 0.1, 0.3]),
                    np.array([0.5, 0.2, 0.1, 0.2]),
                ],
                3,
                ExtensionPolicy(collapse_false=True),
                np.array([0.1, 0.2]),
            ),
        ],
    )
    def test_accd(self, true_prev, estim_prev, nbcl, extpol, result):
        true_prev = [ExtendedPrev(tp, nbcl, extpol=extpol) for tp in true_prev]
        estim_prev = [ExtendedPrev(ep, nbcl, extpol=extpol) for ep in estim_prev]
        _err = error.accd(true_prev, estim_prev)
        assert (np.abs(_err - result) < 1e-15).all()
