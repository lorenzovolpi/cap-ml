import numpy as np
import pytest
import scipy.sparse as sp

from quacc.deprecated.method.base import MultiClassAccuracyEstimator
from quacc.legacy.data import ExtendedData, ExtensionPolicy


@pytest.mark.mcae
class TestMultiClassAccuracyEstimator:
    @pytest.mark.parametrize(
        "instances,pred_proba,extpol,result",
        [
            (
                np.arange(12).reshape((4, 3)),
                np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]),
                ExtensionPolicy(),
                np.array([0.21, 0.39, 0.1, 0.4]),
            ),
            (
                np.arange(12).reshape((4, 3)),
                np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]),
                ExtensionPolicy(collapse_false=True),
                np.array([0.21, 0.39, 0.5]),
            ),
            (
                sp.csr_matrix(np.arange(12).reshape((4, 3))),
                np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.9, 0.1]]),
                ExtensionPolicy(),
                np.array([0.21, 0.39, 0.1, 0.4]),
            ),
            (
                np.arange(12).reshape((4, 3)),
                np.array(
                    [
                        [0.3, 0.2, 0.5],
                        [0.13, 0.67, 0.2],
                        [0.21, 0.09, 0.8],
                        [0.19, 0.1, 0.71],
                    ]
                ),
                ExtensionPolicy(),
                np.array([0.21, 0.09, 0.1, 0.04, 0.06, 0.11, 0.11, 0.18, 0.1]),
            ),
            (
                np.arange(12).reshape((4, 3)),
                np.array(
                    [
                        [0.3, 0.2, 0.5],
                        [0.13, 0.67, 0.2],
                        [0.21, 0.09, 0.8],
                        [0.19, 0.1, 0.71],
                    ]
                ),
                ExtensionPolicy(collapse_false=True),
                np.array([0.21, 0.09, 0.1, 0.7]),
            ),
            (
                sp.csr_matrix(np.arange(12).reshape((4, 3))),
                np.array(
                    [
                        [0.3, 0.2, 0.5],
                        [0.13, 0.67, 0.2],
                        [0.21, 0.09, 0.8],
                        [0.19, 0.1, 0.71],
                    ]
                ),
                ExtensionPolicy(),
                np.array([0.21, 0.09, 0.1, 0.04, 0.06, 0.11, 0.11, 0.18, 0.1]),
            ),
        ],
    )
    def test_estimate(self, monkeypatch, instances, pred_proba, extpol, result):
        ed = ExtendedData(instances, pred_proba, pred_proba, extpol)

        class MockQuantifier:
            def __init__(self):
                self.classes_ = np.arange(result.shape[0])

            def quantify(self, X):
                return result

        def mockinit(self):
            self.extpol = extpol
            self.quantifier = MockQuantifier()

        def mock_extend_instances(self, instances):
            return ed

        monkeypatch.setattr(MultiClassAccuracyEstimator, "__init__", mockinit)
        monkeypatch.setattr(
            MultiClassAccuracyEstimator, "_extend_instances", mock_extend_instances
        )
        mcae = MultiClassAccuracyEstimator()

        ep1 = mcae.estimate(instances)
        ep2 = mcae.estimate(ed)

        assert (ep1.flat == ep2.flat).all()
        assert (ep1.flat == result).all()
