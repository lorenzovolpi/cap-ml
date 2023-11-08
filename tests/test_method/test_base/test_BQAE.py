import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression

from quacc.method.base import BinaryQuantifierAccuracyEstimator


class TestBQAE:
    @pytest.mark.parametrize(
        "instances,preds0,preds1,result",
        [
            (
                np.asarray(
                    [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]]
                ),
                np.asarray([0.3, 0.7]),
                np.asarray([0.4, 0.6]),
                np.asarray([0.15, 0.2, 0.35, 0.3]),
            ),
            (
                sp.csr_matrix(
                    [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]]
                ),
                np.asarray([0.3, 0.7]),
                np.asarray([0.4, 0.6]),
                np.asarray([0.15, 0.2, 0.35, 0.3]),
            ),
            (
                np.asarray([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                np.asarray([0.3, 0.7]),
                np.asarray([0.4, 0.6]),
                np.asarray([0.0, 0.4, 0.0, 0.6]),
            ),
            (
                sp.csr_matrix([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                np.asarray([0.3, 0.7]),
                np.asarray([0.4, 0.6]),
                np.asarray([0.0, 0.4, 0.0, 0.6]),
            ),
            (
                np.asarray([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                np.asarray([0.3, 0.7]),
                np.asarray([0.4, 0.6]),
                np.asarray([0.3, 0.0, 0.7, 0.0]),
            ),
            (
                sp.csr_matrix([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                np.asarray([0.3, 0.7]),
                np.asarray([0.4, 0.6]),
                np.asarray([0.3, 0.0, 0.7, 0.0]),
            ),
        ],
    )
    def test_estimate_ndarray(self, mocker, instances, preds0, preds1, result):
        estimator = BinaryQuantifierAccuracyEstimator(LogisticRegression())
        estimator.n_classes = 4
        with mocker.patch.object(estimator.q_model_0, "quantify"), mocker.patch.object(
            estimator.q_model_1, "quantify"
        ):
            estimator.q_model_0.quantify.return_value = preds0
            estimator.q_model_1.quantify.return_value = preds1
            assert np.array_equal(
                estimator.estimate(instances, ext=True),
                result,
            )
