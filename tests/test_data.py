import pytest
from quacc.data import ExClassManager as ECM, ExtendedCollection
import numpy as np
import scipy.sparse as sp


class TestExClassManager:
    @pytest.mark.parametrize(
        "true_class,pred_class,result",
        [
            (0, 0, 0),
            (0, 1, 1),
            (1, 0, 2),
            (1, 1, 3),
        ],
    )
    def test_get_ex(self, true_class, pred_class, result):
        ncl = 2
        assert ECM.get_ex(ncl, true_class, pred_class) == result

    @pytest.mark.parametrize(
        "ex_class,result",
        [
            (0, 0),
            (1, 1),
            (2, 0),
            (3, 1),
        ],
    )
    def test_get_pred(self, ex_class, result):
        ncl = 2
        assert ECM.get_pred(ncl, ex_class) == result

    @pytest.mark.parametrize(
        "ex_class,result",
        [
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
        ],
    )
    def test_get_true(self, ex_class, result):
        ncl = 2
        assert ECM.get_true(ncl, ex_class) == result


class TestExtendedCollection:
    @pytest.mark.parametrize(
        "instances,result",
        [
            (
                np.asarray(
                    [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]]
                ),
                [np.asarray([1, 3]), np.asarray([0, 2])],
            ),
            (
                sp.csr_matrix(
                    [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]]
                ),
                [np.asarray([1, 3]), np.asarray([0, 2])],
            ),
            (
                np.asarray([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                [np.asarray([], dtype=int), np.asarray([0, 1])],
            ),
            (
                sp.csr_matrix([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                [np.asarray([], dtype=int), np.asarray([0, 1])],
            ),
            (
                np.asarray([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                [np.asarray([0, 1]), np.asarray([], dtype=int)],
            ),
            (
                sp.csr_matrix([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                [np.asarray([0, 1]), np.asarray([], dtype=int)],
            ),
        ],
    )
    def test__split_index_by_pred(self, instances, result):
        ncl = 2
        assert all(
            np.array_equal(a, b)
            for (a, b) in zip(
                ExtendedCollection._split_index_by_pred(ncl, instances),
                result,
            )
        )

    @pytest.mark.parametrize(
        "instances,s_inst,norms",
        [
            (
                np.asarray(
                    [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]]
                ),
                [
                    np.asarray([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                    np.asarray([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                ],
                [0.5, 0.5],
            ),
            (
                sp.csr_matrix(
                    [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]]
                ),
                [
                    sp.csr_matrix([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                    sp.csr_matrix([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                ],
                [0.5, 0.5],
            ),
            (
                np.asarray([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                [
                    np.asarray([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                    np.asarray([], dtype=int),
                ],
                [1.0, 0.0],
            ),
            (
                sp.csr_matrix([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                [
                    sp.csr_matrix([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                    sp.csr_matrix([], dtype=int),
                ],
                [1.0, 0.0],
            ),
            (
                np.asarray([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                [
                    np.asarray([], dtype=int),
                    np.asarray([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                ],
                [0.0, 1.0],
            ),
            (
                sp.csr_matrix([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                [
                    sp.csr_matrix([], dtype=int),
                    sp.csr_matrix([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                ],
                [0.0, 1.0],
            ),
        ],
    )
    def test_split_inst_by_pred(self, instances, s_inst, norms):
        ncl = 2
        _s_inst, _norms = ExtendedCollection.split_inst_by_pred(ncl, instances)
        if isinstance(s_inst, np.ndarray):
            assert all(np.array_equal(a, b) for (a, b) in zip(_s_inst, s_inst))
        if isinstance(s_inst, sp.csr_matrix):
            assert all((a != b).nnz == 0 for (a, b) in zip(_s_inst, s_inst))
        assert all(a == b for (a, b) in zip(_norms, norms))

    @pytest.mark.parametrize(
        "instances,labels,inst0,lbl0,inst1,lbl1",
        [
            (
                np.asarray(
                    [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]]
                ),
                np.asarray([3, 0, 1, 2]),
                np.asarray([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                np.asarray([0, 1]),
                np.asarray([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                np.asarray([1, 0]),
            ),
            (
                sp.csr_matrix(
                    [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]]
                ),
                np.asarray([3, 0, 1, 2]),
                sp.csr_matrix([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                np.asarray([0, 1]),
                sp.csr_matrix([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                np.asarray([1, 0]),
            ),
            (
                np.asarray([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                np.asarray([3, 1]),
                np.asarray([], dtype=int),
                np.asarray([], dtype=int),
                np.asarray([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                np.asarray([1, 0]),
            ),
            (
                sp.csr_matrix([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                np.asarray([3, 1]),
                sp.csr_matrix(np.empty((0, 0), dtype=int)),
                np.asarray([], dtype=int),
                sp.csr_matrix([[0, 0.3, 0.7], [2, 0.28, 0.72]]),
                np.asarray([1, 0]),
            ),
            (
                np.asarray([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                np.asarray([0, 2]),
                np.asarray([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                np.asarray([0, 1]),
                np.asarray([], dtype=int),
                np.asarray([], dtype=int),
            ),
            (
                sp.csr_matrix([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                np.asarray([0, 2]),
                sp.csr_matrix([[1, 0.54, 0.46], [3, 0.6, 0.4]]),
                np.asarray([0, 1]),
                sp.csr_matrix(np.empty((0, 0), dtype=int)),
                np.asarray([], dtype=int),
            ),
        ],
    )
    def test_split_by_pred(self, instances, labels, inst0, lbl0, inst1, lbl1):
        ec = ExtendedCollection(instances, labels, classes=range(0, 4))
        [ec0, ec1] = ec.split_by_pred()
        if isinstance(instances, np.ndarray):
            assert np.array_equal(ec0.X, inst0)
            assert np.array_equal(ec1.X, inst1)
        if isinstance(instances, sp.csr_matrix):
            assert (ec0.X != inst0).nnz == 0
            assert (ec1.X != inst1).nnz == 0
        assert np.array_equal(ec0.y, lbl0)
        assert np.array_equal(ec1.y, lbl1)
