from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

from quacc.legacy.data import (
    ExtBinPrev,
    ExtendedCollection,
    ExtendedData,
    ExtendedLabels,
    ExtendedPrev,
    ExtensionPolicy,
    ExtMulPrev,
    _split_index_by_pred,
)


@pytest.fixture
def nd_1():
    return np.arange(12).reshape((4, 3))


@pytest.fixture
def csr_1(nd_1):
    return sp.csr_matrix(nd_1)


@pytest.mark.ext
class TestData:
    @pytest.mark.parametrize(
        "pred_proba,result",
        [
            (
                np.array([[0.3, 0.7], [0.54, 0.46], [0.28, 0.72], [0.6, 0.4]]),
                [np.array([1, 3]), np.array([0, 2])],
            ),
            (
                np.array([[0.3, 0.7], [0.28, 0.72]]),
                [np.array([]), np.array([0, 1])],
            ),
            (
                np.array([[0.54, 0.46], [0.6, 0.4]]),
                [np.array([0, 1]), np.array([])],
            ),
            (
                np.array(
                    [
                        [0.25, 0.4, 0.35],
                        [0.24, 0.3, 0.46],
                        [0.61, 0.28, 0.11],
                        [0.4, 0.1, 0.5],
                    ]
                ),
                [np.array([2]), np.array([0]), np.array([1, 3])],
            ),
        ],
    )
    def test_split_index_by_pred(self, pred_proba, result):
        _split_index = _split_index_by_pred(pred_proba)
        assert len(_split_index) == len(result)
        assert all((a == b).all() for (a, b) in zip(_split_index, result))


@pytest.mark.ext
@pytest.mark.extpol
class TestExtendedPolicy:
    # fmt: off
    @pytest.mark.parametrize(
        "extpol,nbcl,result",
        [
            (ExtensionPolicy(), 2, np.array([0, 1, 2, 3])),
            (ExtensionPolicy(group_false=True), 2, np.array([0, 1, 2, 3])),
            (ExtensionPolicy(collapse_false=True), 2, np.array([0, 1, 2])),
            (ExtensionPolicy(), 3, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])),
            (ExtensionPolicy(group_false=True), 3, np.array([0, 1, 2, 3, 4, 5])),
            (ExtensionPolicy(collapse_false=True), 3, np.array([0, 1, 2, 3])),
        ],
    )
    def test_qclasses(self, extpol, nbcl, result):
        assert (result == extpol.qclasses(nbcl)).all()

    @pytest.mark.parametrize(
        "extpol,nbcl,result",
        [
            (ExtensionPolicy(), 2, np.array([0, 1, 2, 3])),
            (ExtensionPolicy(group_false=True), 2, np.array([0, 1, 2, 3])),
            (ExtensionPolicy(collapse_false=True), 2, np.array([0, 1, 2, 3])),
            (ExtensionPolicy(), 3, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])),
            (ExtensionPolicy(group_false=True), 3, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])),
            (ExtensionPolicy(collapse_false=True), 3, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])),
        ],
    )
    def test_eclasses(self, extpol, nbcl, result):
        assert (result == extpol.eclasses(nbcl)).all()

    @pytest.mark.parametrize(
        "extpol,nbcl,result",
        [
            (ExtensionPolicy(), 2, np.array([0, 1])),
            (ExtensionPolicy(group_false=True), 2, np.array([0, 1])),
            (ExtensionPolicy(collapse_false=True), 2, np.array([0, 1])),
            (ExtensionPolicy(), 3, np.array([0, 1, 2])),
            (ExtensionPolicy(group_false=True), 3, np.array([0, 1])),
            (ExtensionPolicy(collapse_false=True), 3, np.array([0, 1, 2])),
        ],
    )
    def test_tfp_classes(self, extpol, nbcl, result):
        assert (result == extpol.tfp_classes(nbcl)).all()

    @pytest.mark.parametrize(
        "extpol,nbcl,result",
        [
            (
                ExtensionPolicy(), 2,
                (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])),
            ),
            (
                ExtensionPolicy(group_false=True), 2,
                (np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1])),
            ),
            (
                ExtensionPolicy(collapse_false=True), 2,
                (np.array([0, 1, 0]), np.array([0, 1, 1])),
            ),
            (
                ExtensionPolicy(), 3,
                (np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]), np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])),
            ),
            (
                ExtensionPolicy(group_false=True), 3,
                (np.array([0, 1, 2, 1, 2, 0]), np.array([0, 1, 2, 0, 1, 2])),
            ),
            (
                ExtensionPolicy(collapse_false=True), 3,
                (np.array([0, 1, 2, 0]), np.array([0, 1, 2, 1])),
            ),
        ],
    )
    def test_matrix_idx(self, extpol, nbcl, result):
        _midx = extpol.matrix_idx(nbcl)
        assert len(_midx) == len(result)
        assert all((idx == r).all() for idx, r in zip(_midx, result))

    @pytest.mark.parametrize(
        "extpol,nbcl,true,pred,result",
        [
            (
                ExtensionPolicy(), 2,
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                np.array([3, 0, 2, 3, 1, 0]),
            ),
            (
                ExtensionPolicy(group_false=True), 2,
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                np.array([1, 0, 2, 1, 3, 0]),
            ),
            (
                ExtensionPolicy(collapse_false=True), 2,
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                np.array([1, 0, 2, 1, 2, 0]),
            ),
            (
                ExtensionPolicy(), 3,
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                np.array([4, 6, 0, 3, 1, 7, 2, 5, 8]),
            ),
            (
                ExtensionPolicy(group_false=True), 3,
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                np.array([1, 3, 0, 3, 4, 4, 5, 5, 2]),
            ),
            (
                ExtensionPolicy(collapse_false=True), 3,
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                np.array([1, 3, 0, 3, 3, 3, 3, 3, 2]),
            ),
        ],
    )
    def test_ext_lbl(self, extpol, nbcl, true, pred, result):
        vfun = extpol.ext_lbl(nbcl)
        assert (vfun(true, pred) == result).all()

    @pytest.mark.parametrize(
        "extpol,nbcl,true,pred,result",
        [
            (
                ExtensionPolicy(), 2,
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                np.array([1, 0, 1, 1, 0, 0]),
            ),
            (
                ExtensionPolicy(group_false=True), 2,
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                np.array([0, 0, 1, 0, 1, 0]),
            ),
            (
                ExtensionPolicy(collapse_false=True), 2,
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                np.array([1, 0, 1, 1, 0, 0]),
            ),
            (
                ExtensionPolicy(), 3,
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
            ),
            (
                ExtensionPolicy(group_false=True), 3,
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                np.array([0, 1, 0, 1, 1, 1, 1, 1, 0]),
            ),
            (
                ExtensionPolicy(collapse_false=True), 3,
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
            ),
        ],
    )
    def test_true_lbl_from_pred(self, extpol, nbcl, true, pred, result):
        vfun = extpol.true_lbl_from_pred(nbcl)
        assert (vfun(true, pred) == result).all()
    # fmt: on


@pytest.mark.ext
@pytest.mark.extd
class TestExtendedData:
    @pytest.mark.parametrize(
        "instances_name,indexes,result",
        [
            (
                "nd_1",
                [np.array([0, 2]), np.array([1, 3])],
                [
                    np.array([[0, 1, 2], [6, 7, 8]]),
                    np.array([[3, 4, 5], [9, 10, 11]]),
                ],
            ),
            (
                "nd_1",
                [np.array([0]), np.array([1, 3]), np.array([2])],
                [
                    np.array([[0, 1, 2]]),
                    np.array([[3, 4, 5], [9, 10, 11]]),
                    np.array([[6, 7, 8]]),
                ],
            ),
        ],
    )
    def test_split_by_pred(self, instances_name, indexes, result, monkeypatch, request):
        def mockinit(self):
            self.instances = request.getfixturevalue(instances_name)

        monkeypatch.setattr(ExtendedData, "__init__", mockinit)
        d = ExtendedData()
        split = d.split_by_pred(indexes)
        assert all([(s == r).all() for s, r in zip(split, result)])


@pytest.mark.ext
@pytest.mark.extl
class TestExtendedLabels:
    @pytest.mark.parametrize(
        "true,pred,nbcl,extpol,result",
        [
            (
                np.array([1, 0, 0, 1, 1]),
                np.array([1, 1, 0, 0, 1]),
                2,
                ExtensionPolicy(),
                np.array([3, 1, 0, 2, 3]),
            ),
            (
                np.array([1, 0, 0, 1, 1]),
                np.array([1, 1, 0, 0, 1]),
                2,
                ExtensionPolicy(group_false=True),
                np.array([1, 3, 0, 2, 1]),
            ),
            (
                np.array([1, 0, 0, 1, 1]),
                np.array([1, 1, 0, 0, 1]),
                2,
                ExtensionPolicy(collapse_false=True),
                np.array([1, 2, 0, 2, 1]),
            ),
            (
                np.array([1, 0, 0, 1, 0, 1, 2, 2, 2]),
                np.array([1, 1, 0, 0, 2, 2, 2, 0, 1]),
                3,
                ExtensionPolicy(),
                np.array([4, 1, 0, 3, 2, 5, 8, 6, 7]),
            ),
            (
                np.array([1, 0, 0, 1, 0, 1, 2, 2, 2]),
                np.array([1, 1, 0, 0, 2, 2, 2, 0, 1]),
                3,
                ExtensionPolicy(group_false=True),
                np.array([1, 4, 0, 3, 5, 5, 2, 3, 4]),
            ),
            (
                np.array([1, 0, 0, 1, 0, 1, 2, 2, 2]),
                np.array([1, 1, 0, 0, 2, 2, 2, 0, 1]),
                3,
                ExtensionPolicy(collapse_false=True),
                np.array([1, 3, 0, 3, 3, 3, 2, 3, 3]),
            ),
        ],
    )
    def test_y(self, true, pred, nbcl, extpol, result):
        el = ExtendedLabels(true, pred, nbcl, extpol)
        assert (el.y == result).all()

    @pytest.mark.parametrize(
        "extpol,nbcl,indexes,true,pred,result,rcls",
        [
            (
                ExtensionPolicy(),
                2,
                [np.array([1, 2, 5]), np.array([0, 3, 4])],
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                [np.array([0, 1, 0]), np.array([1, 1, 0])],
                np.array([0, 1]),
            ),
            (
                ExtensionPolicy(group_false=True),
                2,
                [np.array([1, 2, 5]), np.array([0, 3, 4])],
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                [np.array([0, 1, 0]), np.array([0, 0, 1])],
                np.array([0, 1]),
            ),
            (
                ExtensionPolicy(collapse_false=True),
                2,
                [np.array([1, 2, 5]), np.array([0, 3, 4])],
                np.array([1, 0, 1, 1, 0, 0]),
                np.array([1, 0, 0, 1, 1, 0]),
                [np.array([0, 1, 0]), np.array([1, 1, 0])],
                np.array([0, 1]),
            ),
            (
                ExtensionPolicy(),
                3,
                [np.array([1, 2, 3]), np.array([0, 4, 5]), np.array([6, 7, 8])],
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                [np.array([2, 0, 1]), np.array([1, 0, 2]), np.array([0, 1, 2])],
                np.array([0, 1, 2]),
            ),
            (
                ExtensionPolicy(group_false=True),
                3,
                [np.array([1, 2, 3]), np.array([0, 4, 5]), np.array([6, 7, 8])],
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                [np.array([1, 0, 1]), np.array([0, 1, 1]), np.array([1, 1, 0])],
                np.array([0, 1]),
            ),
            (
                ExtensionPolicy(collapse_false=True),
                3,
                [np.array([1, 2, 3]), np.array([0, 4, 5]), np.array([6, 7, 8])],
                np.array([1, 2, 0, 1, 0, 2, 0, 1, 2]),
                np.array([1, 0, 0, 0, 1, 1, 2, 2, 2]),
                [np.array([2, 0, 1]), np.array([1, 0, 2]), np.array([0, 1, 2])],
                np.array([0, 1, 2]),
            ),
        ],
    )
    def test_split_by_pred(self, extpol, nbcl, indexes, true, pred, result, rcls):
        el = ExtendedLabels(true, pred, nbcl, extpol)
        labels, cls = el.split_by_pred(indexes)
        assert (cls == rcls).all()
        assert all([(lbl == r).all() for lbl, r in zip(labels, result)])


@pytest.mark.ext
@pytest.mark.extp
class TestExtendedPrev:
    # fmt: off
    @pytest.mark.parametrize(
        "flat,nbcl,extpol,result",
        [
            (
                np.array([0.05, 0.1, 0.6, 0.25]), 2, ExtensionPolicy(),
                np.array([[0.05, 0.1], [0.6, 0.25]]),
            ),
            (
                np.array([0.05, 0.1, 0.6, 0.25]), 2, ExtensionPolicy(group_false=True),
                np.array([[0.05, 0.25], [0.6, 0.1]]),
            ),
            (
                np.array([0.05, 0.1, 0.85]), 2, ExtensionPolicy(collapse_false=True),
                np.array([[0.05, 0.85], [0, 0.1]]),
            ),
            (
                np.array([0.05, 0.1, 0.2, 0.15, 0.04, 0.06, 0.15, 0.14, 0.1]), 3, ExtensionPolicy(),
                np.array([[0.05, 0.1, 0.2], [0.15, 0.04, 0.06], [0.15, 0.14, 0.1]]),
            ),
            (
                np.array([0.15, 0.2, 0.15, 0.1, 0.15, 0.25]), 3, ExtensionPolicy(group_false=True),
                np.array([[0.15, 0.0, 0.25], [0.1, 0.2, 0.0], [0.0, 0.15, 0.15]]),
            ),
            (
                np.array([0.05, 0.2, 0.65, 0.1]), 3, ExtensionPolicy(collapse_false=True),
                np.array([[0.05, 0.1, 0], [0, 0.2, 0], [0, 0, 0.65]]),
            ),
        ],
    )
    def test__build_matrix(self, monkeypatch, flat, nbcl, extpol, result):
        def mockinit(self, flat, nbcl, extpol):
            self.flat = flat
            self.nbcl = nbcl
            self.extpol = extpol

        monkeypatch.setattr(ExtendedPrev, "__init__", mockinit)
        ep = ExtendedPrev(flat, nbcl, extpol)
        _matrix = ep._ExtendedPrev__build_matrix()
        assert _matrix.shape == result.shape
        assert (_matrix == result).all()

    # fmt: on


@pytest.mark.ext
@pytest.mark.extp
class TestExtMulPrev:
    # fmt: off
    @pytest.mark.parametrize(
        "flat,nbcl,extpol,q_classes,result",
        [
            (np.array([0.2, 0, 0.8, 0]), 2, ExtensionPolicy(), [0, 1, 2, 3], np.array([0.2, 0, 0.8, 0])),
            (np.array([0.2, 0.8]), 2, ExtensionPolicy(), [0, 3], np.array([0.2, 0, 0, 0.8])),
            (np.array([0.2, 0.8]), 2, ExtensionPolicy(group_false=True), [0, 3], np.array([0.2, 0, 0, 0.8])),
            (np.array([0.2, 0.8]), 2, ExtensionPolicy(collapse_false=True), [0, 2], np.array([0.2, 0, 0.8])),
            (np.array([0.1, 0.1, 0.6, 0.2]), 3, ExtensionPolicy(), [0, 1, 3, 5], np.array([0.1, 0.1, 0, 0.6, 0, 0.2, 0, 0, 0])),
            (np.array([0.1, 0.1, 0.6, 0.2]), 3, ExtensionPolicy(group_false=True), [0, 1, 3, 5], np.array([0.1, 0.1, 0, 0.6, 0, 0.2])),
            (np.array([0.1, 0.1, 0.6]), 3, ExtensionPolicy(collapse_false=True), [0, 1, 2], np.array([0.1, 0.1, 0.6, 0])),
        ],
    )
    def test__check_q_classes(self, monkeypatch, flat, nbcl, extpol, q_classes, result):
        def mockinit(self, nbcl, extpol):
            self.nbcl = nbcl
            self.extpol = extpol

        monkeypatch.setattr(ExtMulPrev, "__init__", mockinit)
        ep = ExtMulPrev(nbcl, extpol)
        _flat = ep._ExtMulPrev__check_q_classes(q_classes, flat)
        assert (_flat == result).all()

    # fmt: on


@pytest.mark.ext
@pytest.mark.extp
class TestExtBinPrev:
    # fmt: off
    @pytest.mark.parametrize(
        "flat,nbcl,extpol,q_classes,result",
        [
            ([np.array([0.2, 0]), np.array([0.8, 0])], 2, ExtensionPolicy(), [[0, 1], [0, 1]], np.array([[0.2, 0], [0.8, 0]])),
            ([np.array([0.2]), np.array([0.8])], 2, ExtensionPolicy(), [[0], [1]], np.array([[0.2, 0], [0, 0.8]])),
            ([np.array([0.2]), np.array([0.8])], 2, ExtensionPolicy(group_false=True), [[0], [1]], np.array([[0.2, 0], [0, 0.8]])),
            ([np.array([0.2]), np.array([0.8])], 2, ExtensionPolicy(collapse_false=True), [[0], [1]], np.array([[0.2, 0], [0, 0.8]])),
            ([np.array([0.1, 0.1]), np.array([0.6]), np.array([0.2])], 3, ExtensionPolicy(), [[0, 1], [0], [2]], np.array([[0.1, 0.1, 0], [0.6, 0, 0], [0, 0, 0.2]])),
            ([np.array([0.1, 0.1]), np.array([0.6]), np.array([0.2])], 3, ExtensionPolicy(group_false=True), [[0, 1], [0], [1]], np.array([[0.1, 0.1], [0.6, 0], [0, 0.2]])),
            ([np.array([0.1, 0.1]), np.array([0.6]), np.array([0.2])], 3, ExtensionPolicy(collapse_false=True), [[0, 1], [0], [2]], np.array([[0.1, 0.1, 0], [0.6, 0, 0], [0, 0, 0.2]])),
        ],
    )
    def test__check_q_classes(self, monkeypatch, flat, nbcl, extpol, q_classes, result):
        def mockinit(self, nbcl, extpol):
            self.nbcl = nbcl
            self.extpol = extpol

        monkeypatch.setattr(ExtBinPrev, "__init__", mockinit)
        ep = ExtBinPrev(nbcl, extpol)
        _flat = ep._ExtBinPrev__check_q_classes(q_classes, flat)
        assert (_flat == result).all()

    @pytest.mark.parametrize(
        "flat,result",
        [
            (np.array([[0.2, 0], [0.8, 0]]), np.array([0.2, 0.8, 0, 0])),
            (np.array([[0.2, 0], [0, 0.8]]), np.array([0.2, 0, 0, 0.8])),
            (np.array([[0.1, 0.1, 0], [0.6, 0, 0], [0, 0, 0.2]]), np.array([0.1, 0.6, 0, 0.1, 0, 0, 0, 0, 0.2])),
            (np.array([[0.1, 0.1], [0.6, 0], [0, 0.2]]), np.array([0.1, 0.6, 0, 0.1, 0, 0.2])),
        ],
    )
    def test__build_flat(self, monkeypatch, flat, result):
        def mockinit(self):
            pass

        monkeypatch.setattr(ExtBinPrev, "__init__", mockinit)
        ep = ExtBinPrev()
        _flat = ep._ExtBinPrev__build_flat(flat)
        assert (_flat == result).all()
    # fmt: on


@pytest.mark.ext
@pytest.mark.extc
class TestExtendedCollection:
    @pytest.mark.parametrize(
        "instances_name,labels,pred_proba,extpol,result",
        [
            (
                "nd_1",
                np.array([0, 1, 1, 0]),
                np.array([[0.2, 0.8], [0.3, 0.7], [0.9, 0.1], [0.45, 0.55]]),
                ExtensionPolicy(),
                np.array([0, 0.5, 0.25, 0.25]),
            ),
            (
                "nd_1",
                np.array([0, 1, 1, 0]),
                np.array([[0.2, 0.8], [0.3, 0.7], [0.9, 0.1], [0.45, 0.55]]),
                ExtensionPolicy(collapse_false=True),
                np.array([0, 0.25, 0.75]),
            ),
            (
                "csr_1",
                np.array([0, 1, 1, 0]),
                np.array([[0.2, 0.8], [0.3, 0.7], [0.9, 0.1], [0.45, 0.55]]),
                ExtensionPolicy(),
                np.array([0, 0.5, 0.25, 0.25]),
            ),
            (
                "csr_1",
                np.array([0, 1, 1, 0]),
                np.array([[0.2, 0.8], [0.3, 0.7], [0.9, 0.1], [0.45, 0.55]]),
                ExtensionPolicy(collapse_false=True),
                np.array([0, 0.25, 0.75]),
            ),
        ],
    )
    def test_prevalence(
        self, instances_name, labels, pred_proba, extpol, result, request
    ):
        instances = request.getfixturevalue(instances_name)
        ec = ExtendedCollection(
            instances=instances,
            labels=labels,
            pred_proba=pred_proba,
            ext=pred_proba,
            extpol=extpol,
        )
        assert (ec.prevalence() == result).all()
