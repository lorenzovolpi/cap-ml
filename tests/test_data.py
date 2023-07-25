import pytest
from quacc.data import ExClassManager as ECM, ExtendedCollection
import numpy as np


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
        "instances,labels,inst0,lbl0,inst1,lbl1",
        [
            (
                [[0, 0.3, 0.7], [1, 0.54, 0.46], [2, 0.28, 0.72], [3, 0.6, 0.4]],
                [3, 0, 1, 2],
                [[1, 0.54, 0.46], [3, 0.6, 0.4]],
                [0, 1],
                [[0, 0.3, 0.7], [2, 0.28, 0.72]],
                [1, 0],
            ),
            (
                [[0, 0.3, 0.7], [2, 0.28, 0.72]],
                [3, 1],
                [],
                [],
                [[0, 0.3, 0.7], [2, 0.28, 0.72]],
                [1, 0],
            ),
            (
                [[1, 0.54, 0.46], [3, 0.6, 0.4]],
                [0, 2],
                [[1, 0.54, 0.46], [3, 0.6, 0.4]],
                [0, 1],
                [],
                [],
            ),

        ],
    )
    def test_split_by_pred(self, instances, labels, inst0, lbl0, inst1, lbl1):
        ec = ExtendedCollection(
            np.asarray(instances), np.asarray(labels), classes=range(0, 4)
        )
        [ec0, ec1] = ec.split_by_pred()
        print(ec0.X, np.asarray(inst0))
        assert( np.array_equal(ec0.X, np.asarray(inst0)) )
        print(ec0.y, np.asarray(lbl0))
        assert( np.array_equal(ec0.y, np.asarray(lbl0)) )
        print(ec1.X, np.asarray(inst1))
        assert( np.array_equal(ec1.X, np.asarray(inst1)) )
        print(ec1.y, np.asarray(lbl1))
        assert( np.array_equal(ec1.y, np.asarray(lbl1)) )
        

        
        
