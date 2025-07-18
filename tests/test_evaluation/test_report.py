import numpy as np
import pytest

from quacc.legacy.evaluation.report import (
    CompReport,
    DatasetReport,
    EvaluationReport,
    _get_shift,
)


@pytest.fixture
def empty_er():
    return EvaluationReport("empty")


@pytest.fixture
def er_list():
    er1 = EvaluationReport("er1")
    er1.append_row(np.array([0.2, 0.8]), **dict(acc=0.9, acc_score=0.1))
    er1.append_row(np.array([0.2, 0.8]), **dict(acc=0.6, acc_score=0.4))
    er1.append_row(np.array([0.3, 0.7]), **dict(acc=0.7, acc_score=0.3))
    er2 = EvaluationReport("er2")
    er2.append_row(np.array([0.2, 0.8]), **dict(acc=0.9, acc_score=0.1))
    er2.append_row(
        np.array([0.2, 0.8]), **dict(acc=0.6, acc_score=0.4, f1=0.9, f1_score=0.6)
    )
    er2.append_row(np.array([0.4, 0.6]), **dict(acc=0.7, acc_score=0.3))
    return [er1, er2]


@pytest.fixture
def er_list2():
    er1 = EvaluationReport("er12")
    er1.append_row(np.array([0.2, 0.8]), **dict(acc=0.9, acc_score=0.1))
    er1.append_row(np.array([0.2, 0.8]), **dict(acc=0.6, acc_score=0.4))
    er1.append_row(np.array([0.3, 0.7]), **dict(acc=0.7, acc_score=0.3))
    er2 = EvaluationReport("er2")
    er2.append_row(np.array([0.2, 0.8]), **dict(acc=0.9, acc_score=0.1))
    er2.append_row(
        np.array([0.2, 0.8]), **dict(acc=0.6, acc_score=0.4, f1=0.9, f1_score=0.6)
    )
    er2.append_row(np.array([0.4, 0.6]), **dict(acc=0.8, acc_score=0.3))
    return [er1, er2]


@pytest.fixture
def er_list3():
    er1 = EvaluationReport("er31")
    er1.append_row(np.array([0.2, 0.5, 0.3]), **dict(acc=0.9, acc_score=0.1))
    er1.append_row(np.array([0.2, 0.4, 0.4]), **dict(acc=0.6, acc_score=0.4))
    er1.append_row(np.array([0.3, 0.6, 0.1]), **dict(acc=0.7, acc_score=0.3))
    er2 = EvaluationReport("er32")
    er2.append_row(np.array([0.2, 0.5, 0.3]), **dict(acc=0.9, acc_score=0.1))
    er2.append_row(
        np.array([0.2, 0.5, 0.3]), **dict(acc=0.6, acc_score=0.4, f1=0.9, f1_score=0.6)
    )
    er2.append_row(np.array([0.3, 0.3, 0.4]), **dict(acc=0.8, acc_score=0.3))
    return [er1, er2]


@pytest.fixture
def cr_1(er_list):
    return CompReport(
        er_list,
        "cr_1",
        train_prev=np.array([0.2, 0.8]),
        valid_prev=np.array([0.25, 0.75]),
        g_time=0.0,
    )


@pytest.fixture
def cr_2(er_list2):
    return CompReport(
        er_list2,
        "cr_2",
        train_prev=np.array([0.3, 0.7]),
        valid_prev=np.array([0.35, 0.65]),
        g_time=0.0,
    )


@pytest.fixture
def cr_3(er_list3):
    return CompReport(
        er_list3,
        "cr_3",
        train_prev=np.array([0.4, 0.1, 0.5]),
        valid_prev=np.array([0.45, 0.25, 0.2]),
        g_time=0.0,
    )


@pytest.fixture
def cr_4(er_list3):
    return CompReport(
        er_list3,
        "cr_4",
        train_prev=np.array([0.5, 0.1, 0.4]),
        valid_prev=np.array([0.45, 0.25, 0.2]),
        g_time=0.0,
    )


@pytest.fixture
def dr_1(cr_1, cr_2):
    return DatasetReport("dr_1", [cr_1, cr_2])


@pytest.fixture
def dr_2(cr_3, cr_4):
    return DatasetReport("dr_2", [cr_3, cr_4])


@pytest.mark.rep
@pytest.mark.mrep
class TestReport:
    @pytest.mark.parametrize(
        "cr_name,train_prev,shift",
        [
            (
                "cr_1",
                np.array([0.2, 0.8]),
                np.array([0.2, 0.1, 0.0, 0.0]),
            ),
            (
                "cr_3",
                np.array([0.2, 0.5, 0.3]),
                np.array([0.2, 0.2, 0.0, 0.0, 0.1]),
            ),
        ],
    )
    def test_get_shift(self, cr_name, train_prev, shift, request):
        cr = request.getfixturevalue(cr_name)
        assert (
            _get_shift(cr._data.index.get_level_values(0), train_prev) == shift
        ).all()


@pytest.mark.rep
@pytest.mark.erep
class TestEvaluationReport:
    def test_init(self, empty_er):
        assert empty_er.data is None

    @pytest.mark.parametrize(
        "rows,index,columns,data",
        [
            (
                [
                    (np.array([0.2, 0.8]), dict(acc=0.9, acc_score=0.1)),
                    (np.array([0.2, 0.8]), dict(acc=0.6, acc_score=0.4)),
                    (np.array([0.3, 0.7]), dict(acc=0.7, acc_score=0.3)),
                ],
                [((0.2, 0.8), 0), ((0.2, 0.8), 1), ((0.3, 0.7), 0)],
                ["acc", "acc_score"],
                np.array([[0.9, 0.1], [0.6, 0.4], [0.7, 0.3]]),
            ),
        ],
    )
    def test_append_row(self, empty_er, rows, index, columns, data):
        er: EvaluationReport = empty_er
        for prev, r in rows:
            er.append_row(prev, **r)
        assert er.data.index.to_list() == index
        assert er.data.columns.to_list() == columns
        assert (er.data.to_numpy() == data).all()


@pytest.mark.rep
@pytest.mark.crep
class TestCompReport:
    @pytest.mark.parametrize(
        "train_prev,valid_prev,index,columns",
        [
            (
                np.array([0.2, 0.8]),
                np.array([0.25, 0.75]),
                [
                    ((0.4, 0.6), 0),
                    ((0.3, 0.7), 0),
                    ((0.2, 0.8), 0),
                    ((0.2, 0.8), 1),
                ],
                [
                    ("acc", "er1"),
                    ("acc", "er2"),
                    ("acc_score", "er1"),
                    ("acc_score", "er2"),
                    ("f1", "er2"),
                    ("f1_score", "er2"),
                ],
            )
        ],
    )
    def test_init(self, er_list, train_prev, valid_prev, index, columns):
        cr = CompReport(er_list, "cr", train_prev, valid_prev, g_time=0.0)
        assert cr._data.index.to_list() == index
        assert cr._data.columns.to_list() == columns
        assert (cr.train_prev == train_prev).all()
        assert (cr.valid_prev == valid_prev).all()

    @pytest.mark.parametrize(
        "cr_name,prev",
        [
            ("cr_1", [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_2", [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            (
                "cr_3",
                [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)],
            ),
            (
                "cr_4",
                [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)],
            ),
        ],
    )
    def test_prevs(self, cr_name, prev, request):
        cr = request.getfixturevalue(cr_name)
        assert cr.prevs.tolist() == prev

    def test_join(self, er_list, er_list2):
        tp = np.array([0.2, 0.8])
        vp = np.array([0.25, 0.75])
        cr1 = CompReport(er_list, "cr1", train_prev=tp, valid_prev=vp)
        cr2 = CompReport(er_list2, "cr2", train_prev=tp, valid_prev=vp)
        crj = cr1.join(cr2)
        _loc = crj._data.loc[((0.4, 0.6), 0), ("acc", "er2")].to_numpy()
        assert (_loc == np.array([0.8])).all()

    @pytest.mark.parametrize(
        "cr_name,metric,estimators,columns",
        [
            ("cr_1", "acc", None, ["er1", "er2"]),
            ("cr_1", "acc", ["er1"], ["er1"]),
            ("cr_1", "acc", ["er1", "er2"], ["er1", "er2"]),
            ("cr_1", "f1", None, ["er2"]),
            ("cr_1", "f1", ["er2"], ["er2"]),
            ("cr_3", "acc", None, ["er31", "er32"]),
            ("cr_3", "acc", ["er31"], ["er31"]),
            ("cr_3", "acc", ["er31", "er32"], ["er31", "er32"]),
            ("cr_3", "f1", None, ["er32"]),
            ("cr_3", "f1", ["er32"], ["er32"]),
        ],
    )
    def test_data(self, cr_name, metric, estimators, columns, request):
        cr = request.getfixturevalue(cr_name)
        _data = cr.data(metric=metric, estimators=estimators)
        assert _data.columns.to_list() == columns
        assert all(_data.index == cr._data.index)

    # fmt: off
    @pytest.mark.parametrize(
        "cr_name,metric,estimators,columns,index",
        [
            ("cr_1", "acc", None, ["er1", "er2"], [(0.0, 0), (0.0, 1), (0.1, 0), (0.2, 0)]),
            ("cr_1", "acc", ["er1"], ["er1"], [(0.0, 0), (0.0, 1), (0.1, 0), (0.2, 0)]),
            ("cr_1", "acc", ["er1", "er2"], ["er1", "er2"], [(0.0, 0), (0.0, 1), (0.1, 0), (0.2, 0)]),
            ("cr_1", "f1", None, ["er2"], [(0.0, 0), (0.0, 1), (0.1, 0), (0.2, 0)]),
            ("cr_1", "f1", ["er2"], ["er2"], [(0.0, 0), (0.0, 1), (0.1, 0), (0.2, 0)]),
            ("cr_3", "acc", None, ["er31", "er32"], [(0.2, 0), (0.3, 0), (0.4, 0), (0.4, 1), (0.5,0)]),
            ("cr_3", "acc", ["er31"], ["er31"], [(0.2, 0), (0.3, 0), (0.4, 0), (0.4, 1), (0.5,0)]),
            ("cr_3", "acc", ["er31", "er32"], ["er31", "er32"], [(0.2, 0), (0.3, 0), (0.4, 0), (0.4, 1), (0.5,0)]),
            ("cr_3", "f1", None, ["er32"], [(0.2, 0), (0.3, 0), (0.4, 0), (0.4, 1), (0.5,0)]),
            ("cr_3", "f1", ["er32"], ["er32"], [(0.2, 0), (0.3, 0), (0.4, 0), (0.4, 1), (0.5,0)]),
        ],
    )
    def test_shift_data(self, cr_name, metric, estimators, columns, index, request):
        cr = request.getfixturevalue(cr_name)
        _data = cr.shift_data(metric=metric, estimators=estimators)
        assert _data.columns.to_list() == columns
        assert _data.index.to_list() == index

    @pytest.mark.parametrize(
        "cr_name,metric,estimators,columns,index",
        [
            ("cr_1", "acc", None, ["er1", "er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_1", "acc", ["er1"], ["er1"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_1", "acc", ["er1", "er2"], ["er1", "er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_1", "f1", None, ["er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_1", "f1", ["er2"], ["er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_3", "acc", None, ["er31", "er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
            ("cr_3", "acc", ["er31"], ["er31"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
            ("cr_3", "acc", ["er31", "er32"], ["er31", "er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
            ("cr_3", "f1", None, ["er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
            ("cr_3", "f1", ["er32"], ["er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
        ],
    )
    def test_avg_by_prevs(self, cr_name, metric, estimators, columns, index, request):
        cr = request.getfixturevalue(cr_name)
        _data = cr.avg_by_prevs(metric=metric, estimators=estimators)
        assert _data.columns.to_list() == columns
        assert _data.index.to_list() == index

    @pytest.mark.parametrize(
        "cr_name,metric,estimators,columns,index",
        [
            ("cr_1", "acc", None, ["er1", "er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_1", "acc", ["er1"], ["er1"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_1", "acc", ["er1", "er2"], ["er1", "er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_1", "f1", None, ["er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_1", "f1", ["er2"], ["er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8)]),
            ("cr_3", "acc", None, ["er31", "er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
            ("cr_3", "acc", ["er31"], ["er31"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
            ("cr_3", "acc", ["er31", "er32"], ["er31", "er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
            ("cr_3", "f1", None, ["er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
            ("cr_3", "f1", ["er32"], ["er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4)]),
        ],
    )
    def test_stdev_by_prevs(self, cr_name, metric, estimators, columns, index, request):
        cr = request.getfixturevalue(cr_name)
        _data = cr.stdev_by_prevs(metric=metric, estimators=estimators)
        assert _data.columns.to_list() == columns
        assert _data.index.to_list() == index

    @pytest.mark.parametrize(
        "cr_name,metric,estimators,columns,index",
        [
            ("cr_1", "acc", None, ["er1", "er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), "mean"]),
            ("cr_1", "acc", ["er1"], ["er1"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), "mean"]),
            ("cr_1", "acc", ["er1", "er2"], ["er1", "er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), "mean"]),
            ("cr_1", "f1", None, ["er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), "mean"]),
            ("cr_1", "f1", ["er2"], ["er2"], [(0.4, 0.6), (0.3, 0.7), (0.2, 0.8), "mean"]),
            ("cr_3", "acc", None, ["er31", "er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4), "mean"]),
            ("cr_3", "acc", ["er31"], ["er31"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4), "mean"]),
            ("cr_3", "acc", ["er31", "er32"], ["er31", "er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4), "mean"]),
            ("cr_3", "f1", None, ["er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4), "mean"]),
            ("cr_3", "f1", ["er32"], ["er32"], [(0.3, 0.6, 0.1), (0.3, 0.3, 0.4), (0.2, 0.5, 0.3), (0.2, 0.4, 0.4), "mean"]),
        ],
    )
    def test_train_table(self, cr_name, metric, estimators, columns, index, request):
        cr = request.getfixturevalue(cr_name)
        _data = cr.train_table(metric=metric, estimators=estimators)
        assert _data.columns.to_list() == columns
        assert _data.index.to_list() == index

    @pytest.mark.parametrize(
        "cr_name,metric,estimators,columns,index",
        [
            ("cr_1", "acc", None, ["er1", "er2"], [0.0, 0.1, 0.2, "mean"]),
            ("cr_1", "acc", ["er1"], ["er1"], [0.0, 0.1, 0.2, "mean"]),
            ("cr_1", "acc", ["er1", "er2"], ["er1", "er2"], [0.0, 0.1, 0.2, "mean"]),
            ("cr_1", "f1", None, ["er2"], [0.0, 0.1, 0.2, "mean"]),
            ("cr_1", "f1", ["er2"], ["er2"], [0.0, 0.1, 0.2, "mean"]),
            ("cr_3", "acc", None, ["er31", "er32"], [0.2, 0.3, 0.4, 0.5, "mean"]),
            ("cr_3", "acc", ["er31"], ["er31"], [0.2, 0.3, 0.4, 0.5, "mean"]),
            ("cr_3", "acc", ["er31", "er32"], ["er31", "er32"], [0.2, 0.3, 0.4, 0.5, "mean"]),
            ("cr_3", "f1", None, ["er32"], [0.2, 0.3, 0.4, 0.5, "mean"]),
            ("cr_3", "f1", ["er32"], ["er32"], [0.2, 0.3, 0.4, 0.5, "mean"]),
        ],
    )
    def test_shift_table(self, cr_name, metric, estimators, columns, index, request):
        cr = request.getfixturevalue(cr_name)
        _data = cr.shift_table(metric=metric, estimators=estimators)
        assert _data.columns.to_list() == columns
        assert _data.index.to_list() == index
    # fmt: on


@pytest.mark.rep
@pytest.mark.drep
class TestDatasetReport:
    # fmt: off
    @pytest.mark.parametrize(
        "dr_name,metric,estimators,columns,index",
        [
            (
                "dr_1", "acc", None, ["er1", "er2", "er12"],
                [
                    ((0.3, 0.7), (0.4, 0.6), 0),
                    ((0.3, 0.7), (0.3, 0.7), 0),
                    ((0.3, 0.7), (0.2, 0.8), 0),
                    ((0.3, 0.7), (0.2, 0.8), 1),
                    ((0.2, 0.8), (0.4, 0.6), 0),
                    ((0.2, 0.8), (0.3, 0.7), 0),
                    ((0.2, 0.8), (0.2, 0.8), 0),
                    ((0.2, 0.8), (0.2, 0.8), 1),
                ],
            ),
            (
                "dr_2", "acc", None, ["er31", "er32"],
                [
                    ((0.5, 0.1, 0.4), (0.3, 0.6, 0.1), 0),
                    ((0.5, 0.1, 0.4), (0.3, 0.3, 0.4), 0),
                    ((0.5, 0.1, 0.4), (0.2, 0.5, 0.3), 0),
                    ((0.5, 0.1, 0.4), (0.2, 0.5, 0.3), 1),
                    ((0.5, 0.1, 0.4), (0.2, 0.4, 0.4), 0),
                    ((0.4, 0.1, 0.5), (0.3, 0.6, 0.1), 0),
                    ((0.4, 0.1, 0.5), (0.3, 0.3, 0.4), 0),
                    ((0.4, 0.1, 0.5), (0.2, 0.5, 0.3), 0),
                    ((0.4, 0.1, 0.5), (0.2, 0.5, 0.3), 1),
                    ((0.4, 0.1, 0.5), (0.2, 0.4, 0.4), 0),
                ],
            ),
        ],
    )
    def test_data(self, dr_name, metric, estimators, columns, index, request):
        dr = request.getfixturevalue(dr_name)
        _data = dr.data(metric=metric, estimators=estimators)
        assert _data.columns.to_list() == columns
        assert _data.index.to_list() == index

    @pytest.mark.parametrize(
        "dr_name,metric,estimators,columns,index",
        [
            (
                "dr_1", "acc", None, ["er1", "er2", "er12"],
                [(0.0, 0), (0.0, 1), (0.0, 2), (0.1, 0), 
                 (0.1, 1), (0.1, 2), (0.1, 3), (0.2, 0)],
            ),
            (
                "dr_2", "acc", None, ["er31", "er32"],
                [(0.2, 0), (0.2, 1), (0.3, 0), (0.3, 1), (0.4, 0), 
                 (0.4, 1), (0.4, 2), (0.4, 3), (0.5, 0), (0.5, 1)],
            ),
        ],
    )
    def test_shift_data(self, dr_name, metric, estimators, columns, index, request):
        dr = request.getfixturevalue(dr_name)
        _data = dr.shift_data(metric=metric, estimators=estimators)
        print(_data.index.tolist())
        assert _data.columns.to_list() == columns
        assert _data.index.to_list() == index
    # fmt: off
