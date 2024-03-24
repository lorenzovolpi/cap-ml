import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

import quacc as qc
import quacc.plot as plot
from quacc.utils import fmt_line_md


def _get_metric(metric: str):
    return slice(None) if metric is None else metric


def _get_estimators(estimators: List[str], cols: np.ndarray):
    if estimators is None:
        return slice(None)

    estimators = np.array(estimators)
    return estimators[np.isin(estimators, cols)]


def _get_shift(index: np.ndarray, train_prev: np.ndarray):
    index = np.array([np.array(tp) for tp in index])
    train_prevs = np.tile(train_prev, (index.shape[0], 1))
    # assert index.shape[1] == train_prev.shape[0], "Mismatch in prevalence shape"
    # _shift = np.abs(index - train_prev)[:, 1:].sum(axis=1)
    _shift = qc.error.nae(index, train_prevs)
    return np.around(_shift, decimals=2)


class EvaluationReport:
    def __init__(self, name=None):
        self.data: pd.DataFrame | None = None
        self.name = name if name is not None else "default"
        self.time = 0.0
        self.fit_score = None

    def append_row(self, basep: np.ndarray | Tuple, **row):
        # bp = basep[1]
        bp = tuple(basep)
        _keys, _values = zip(*row.items())
        # _keys = list(row.keys())
        # _values = list(row.values())

        if self.data is None:
            _idx = 0
            self.data = pd.DataFrame(
                {k: [v] for k, v in row.items()},
                index=pd.MultiIndex.from_tuples([(bp, _idx)]),
                columns=_keys,
            )
            return

        _idx = len(self.data.loc[(bp,), :]) if (bp,) in self.data.index else 0
        not_in_data = np.setdiff1d(list(row.keys()), self.data.columns.unique(0))
        self.data.loc[:, not_in_data] = np.nan
        self.data.loc[(bp, _idx), :] = row
        return

    @property
    def columns(self) -> np.ndarray:
        return self.data.columns.unique(0)

    @property
    def prevs(self):
        return np.sort(self.data.index.unique(0))


class CompReport:
    _default_modes = [
        "delta_train",
        "stdev_train",
        "train_table",
        "shift",
        "shift_table",
        "diagonal",
        "stats_table",
    ]

    def __init__(
        self,
        datas: List[EvaluationReport] | pd.DataFrame,
        name="default",
        train_prev: np.ndarray = None,
        valid_prev: np.ndarray = None,
        times=None,
        fit_scores=None,
        g_time=None,
    ):
        if isinstance(datas, pd.DataFrame):
            self._data: pd.DataFrame = datas
        else:
            self._data: pd.DataFrame = (
                pd.concat(
                    [er.data for er in datas],
                    keys=[er.name for er in datas],
                    axis=1,
                )
                .swaplevel(0, 1, axis=1)
                .sort_index(axis=1, level=0, sort_remaining=False)
                .sort_index(axis=0, level=0, ascending=False, sort_remaining=False)
            )

        if fit_scores is None:
            self.fit_scores = {
                er.name: er.fit_score for er in datas if er.fit_score is not None
            }
        else:
            self.fit_scores = fit_scores

        if times is None:
            self.times = {er.name: er.time for er in datas}
        else:
            self.times = times

        self.times["tot"] = g_time if g_time is not None else 0.0
        self.train_prev = train_prev
        self.valid_prev = valid_prev

    def postprocess(
        self,
        f_data: pd.DataFrame,
        _data: pd.DataFrame,
        metric=None,
        estimators=None,
    ) -> pd.DataFrame:
        _mapping = {
            "sld_lr_gs": [
                "bin_sld_lr_gs",
                "mul_sld_lr_gs",
                "m3w_sld_lr_gs",
            ],
            "kde_lr_gs": [
                "bin_kde_lr_gs",
                "mul_kde_lr_gs",
                "m3w_kde_lr_gs",
            ],
            "cc_lr_gs": [
                "bin_cc_lr_gs",
                "mul_cc_lr_gs",
                "m3w_cc_lr_gs",
            ],
            "QuAcc": [
                "bin_sld_lr_gs",
                "mul_sld_lr_gs",
                "m3w_sld_lr_gs",
                "bin_kde_lr_gs",
                "mul_kde_lr_gs",
                "m3w_kde_lr_gs",
            ],
        }

        for name, methods in _mapping.items():
            if estimators is not None and name not in estimators:
                continue

            available_idx = np.where(np.in1d(methods, self._data.columns.unique(1)))[0]
            if len(available_idx) == 0:
                continue
            methods = np.array(methods)[available_idx]

            _metric = _get_metric(metric)
            m_data = _data.loc[:, (_metric, methods)]
            _fit_scores = [(k, v) for (k, v) in self.fit_scores.items() if k in methods]
            _best_method = [k for k, v in _fit_scores][
                np.argmin([v for k, v in _fit_scores])
            ]
            _metric = (
                [_metric]
                if _metric is isinstance(_metric, str)
                else m_data.columns.unique(0)
            )
            for _m in _metric:
                f_data.loc[:, (_m, name)] = m_data.loc[:, (_m, _best_method)]

        return f_data

    @property
    def prevs(self) -> np.ndarray:
        return self.data().index.unique(0)

    def join(self, other, how="update", estimators=None):
        if how not in ["update"]:
            how = "update"

        if not (self.train_prev == other.train_prev).all():
            raise ValueError(
                f"self has train prev. {self.train_prev} while other has {other.train_prev}"
            )

        self_data = self.data(estimators=estimators)
        other_data = other.data(estimators=estimators)

        if not (self_data.index == other_data.index).all():
            raise ValueError("self and other have different indexes")

        update_col = self_data.columns.intersection(other_data.columns)
        other_join_col = other_data.columns.difference(update_col)

        _join = pd.concat(
            [self_data, other_data.loc[:, other_join_col.to_list()]],
            axis=1,
        )
        _join.loc[:, update_col.to_list()] = other_data.loc[:, update_col.to_list()]
        _join.sort_index(axis=1, level=0, sort_remaining=False, inplace=True)

        df = CompReport(
            _join,
            self.name if hasattr(self, "name") else "default",
            train_prev=self.train_prev,
            valid_prev=self.valid_prev,
            times=self.times | other.times,
            fit_scores=self.fit_scores | other.fit_scores,
            g_time=self.times["tot"] + other.times["tot"],
        )

        return df

    def data(self, metric: str = None, estimators: List[str] = None) -> pd.DataFrame:
        _metric = _get_metric(metric)
        _estimators = _get_estimators(
            estimators, self._data.loc[:, (_metric, slice(None))].columns.unique(1)
        )
        _data: pd.DataFrame = self._data.copy()
        f_data: pd.DataFrame = _data.loc[:, (_metric, _estimators)]

        f_data = self.postprocess(f_data, _data, metric=metric, estimators=estimators)

        if len(f_data.columns.unique(0)) == 1:
            f_data = f_data.droplevel(level=0, axis=1)

        return f_data

    def shift_data(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        shift_idx_0 = _get_shift(
            self._data.index.get_level_values(0).to_numpy(),
            self.train_prev,
        )

        shift_idx_1 = np.zeros(shape=shift_idx_0.shape[0], dtype="<i4")
        for _id in np.unique(shift_idx_0):
            _wh = (shift_idx_0 == _id).nonzero()[0]
            shift_idx_1[_wh] = np.arange(_wh.shape[0], dtype="<i4")

        shift_data = self._data.copy()
        shift_data.index = pd.MultiIndex.from_arrays([shift_idx_0, shift_idx_1])
        shift_data = shift_data.sort_index(axis=0, level=0)

        _metric = _get_metric(metric)
        _estimators = _get_estimators(
            estimators, shift_data.loc[:, (_metric, slice(None))].columns.unique(1)
        )
        s_data: pd.DataFrame = shift_data
        shift_data: pd.DataFrame = shift_data.loc[:, (_metric, _estimators)]
        shift_data = self.postprocess(
            shift_data, s_data, metric=metric, estimators=estimators
        )

        if len(shift_data.columns.unique(0)) == 1:
            shift_data = shift_data.droplevel(level=0, axis=1)

        return shift_data

    def avg_by_prevs(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_dict = self.data(metric=metric, estimators=estimators)
        return f_dict.groupby(level=0, sort=False).mean()

    def stdev_by_prevs(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_dict = self.data(metric=metric, estimators=estimators)
        return f_dict.groupby(level=0, sort=False).std()

    def train_table(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_data = self.data(metric=metric, estimators=estimators)
        avg_p = f_data.groupby(level=0, sort=False).mean()
        avg_p.loc["mean", :] = f_data.mean()
        return avg_p

    def shift_table(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_data = self.shift_data(metric=metric, estimators=estimators)
        avg_p = f_data.groupby(level=0, sort=False).mean()
        avg_p.loc["mean", :] = f_data.mean()
        return avg_p

    def get_plots(
        self,
        mode="delta_train",
        metric="acc",
        estimators=None,
        conf="default",
        save_fig=True,
        base_path=None,
        backend=None,
    ) -> List[Tuple[str, Path]]:
        if mode == "delta_train":
            avg_data = self.avg_by_prevs(metric=metric, estimators=estimators)
            if avg_data.empty:
                return None

            return plot.plot_delta(
                base_prevs=self.prevs,
                columns=avg_data.columns.to_numpy(),
                data=avg_data.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=self.train_prev,
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "stdev_train":
            avg_data = self.avg_by_prevs(metric=metric, estimators=estimators)
            if avg_data.empty is True:
                return None

            st_data = self.stdev_by_prevs(metric=metric, estimators=estimators)
            return plot.plot_delta(
                base_prevs=self.prevs,
                columns=avg_data.columns.to_numpy(),
                data=avg_data.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=self.train_prev,
                stdevs=st_data.T.to_numpy(),
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "diagonal":
            f_data = self.data(metric=metric + "_score", estimators=estimators)
            if f_data.empty is True:
                return None

            ref: pd.Series = f_data.loc[:, "ref"]
            f_data.drop(columns=["ref"], inplace=True)
            return plot.plot_diagonal(
                reference=ref.to_numpy(),
                columns=f_data.columns.to_numpy(),
                data=f_data.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=self.train_prev,
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "shift":
            _shift_data = self.shift_data(metric=metric, estimators=estimators)
            if _shift_data.empty is True:
                return None

            shift_avg = _shift_data.groupby(level=0, sort=False).mean()
            shift_counts = _shift_data.groupby(level=0, sort=False).count()
            shift_prevs = shift_avg.index.unique(0)
            # shift_prevs = np.around(
            #     [(1.0 - p, p) for p in np.sort(shift_avg.index.unique(0))],
            #     decimals=2,
            # )
            return plot.plot_shift(
                shift_prevs=shift_prevs,
                columns=shift_avg.columns.to_numpy(),
                data=shift_avg.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=self.train_prev,
                counts=shift_counts.T.to_numpy(),
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )

    def to_md(
        self,
        conf="default",
        metric="acc",
        estimators=None,
        modes=_default_modes,
        plot_path=None,
    ) -> str:
        res = f"## {int(np.around(self.train_prev, decimals=2)[1]*100)}% positives\n"
        res += fmt_line_md(f"train: {str(self.train_prev)}")
        res += fmt_line_md(f"validation: {str(self.valid_prev)}")
        for k, v in self.times.items():
            if estimators is not None and k not in estimators:
                continue
            res += fmt_line_md(f"{k}: {v:.3f}s")
        res += "\n"
        if "train_table" in modes:
            res += "### table\n"
            res += (
                self.train_table(metric=metric, estimators=estimators).to_html()
                + "\n\n"
            )
        if "shift_table" in modes:
            res += "### shift table\n"
            res += (
                self.shift_table(metric=metric, estimators=estimators).to_html()
                + "\n\n"
            )

        plot_modes = [m for m in modes if not m.endswith("table")]
        for mode in plot_modes:
            res += f"### {mode}\n"
            _, op = self.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf=conf,
                save_fig=True,
                base_path=plot_path,
            )
            res += f"![plot_{mode}]({op.relative_to(op.parents[1]).as_posix()})\n"

        return res


def _cr_train_prev(cr: CompReport):
    return tuple(np.around(cr.train_prev, decimals=2))


def _cr_data(cr: CompReport, metric=None, estimators=None):
    return cr.data(metric, estimators)


def _key_reverse_delta_train(idx):
    idx = idx.to_numpy()
    sorted_idx = np.array(
        sorted(list(idx), key=lambda x: x[-1]), dtype=("float," * len(idx[0]))[:-1]
    )
    # get sorting index
    nparr = np.nonzero(idx[:, None] == sorted_idx)[1]
    return nparr


class DatasetReport:
    _default_dr_modes = [
        "delta_train",
        "stdev_train",
        "train_table",
        "train_std_table",
        "shift",
        "shift_table",
        "delta_test",
        "stdev_test",
        "test_table",
        "diagonal",
        "stats_table",
        "fit_scores",
    ]
    _default_cr_modes = CompReport._default_modes

    def __init__(self, name, crs=None):
        self.name = name
        self.crs: List[CompReport] = [] if crs is None else crs

    def sort_delta_train_index(self, data):
        # data_ = data.sort_index(axis=0, level=0, ascending=True, sort_remaining=False)
        data_ = data.sort_index(
            axis=0,
            level=0,
            key=_key_reverse_delta_train,
        )
        print(data_.index)
        return data_

    def join(self, other, estimators=None):
        _crs = [
            s_cr.join(o_cr, estimators=estimators)
            for s_cr, o_cr in zip(self.crs, other.crs)
        ]

        return DatasetReport(self.name, _crs)

    def fit_scores(self, metric: str = None, estimators: List[str] = None):
        def _get_sort_idx(arr):
            return np.array([np.searchsorted(np.sort(a), a) + 1 for a in arr])

        def _get_best_idx(arr):
            return np.argmin(arr, axis=1)

        def _fdata_idx(idx) -> np.ndarray:
            return _fdata.loc[(idx, slice(None), slice(None)), :].to_numpy()

        _crs_train = [_cr_train_prev(cr) for cr in self.crs]

        for cr in self.crs:
            if not hasattr(cr, "fit_scores"):
                return None

        _crs_fit_scores = [cr.fit_scores for cr in self.crs]

        _fit_scores = pd.DataFrame(_crs_fit_scores, index=_crs_train)
        _fit_scores = _fit_scores.sort_index(axis=0, ascending=False)

        _estimators = _get_estimators(estimators, _fit_scores.columns)
        if _estimators.shape[0] == 0:
            return None

        _fdata = self.data(metric=metric, estimators=_estimators)

        # ensure that columns in _fit_scores have the same ordering of _fdata
        _fit_scores = _fit_scores.loc[:, _fdata.columns]

        _best_fit_estimators = _get_best_idx(_fit_scores.to_numpy())

        # scores = np.array(
        #     [
        #         _get_sort_idx(
        #             _fdata.loc[(idx, slice(None), slice(None)), :].to_numpy()
        #         )[:, cl].mean()
        #         for idx, cl in zip(_fit_scores.index, _best_fit_estimators)
        #     ]
        # )
        # for idx, cl in zip(_fit_scores.index, _best_fit_estimators):
        #     print(_fdata_idx(idx)[:, cl])
        #     print(_fdata_idx(idx).min(axis=1), end="\n\n")

        scores = np.array(
            [
                np.abs(_fdata_idx(idx)[:, cl] - _fdata_idx(idx).min(axis=1)).mean()
                for idx, cl in zip(_fit_scores.index, _best_fit_estimators)
            ]
        )

        return scores

    def data(self, metric: str = None, estimators: List[str] = None) -> pd.DataFrame:
        _crs_sorted = sorted(
            [(_cr_train_prev(cr), _cr_data(cr, metric, estimators)) for cr in self.crs],
            key=lambda cr: len(cr[1].columns),
            reverse=True,
        )
        _crs_train, _crs_data = zip(*_crs_sorted)

        _data: pd.DataFrame = pd.concat(
            _crs_data,
            axis=0,
            keys=_crs_train,
        )

        # The MultiIndex is recreated to make the outer-most level a tuple and not a
        # sequence of values
        _len_tr_idx = len(_crs_train[0])
        _idx = _data.index.to_list()
        _idx = pd.MultiIndex.from_tuples(
            [tuple([midx[:_len_tr_idx]] + list(midx[_len_tr_idx:])) for midx in _idx]
        )
        _data.index = _idx

        _data = _data.sort_index(axis=0, level=0, ascending=False, sort_remaining=False)

        return _data

    def shift_data(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        _shift_data: pd.DataFrame = pd.concat(
            sorted(
                [cr.shift_data(metric, estimators) for cr in self.crs],
                key=lambda d: len(d.columns),
                reverse=True,
            ),
            axis=0,
        )

        shift_idx_0 = _shift_data.index.get_level_values(0)

        shift_idx_1 = np.empty(shape=shift_idx_0.shape, dtype="<i4")
        for _id in np.unique(shift_idx_0):
            _wh = np.where(shift_idx_0 == _id)[0]
            shift_idx_1[_wh] = np.arange(_wh.shape[0])

        _shift_data.index = pd.MultiIndex.from_arrays([shift_idx_0, shift_idx_1])
        _shift_data = _shift_data.sort_index(axis=0, level=0)

        return _shift_data

    def add(self, cr: CompReport):
        if cr is None:
            return

        self.crs.append(cr)

    def __add__(self, cr: CompReport):
        if cr is None:
            return

        return DatasetReport(self.name, crs=self.crs + [cr])

    def __iadd__(self, cr: CompReport):
        self.add(cr)
        return self

    def train_table(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_data = self.data(metric=metric, estimators=estimators)
        avg_p = f_data.groupby(level=1, sort=False).mean()
        avg_p.loc["mean", :] = f_data.mean()
        return avg_p

    def train_std_table(self, metric: str = None, estimators: List[str] = None):
        f_data = self.data(metric=metric, estimators=estimators)
        avg_p = f_data.groupby(level=1, sort=False).mean()
        avg_p.loc["mean", :] = f_data.mean()
        avg_s = f_data.groupby(level=1, sort=False).std()
        avg_s.loc["mean", :] = f_data.std()
        avg_r = pd.concat([avg_p, avg_s], axis=1, keys=["avg", "std"])
        return avg_r

    def test_table(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_data = self.data(metric=metric, estimators=estimators)
        avg_p = f_data.groupby(level=0, sort=False).mean()
        avg_p.loc["mean", :] = f_data.mean()
        return avg_p

    def shift_table(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_data = self.shift_data(metric=metric, estimators=estimators)
        avg_p = f_data.groupby(level=0, sort=False).mean()
        avg_p.loc["mean", :] = f_data.mean()
        return avg_p

    def get_plots(
        self,
        data=None,
        mode="delta_train",
        metric="acc",
        estimators=None,
        conf="default",
        save_fig=True,
        base_path=None,
        backend=None,
    ):
        if mode == "delta_train":
            _data = self.data(metric, estimators) if data is None else data
            avg_on_train = _data.groupby(level=1, sort=False).mean()
            if avg_on_train.empty:
                return None
            # sort index in reverse order
            avg_on_train = self.sort_delta_train_index(avg_on_train)
            prevs_on_train = avg_on_train.index.unique(0)
            return plot.plot_delta(
                # base_prevs=np.around(
                #     [(1.0 - p, p) for p in prevs_on_train], decimals=2
                # ),
                base_prevs=prevs_on_train,
                columns=avg_on_train.columns.to_numpy(),
                data=avg_on_train.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=None,
                avg="train",
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "stdev_train":
            _data = self.data(metric, estimators) if data is None else data
            avg_on_train = _data.groupby(level=1, sort=False).mean()
            if avg_on_train.empty:
                return None
            prevs_on_train = avg_on_train.index.unique(0)
            stdev_on_train = _data.groupby(level=1, sort=False).std()
            return plot.plot_delta(
                # base_prevs=np.around(
                #     [(1.0 - p, p) for p in prevs_on_train], decimals=2
                # ),
                base_prevs=prevs_on_train,
                columns=avg_on_train.columns.to_numpy(),
                data=avg_on_train.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=None,
                stdevs=stdev_on_train.T.to_numpy(),
                avg="train",
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "delta_test":
            _data = self.data(metric, estimators) if data is None else data
            avg_on_test = _data.groupby(level=0, sort=False).mean()
            if avg_on_test.empty:
                return None
            prevs_on_test = avg_on_test.index.unique(0)
            return plot.plot_delta(
                # base_prevs=np.around([(1.0 - p, p) for p in prevs_on_test], decimals=2),
                base_prevs=prevs_on_test,
                columns=avg_on_test.columns.to_numpy(),
                data=avg_on_test.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=None,
                avg="test",
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "stdev_test":
            _data = self.data(metric, estimators) if data is None else data
            avg_on_test = _data.groupby(level=0, sort=False).mean()
            if avg_on_test.empty:
                return None
            prevs_on_test = avg_on_test.index.unique(0)
            stdev_on_test = _data.groupby(level=0, sort=False).std()
            return plot.plot_delta(
                # base_prevs=np.around([(1.0 - p, p) for p in prevs_on_test], decimals=2),
                base_prevs=prevs_on_test,
                columns=avg_on_test.columns.to_numpy(),
                data=avg_on_test.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=None,
                stdevs=stdev_on_test.T.to_numpy(),
                avg="test",
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "shift":
            _shift_data = self.shift_data(metric, estimators) if data is None else data
            avg_shift = _shift_data.groupby(level=0, sort=False).mean()
            if avg_shift.empty:
                return None
            count_shift = _shift_data.groupby(level=0, sort=False).count()
            prevs_shift = avg_shift.index.unique(0)
            return plot.plot_shift(
                # shift_prevs=np.around([(1.0 - p, p) for p in prevs_shift], decimals=2),
                shift_prevs=prevs_shift,
                columns=avg_shift.columns.to_numpy(),
                data=avg_shift.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=None,
                counts=count_shift.T.to_numpy(),
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "fit_scores":
            _fit_scores = self.fit_scores(metric, estimators) if data is None else data
            if _fit_scores is None:
                return None
            train_prevs = self.data(metric, estimators).index.unique(0)
            return plot.plot_fit_scores(
                train_prevs=train_prevs,
                scores=_fit_scores,
                metric=metric,
                name=conf,
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )
        elif mode == "diagonal":
            f_data = self.data(metric=metric + "_score", estimators=estimators)
            if f_data.empty:
                return None

            ref: pd.Series = f_data.loc[:, "ref"]
            f_data.drop(columns=["ref"], inplace=True)
            return plot.plot_diagonal(
                reference=ref.to_numpy(),
                columns=f_data.columns.to_numpy(),
                data=f_data.T.to_numpy(),
                metric=metric,
                name=conf,
                # train_prev=self.train_prev,
                fixed_lim=True,
                save_fig=save_fig,
                base_path=base_path,
                backend=backend,
            )

    def to_md(
        self,
        conf="default",
        metric="acc",
        estimators=[],
        dr_modes=_default_dr_modes,
        cr_modes=_default_cr_modes,
        cr_prevs: List[str] = None,
        plot_path=None,
    ):
        res = f"# {self.name}\n\n"
        for cr in self.crs:
            if (
                cr_prevs is not None
                and str(round(cr.train_prev[1] * 100)) not in cr_prevs
            ):
                continue
            _md = cr.to_md(
                conf,
                metric=metric,
                estimators=estimators,
                modes=cr_modes,
                plot_path=plot_path,
            )
            res += f"{_md}\n\n"

        _data = self.data(metric=metric, estimators=estimators)
        _shift_data = self.shift_data(metric=metric, estimators=estimators)

        res += "## avg\n"

        ######################## avg on train ########################
        res += "### avg on train\n"

        if "train_table" in dr_modes:
            avg_on_train_tbl = _data.groupby(level=1, sort=False).mean()
            avg_on_train_tbl.loc["avg", :] = _data.mean()
            res += avg_on_train_tbl.to_html() + "\n\n"

        if "delta_train" in dr_modes:
            _, delta_op = self.get_plots(
                data=_data,
                mode="delta_train",
                metric=metric,
                estimators=estimators,
                conf=conf,
                base_path=plot_path,
                save_fig=True,
            )
            _op = delta_op.relative_to(delta_op.parents[1]).as_posix()
            res += f"![plot_delta]({_op})\n"

        if "stdev_train" in dr_modes:
            _, delta_stdev_op = self.get_plots(
                data=_data,
                mode="stdev_train",
                metric=metric,
                estimators=estimators,
                conf=conf,
                base_path=plot_path,
                save_fig=True,
            )
            _op = delta_stdev_op.relative_to(delta_stdev_op.parents[1]).as_posix()
            res += f"![plot_delta_stdev]({_op})\n"

        ######################## avg on test ########################
        res += "### avg on test\n"

        if "test_table" in dr_modes:
            avg_on_test_tbl = _data.groupby(level=0, sort=False).mean()
            avg_on_test_tbl.loc["avg", :] = _data.mean()
            res += avg_on_test_tbl.to_html() + "\n\n"

        if "delta_test" in dr_modes:
            _, delta_op = self.get_plots(
                data=_data,
                mode="delta_test",
                metric=metric,
                estimators=estimators,
                conf=conf,
                base_path=plot_path,
                save_fig=True,
            )
            _op = delta_op.relative_to(delta_op.parents[1]).as_posix()
            res += f"![plot_delta]({_op})\n"

        if "stdev_test" in dr_modes:
            _, delta_stdev_op = self.get_plots(
                data=_data,
                mode="stdev_test",
                metric=metric,
                estimators=estimators,
                conf=conf,
                base_path=plot_path,
                save_fig=True,
            )
            _op = delta_stdev_op.relative_to(delta_stdev_op.parents[1]).as_posix()
            res += f"![plot_delta_stdev]({_op})\n"

        ######################## avg shift ########################
        res += "### avg dataset shift\n"

        if "shift_table" in dr_modes:
            shift_on_train_tbl = _shift_data.groupby(level=0, sort=False).mean()
            shift_on_train_tbl.loc["avg", :] = _shift_data.mean()
            res += shift_on_train_tbl.to_html() + "\n\n"

        if "shift" in dr_modes:
            _, shift_op = self.get_plots(
                data=_shift_data,
                mode="shift",
                metric=metric,
                estimators=estimators,
                conf=conf,
                base_path=plot_path,
                save_fig=True,
            )
            _op = shift_op.relative_to(shift_op.parents[1]).as_posix()
            res += f"![plot_shift]({_op})\n"

        return res

    def pickle(self, pickle_path: Path):
        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)

        return self

    @classmethod
    def unpickle(cls, pickle_path: Path, report_info=False):
        with open(pickle_path, "rb") as f:
            dr = pickle.load(f)

        if report_info:
            return DatasetReportInfo(dr, pickle_path)

        return dr

    def __iter__(self):
        return (cr for cr in self.crs)


class DatasetReportInfo:
    def __init__(self, dr: DatasetReport, path: Path):
        self.dr = dr
        self.name = str(path.parent)
        _data = dr.data()
        self.columns = defaultdict(list)
        for metric, estim in _data.columns:
            self.columns[estim].append(metric)
        # self.columns = list(_data.columns.unique(1))
        self.train_prevs = len(self.dr.crs)
        self.test_prevs = len(_data.index.unique(1))
        self.repeats = len(_data.index.unique(2))

    def __repr__(self) -> str:
        _d = {
            "train prevs.": self.train_prevs,
            "test prevs.": self.test_prevs,
            "repeats": self.repeats,
            "columns": self.columns,
        }
        _r = f"{self.name}\n{json.dumps(_d, indent=2)}\n"

        return _r
