from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from quacc import plot
from quacc.environment import env
from quacc.utils import fmt_line_md


def _get_metric(metric: str):
    return slice(None) if metric is None else metric


def _get_estimators(estimators: List[str], cols: np.ndarray):
    return slice(None) if estimators is None else cols[np.in1d(cols, estimators)]


class EvaluationReport:
    def __init__(self, name=None):
        self.data: pd.DataFrame = None
        self.fit_score = None
        self.name = name if name is not None else "default"

    def append_row(self, basep: np.ndarray | Tuple, **row):
        bp = basep[1]
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
    def __init__(
        self,
        reports: List[EvaluationReport],
        name="default",
        train_prev=None,
        valid_prev=None,
        times=None,
    ):
        self._data = (
            pd.concat(
                [er.data for er in reports],
                keys=[er.name for er in reports],
                axis=1,
            )
            .swaplevel(0, 1, axis=1)
            .sort_index(axis=1, level=0, sort_remaining=False)
            .sort_index(axis=0, level=0)
        )

        self.fit_scores = {
            er.name: er.fit_score for er in reports if er.fit_score is not None
        }
        self.train_prev = train_prev
        self.valid_prev = valid_prev
        self.times = times

    @property
    def prevs(self) -> np.ndarray:
        return np.sort(self._data.index.unique(0))

    @property
    def np_prevs(self) -> np.ndarray:
        return np.around([(1.0 - p, p) for p in self.prevs], decimals=2)

    def data(self, metric: str = None, estimators: List[str] = None) -> pd.DataFrame:
        _metric = _get_metric(metric)
        _estimators = _get_estimators(estimators, self._data.columns.unique(1))
        f_data: pd.DataFrame = self._data.copy().loc[:, (_metric, _estimators)]

        if len(f_data.columns.unique(0)) == 1:
            f_data = f_data.droplevel(level=0, axis=1)

        return f_data

    def shift_data(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        shift_idx_0 = np.around(
            np.abs(
                self._data.index.get_level_values(0).to_numpy() - self.train_prev[1]
            ),
            decimals=2,
        )

        shift_idx_1 = np.empty(shape=shift_idx_0.shape, dtype="<i4")
        for _id in np.unique(shift_idx_0):
            _wh = np.where(shift_idx_0 == _id)[0]
            shift_idx_1[_wh] = np.arange(_wh.shape[0], dtype="<i4")

        shift_data = self._data.copy()
        shift_data.index = pd.MultiIndex.from_arrays([shift_idx_0, shift_idx_1])
        shift_data.sort_index(axis=0, level=0)

        _metric = _get_metric(metric)
        _estimators = _get_estimators(estimators, shift_data.columns.unique(1))
        shift_data: pd.DataFrame = shift_data.loc[:, (_metric, _estimators)]

        if len(shift_data.columns.unique(0)) == 1:
            shift_data = shift_data.droplevel(level=0, axis=1)

        return shift_data

    def avg_by_prevs(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_dict = self.data(metric=metric, estimators=estimators)
        return f_dict.groupby(level=0).mean()

    def stdev_by_prevs(
        self, metric: str = None, estimators: List[str] = None
    ) -> pd.DataFrame:
        f_dict = self.data(metric=metric, estimators=estimators)
        return f_dict.groupby(level=0).std()

    def table(self, metric: str = None, estimators: List[str] = None) -> pd.DataFrame:
        f_data = self.data(metric=metric, estimators=estimators)
        avg_p = f_data.groupby(level=0).mean()
        avg_p.loc["avg", :] = f_data.mean()
        return avg_p

    def get_plots(
        self, mode="delta", metric="acc", estimators=None, conf="default", stdev=False
    ) -> List[Tuple[str, Path]]:
        if mode == "delta":
            avg_data = self.avg_by_prevs(metric=metric, estimators=estimators)
            return plot.plot_delta(
                base_prevs=self.np_prevs,
                columns=avg_data.columns.to_numpy(),
                data=avg_data.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=self.train_prev,
            )
        elif mode == "delta_stdev":
            avg_data = self.avg_by_prevs(metric=metric, estimators=estimators)
            st_data = self.stdev_by_prevs(metric=metric, estimators=estimators)
            return plot.plot_delta(
                base_prevs=self.np_prevs,
                columns=avg_data.columns.to_numpy(),
                data=avg_data.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=self.train_prev,
                stdevs=st_data.T.to_numpy(),
            )
        elif mode == "diagonal":
            f_data = self.data(metric=metric + "_score", estimators=estimators)
            ref: pd.Series = f_data.loc[:, "ref"]
            f_data.drop(columns=["ref"], inplace=True)
            return plot.plot_diagonal(
                reference=ref.to_numpy(),
                columns=f_data.columns.to_numpy(),
                data=f_data.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=self.train_prev,
            )
        elif mode == "shift":
            shift_data = (
                self.shift_data(metric=metric, estimators=estimators)
                .groupby(level=0)
                .mean()
            )
            shift_prevs = np.around(
                [(1.0 - p, p) for p in np.sort(shift_data.index.unique(0))],
                decimals=2,
            )
            return plot.plot_shift(
                shift_prevs=shift_prevs,
                columns=shift_data.columns.to_numpy(),
                data=shift_data.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=self.train_prev,
            )

    def to_md(self, conf="default", metric="acc", estimators=None, stdev=False) -> str:
        res = f"## {int(np.around(self.train_prev, decimals=2)[1]*100)}% positives\n"
        res += fmt_line_md(f"train: {str(self.train_prev)}")
        res += fmt_line_md(f"validation: {str(self.valid_prev)}")
        for k, v in self.times.items():
            res += fmt_line_md(f"{k}: {v:.3f}s")
        res += "\n"
        res += self.table(metric=metric, estimators=estimators).to_html() + "\n\n"

        plot_modes = np.array(["delta", "diagonal", "shift"], dtype="object")
        whd = np.where(plot_modes == "delta")[0]
        if len(whd) > 0:
            plot_modes = np.insert(plot_modes, whd + 1, "delta_stdev")
        for mode in plot_modes:
            op = self.get_plots(
                mode=mode,
                metric=metric,
                estimators=estimators,
                conf=conf,
                stdev=stdev,
            )
            res += f"![plot_{mode}]({op.relative_to(env.OUT_DIR).as_posix()})\n"

        return res


class DatasetReport:
    def __init__(self, name, crs=None):
        self.name = name
        self.crs: List[CompReport] = [] if crs is None else crs

    def data(self, metric: str = None, estimators: str = None) -> pd.DataFrame:
        def _cr_train_prev(cr: CompReport):
            return cr.train_prev[1]

        def _cr_data(cr: CompReport):
            return cr.data(metric, estimators)

        _crs_sorted = sorted(
            [(_cr_train_prev(cr), _cr_data(cr)) for cr in self.crs],
            key=lambda cr: len(cr[1].columns),
            reverse=True,
        )
        _crs_train, _crs_data = zip(*_crs_sorted)

        _data = pd.concat(_crs_data, axis=0, keys=_crs_train)
        _data = _data.sort_index(axis=0, level=0)
        return _data

    def shift_data(self, metric: str = None, estimators: str = None) -> pd.DataFrame:
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

    def to_md(self, conf="default", metric="acc", estimators=[], stdev=False):
        res = f"# {self.name}\n\n"
        for cr in self.crs:
            res += f"{cr.to_md(conf, metric=metric, estimators=estimators, stdev=stdev)}\n\n"

        _data = self.data(metric=metric, estimators=estimators)
        _shift_data = self.shift_data(metric=metric, estimators=estimators)

        avg_x_test = _data.groupby(level=1).mean()
        prevs_x_test = np.sort(avg_x_test.index.unique(0))
        stdev_x_test = _data.groupby(level=1).std() if stdev else None
        avg_x_test_tbl = _data.groupby(level=1).mean()
        avg_x_test_tbl.loc["avg", :] = _data.mean()

        avg_x_shift = _shift_data.groupby(level=0).mean()
        prevs_x_shift = np.sort(avg_x_shift.index.unique(0))

        res += "## avg\n"
        res += avg_x_test_tbl.to_html() + "\n\n"

        delta_op = plot.plot_delta(
            base_prevs=np.around([(1.0 - p, p) for p in prevs_x_test], decimals=2),
            columns=avg_x_test.columns.to_numpy(),
            data=avg_x_test.T.to_numpy(),
            metric=metric,
            name=conf,
            train_prev=None,
        )
        res += f"![plot_delta]({delta_op.relative_to(env.OUT_DIR).as_posix()})\n"

        if stdev:
            delta_stdev_op = plot.plot_delta(
                base_prevs=np.around([(1.0 - p, p) for p in prevs_x_test], decimals=2),
                columns=avg_x_test.columns.to_numpy(),
                data=avg_x_test.T.to_numpy(),
                metric=metric,
                name=conf,
                train_prev=None,
                fit_scores={k: np.mean(vals) for k, vals in self.fit_scores.items()},
                stdevs=stdev_x_test.T.to_numpy(),
            )
            res += f"![plot_delta_stdev]({delta_stdev_op.relative_to(env.OUT_DIR).as_posix()})\n"

        shift_op = plot.plot_shift(
            shift_prevs=np.around([(1.0 - p, p) for p in prevs_x_shift], decimals=2),
            columns=avg_x_shift.columns.to_numpy(),
            data=avg_x_shift.T.to_numpy(),
            metric=metric,
            name=conf,
            train_prev=None,
        )
        res += f"![plot_shift]({shift_op.relative_to(env.OUT_DIR).as_posix()})\n"

        return res

    def __iter__(self):
        return (cr for cr in self.crs)


def __test():
    df = None
    print(f"{df is None = }")
    if df is None:
        bp = 0.75
        idx = 0
        d = {"a": 0.0, "b": 0.1}
        df = pd.DataFrame(
            d,
            index=pd.MultiIndex.from_tuples([(bp, idx)]),
            columns=d.keys(),
        )
    print(df)
    print("-" * 100)

    bp = 0.75
    idx = len(df.loc[bp, :])
    df.loc[(bp, idx), :] = {"a": 0.2, "b": 0.3}
    print(df)
    print("-" * 100)

    bp = 0.90
    idx = len(df.loc[bp, :]) if bp in df.index else 0
    df.loc[(bp, idx), :] = {"a": 0.2, "b": 0.3}
    print(df)
    print("-" * 100)

    bp = 0.90
    idx = len(df.loc[bp, :]) if bp in df.index else 0
    d = {"a": 0.2, "v": 0.3, "e": 0.4}
    notin = np.setdiff1d(list(d.keys()), df.columns)
    df.loc[:, notin] = np.nan
    df.loc[(bp, idx), :] = d
    print(df)
    print("-" * 100)

    bp = 0.90
    idx = len(df.loc[bp, :]) if bp in df.index else 0
    d = {"a": 0.3, "v": 0.4, "e": 0.5}
    notin = np.setdiff1d(list(d.keys()), df.columns)
    print(f"{notin = }")
    df.loc[:, notin] = np.nan
    df.loc[(bp, idx), :] = d
    print(df)
    print("-" * 100)
    print(f"{np.sort(np.unique(df.index.get_level_values(0))) = }")
    print("-" * 100)

    print(f"{df.loc[(0.75, ),:] = }\n")
    print(f"{df.loc[(slice(None), 1),:] = }")
    print("-" * 100)

    print(f"{(0.75, ) in df.index = }")
    print(f"{(0.7, ) in df.index = }")
    print("-" * 100)

    df1 = pd.DataFrame(
        {
            "a": np.linspace(0.0, 1.0, 6),
            "b": np.linspace(1.0, 2.0, 6),
            "e": np.linspace(2.0, 3.0, 6),
            "v": np.linspace(0.0, 1.0, 6),
        },
        index=pd.MultiIndex.from_product([[0.75, 0.9], [0, 1, 2]]),
        columns=["a", "b", "e", "v"],
    )

    df2 = (
        pd.concat([df, df1], keys=["a", "b"], axis=1)
        .swaplevel(0, 1, axis=1)
        .sort_index(axis=1, level=0)
    )
    df3 = pd.concat([df1, df], keys=["b", "a"], axis=1)
    print(df)
    print(df1)
    print(df2)
    print(df3)
    df = df3
    print("-" * 100)

    print(df.loc[:, ("b", ["e", "v"])])
    print(df.loc[:, (slice(None), ["e", "v"])])
    print(df.loc[:, ("b", slice(None))])
    print(df.loc[:, ("b", slice(None))].droplevel(level=0, axis=1))
    print(df.loc[:, (slice(None), ["e", "v"])].droplevel(level=0, axis=1))
    print(len(df.loc[:, ("b", slice(None))].columns.unique(0)))
    print("-" * 100)

    idx_0 = np.around(np.abs(df.index.get_level_values(0).to_numpy() - 0.8), decimals=2)
    midx = pd.MultiIndex.from_arrays([idx_0, df.index.get_level_values(1)])
    print(midx)
    dfs = df.copy()
    dfs.index = midx
    print(df)
    print(dfs)
    print("-" * 100)

    df.loc[(0.85, 0), :] = np.linspace(0, 1, 8)
    df.loc[(0.85, 1), :] = np.linspace(0, 1, 8)
    df.loc[(0.85, 2), :] = np.linspace(0, 1, 8)
    idx_0 = np.around(np.abs(df.index.get_level_values(0).to_numpy() - 0.8), decimals=2)
    print(np.where(idx_0 == 0.05))
    idx_1 = np.empty(shape=idx_0.shape, dtype="<i4")
    print(idx_1)
    for _id in np.unique(idx_0):
        wh = np.where(idx_0 == _id)[0]
        idx_1[wh] = np.arange(wh.shape[0])
    midx = pd.MultiIndex.from_arrays([idx_0, idx_1])
    dfs = df.copy()
    dfs.index = midx
    dfs.sort_index(level=0, axis=0, inplace=True)
    print(df)
    print(dfs)
    print("-" * 100)

    print(np.sort(dfs.index.unique(0)))
    print("-" * 100)

    print(df.groupby(level=0).mean())
    print(dfs.groupby(level=0).mean())
    print("-" * 100)

    s = df.mean(axis=0)
    dfa = df.groupby(level=0).mean()
    dfa.loc["avg", :] = s
    print(dfa)
    print("-" * 100)

    print(df)
    dfn = df.loc[:, (slice(None), slice(None))]
    print(dfn)
    print(f"{df is dfn = }")
    print("-" * 100)

    a = np.array(["abc", "bcd", "cde", "bcd"], dtype="object")
    print(a)
    whb = np.where(a == "bcd")[0]
    if len(whb) > 0:
        a = np.insert(a, whb + 1, "pippo")
    print(a)
    print("-" * 100)

    dff: pd.DataFrame = df.loc[:, ("a",)]
    print(dff.to_dict(orient="list"))
    dff = dff.drop(columns=["v"])
    print(dff)
    s: pd.Series = dff.loc[:, "e"]
    print(s)
    print(s.to_numpy())
    print(type(s.to_numpy()))
    print("-" * 100)

    df3 = pd.concat([df, df], axis=0, keys=[0.5, 0.3]).sort_index(axis=0, level=0)
    print(df3)
    df3n = pd.concat([df, df], axis=0).sort_index(axis=0, level=0)
    print(df3n)
    df = df3
    print("-" * 100)

    print(df.groupby(level=1).mean(), df.groupby(level=1).count())
    print("-" * 100)

    print(df)
    for ls in df.T.to_numpy():
        print(ls)
    print("-" * 100)


if __name__ == "__main__":
    __test()
