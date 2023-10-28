from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from quacc import plot
from quacc.environment import env
from quacc.utils import fmt_line_md


class EvaluationReport:
    def __init__(self, name=None):
        self._prevs = []
        self._dict = {}
        self.fit_score = None
        self.name = name if name is not None else "default"

    def append_row(self, basep: np.ndarray | Tuple, **row):
        bp = basep[1]
        self._prevs.append(bp)
        for k, v in row.items():
            if k not in self._dict:
                self._dict[k] = {}
            if bp not in self._dict[k]:
                self._dict[k][bp] = []
            self._dict[k][bp] = np.append(self._dict[k][bp], [v])

    @property
    def columns(self):
        return self._dict.keys()

    @property
    def prevs(self):
        return np.sort(np.unique([list(self._dict[_k].keys()) for _k in self._dict]))

    # def group_by_prevs(self, metric: str = None, estimators: List[str] = None):
    #     if self._g_dict is None:
    #         self._g_prevs = []
    #         self._g_dict = {k: [] for k in self._dict.keys()}

    #         for col, vals in self._dict.items():
    #             col_grouped = {}
    #             for bp, v in zip(self._prevs, vals):
    #                 if bp not in col_grouped:
    #                     col_grouped[bp] = []
    #                 col_grouped[bp].append(v)

    #             self._g_dict[col] = [
    #                 vs
    #                 for bp, vs in sorted(col_grouped.items(), key=lambda cg: cg[0][1])
    #             ]

    #         self._g_prevs = sorted(
    #             [(p0, p1) for [p0, p1] in np.unique(self._prevs, axis=0).tolist()],
    #             key=lambda bp: bp[1],
    #         )

    #     fg_dict = _filter_dict(self._g_dict, metric, estimators)
    #     return self._g_prevs, fg_dict

    # def merge(self, other):
    #     if not all(v1 == v2 for v1, v2 in zip(self._prevs, other._prevs)):
    #         raise ValueError("other has not same base prevalences of self")

    #     inters_keys = set(self._dict.keys()).intersection(set(other._dict.keys()))
    #     if len(inters_keys) > 0:
    #         raise ValueError(f"self and other have matching keys {str(inters_keys)}.")

    #     report = EvaluationReport()
    #     report._prevs = self._prevs
    #     report._dict = self._dict | other._dict
    #     return report


class CompReport:
    def __init__(
        self,
        reports: List[EvaluationReport],
        name="default",
        train_prev=None,
        valid_prev=None,
        times=None,
    ):
        all_prevs = np.array([er.prevs for er in reports])
        if not np.all(all_prevs == all_prevs[0, :], axis=0).all():
            raise ValueError(
                "Not all evaluation reports have the same base prevalences"
            )
        uq_names, name_c = np.unique([er.name for er in reports], return_counts=True)
        if np.sum(name_c) > uq_names.shape[0]:
            _matching = uq_names[[c > 1 for c in name_c]]
            raise ValueError(
                f"Evaluation reports have matching names: {_matching.tolist()}."
            )

        all_dicts = [{(k, er.name): v for k, v in er._dict.items()} for er in reports]
        self._dict = {}
        for d in all_dicts:
            self._dict = self._dict | d

        self.fit_scores = {
            er.name: er.fit_score for er in reports if er.fit_score is not None
        }
        self.train_prev = train_prev
        self.valid_prev = valid_prev
        self.times = times

    @property
    def prevs(self):
        return np.sort(np.unique([list(self._dict[_k].keys()) for _k in self._dict]))

    @property
    def cprevs(self):
        return np.around([(1.0 - p, p) for p in self.prevs], decimals=2)

    def data(self, metric: str = None, estimators: List[str] = None) -> dict:
        f_dict = self._dict.copy()
        if metric is not None:
            f_dict = {(c0, c1): ls for ((c0, c1), ls) in f_dict.items() if c0 == metric}
        if estimators is not None:
            f_dict = {
                (c0, c1): ls for ((c0, c1), ls) in f_dict.items() if c1 in estimators
            }
        if (metric, estimators) != (None, None):
            f_dict = {c1: ls for ((c0, c1), ls) in f_dict.items()}

        return f_dict

    def group_by_shift(self, metric: str = None, estimators: List[str] = None):
        f_dict = self.data(metric=metric, estimators=estimators)
        shift_prevs = np.around(
            np.absolute(self.prevs - self.train_prev[1]), decimals=2
        )
        shift_dict = {col: {sp: [] for sp in shift_prevs} for col in f_dict.keys()}
        for col, vals in f_dict.items():
            for sp, bp in zip(shift_prevs, self.prevs):
                shift_dict[col][sp] = np.concatenate(
                    [shift_dict[col][sp], f_dict[col][bp]]
                )

        return np.sort(np.unique(shift_prevs)), shift_dict

    def avg_by_prevs(self, metric: str = None, estimators: List[str] = None):
        f_dict = self.data(metric=metric, estimators=estimators)
        return {
            col: np.array([np.mean(vals[bp]) for bp in self.prevs])
            for col, vals in f_dict.items()
        }

    def stdev_by_prevs(self, metric: str = None, estimators: List[str] = None):
        f_dict = self.data(metric=metric, estimators=estimators)
        return {
            col: np.array([np.std(vals[bp]) for bp in self.prevs])
            for col, vals in f_dict.items()
        }

    def avg_all(self, metric: str = None, estimators: List[str] = None):
        f_dict = self.data(metric=metric, estimators=estimators)
        return {
            col: [np.mean(np.concatenate(list(vals.values())))]
            for col, vals in f_dict.items()
        }

    def get_dataframe(self, metric="acc", estimators=None):
        avg_dict = self.avg_by_prevs(metric=metric, estimators=estimators)
        all_dict = self.avg_all(metric=metric, estimators=estimators)
        for col in avg_dict.keys():
            avg_dict[col] = np.append(avg_dict[col], all_dict[col])
        return pd.DataFrame(
            avg_dict,
            index=self.prevs.tolist() + ["tot"],
            columns=avg_dict.keys(),
        )

    def get_plots(
        self,
        modes=["delta", "diagonal", "shift"],
        metric="acc",
        estimators=None,
        conf="default",
        stdev=False,
    ) -> Path:
        pps = []
        for mode in modes:
            pp = []
            if mode == "delta":
                f_dict = self.avg_by_prevs(metric=metric, estimators=estimators)
                _pp0 = plot.plot_delta(
                    self.cprevs,
                    f_dict,
                    metric=metric,
                    name=conf,
                    train_prev=self.train_prev,
                    fit_scores=self.fit_scores,
                )
                pp = [(mode, _pp0)]
                if stdev:
                    fs_dict = self.stdev_by_prevs(metric=metric, estimators=estimators)
                    _pp1 = plot.plot_delta(
                        self.cprevs,
                        f_dict,
                        metric=metric,
                        name=conf,
                        train_prev=self.train_prev,
                        fit_scores=self.fit_scores,
                        stdevs=fs_dict,
                    )
                    pp.append((f"{mode}_stdev", _pp1))
            elif mode == "diagonal":
                f_dict = {
                    col: np.concatenate([vals[bp] for bp in self.prevs])
                    for col, vals in self.data(
                        metric=metric + "_score", estimators=estimators
                    ).items()
                }
                reference = f_dict["ref"]
                f_dict = {k: v for k, v in f_dict.items() if k != "ref"}
                _pp0 = plot.plot_diagonal(
                    reference,
                    f_dict,
                    metric=metric,
                    name=conf,
                    train_prev=self.train_prev,
                )
                pp = [(mode, _pp0)]

            elif mode == "shift":
                s_prevs, s_dict = self.group_by_shift(
                    metric=metric, estimators=estimators
                )
                _pp0 = plot.plot_shift(
                    np.around([(1.0 - p, p) for p in s_prevs], decimals=2),
                    {
                        col: np.array([np.mean(vals[sp]) for sp in s_prevs])
                        for col, vals in s_dict.items()
                    },
                    metric=metric,
                    name=conf,
                    train_prev=self.train_prev,
                    fit_scores=self.fit_scores,
                )
                pp = [(mode, _pp0)]

            pps.extend(pp)

        return pps

    def to_md(self, conf="default", metric="acc", estimators=None, stdev=False):
        res = f"## {int(np.around(self.train_prev, decimals=2)[1]*100)}% positives\n"
        res += fmt_line_md(f"train: {str(self.train_prev)}")
        res += fmt_line_md(f"validation: {str(self.valid_prev)}")
        for k, v in self.times.items():
            res += fmt_line_md(f"{k}: {v:.3f}s")
        res += "\n"
        res += (
            self.get_dataframe(metric=metric, estimators=estimators).to_html() + "\n\n"
        )
        plot_modes = ["delta", "diagonal", "shift"]
        for mode, op in self.get_plots(
            modes=plot_modes,
            metric=metric,
            estimators=estimators,
            conf=conf,
            stdev=stdev,
        ):
            res += f"![plot_{mode}]({op.relative_to(env.OUT_DIR).as_posix()})\n"

        return res


class DatasetReport:
    def __init__(self, name):
        self.name = name
        self._dict = None
        self.crs: List[CompReport] = []

    @property
    def cprevs(self):
        return np.around([(1.0 - p, p) for p in self.prevs], decimals=2)

    def add(self, cr: CompReport):
        if cr is None:
            return

        self.crs.append(cr)

        if self._dict is None:
            self.prevs = cr.prevs
            self._dict = {
                col: {bp: vals[bp] for bp in self.prevs}
                for col, vals in cr.data().items()
            }
            self.s_prevs, self.s_dict = cr.group_by_shift()
            self.fit_scores = {k: [score] for k, score in cr.fit_scores.items()}
            return

        cr_dict = cr.data()
        both_prevs = np.array([self.prevs, cr.prevs])
        if not np.all(both_prevs == both_prevs[0, :]).all():
            raise ValueError("Comp report has incompatible base prevalences")

        for col, vals in cr_dict.items():
            if col not in self._dict:
                self._dict[col] = {}
            for bp in self.prevs:
                if bp not in self._dict[col]:
                    self._dict[col][bp] = []
                self._dict[col][bp] = np.concatenate(
                    [self._dict[col][bp], cr_dict[col][bp]]
                )

        cr_s_prevs, cr_s_dict = cr.group_by_shift()
        self.s_prevs = np.sort(np.unique(np.concatenate([self.s_prevs, cr_s_prevs])))

        for col, vals in cr_s_dict.items():
            if col not in self.s_dict:
                self.s_dict[col] = {}
            for sp in cr_s_prevs:
                if sp not in self.s_dict[col]:
                    self.s_dict[col][sp] = []
                self.s_dict[col][sp] = np.concatenate(
                    [self.s_dict[col][sp], cr_s_dict[col][sp]]
                )

        for sp in self.s_prevs:
            for col, vals in self.s_dict.items():
                if sp not in vals:
                    vals[sp] = []

        for k, score in cr.fit_scores.items():
            if k not in self.fit_scores:
                self.fit_scores[k] = []
            self.fit_scores[k].append(score)

    def __add__(self, cr: CompReport):
        self.add(cr)
        return self

    def __iadd__(self, cr: CompReport):
        self.add(cr)
        return self

    def to_md(self, conf="default", metric="acc", estimators=[], stdev=False):
        res = f"# {self.name}\n\n"
        for cr in self.crs:
            res += f"{cr.to_md(conf, metric=metric, estimators=estimators, stdev=stdev)}\n\n"

        f_dict = {
            c1: v
            for ((c0, c1), v) in self._dict.items()
            if c0 == metric and c1 in estimators
        }
        s_avg_dict = {
            col: np.array([np.mean(vals[sp]) for sp in self.s_prevs])
            for col, vals in {
                c1: v
                for ((c0, c1), v) in self.s_dict.items()
                if c0 == metric and c1 in estimators
            }.items()
        }
        avg_dict = {
            col: np.array([np.mean(vals[bp]) for bp in self.prevs])
            for col, vals in f_dict.items()
        }
        if stdev:
            stdev_dict = {
                col: np.array([np.std(vals[bp]) for bp in self.prevs])
                for col, vals in f_dict.items()
            }
        all_dict = {
            col: [np.mean(np.concatenate(list(vals.values())))]
            for col, vals in f_dict.items()
        }
        df = pd.DataFrame(
            {col: np.append(avg_dict[col], val) for col, val in all_dict.items()},
            index=self.prevs.tolist() + ["tot"],
            columns=all_dict.keys(),
        )

        res += "## avg\n"
        res += df.to_html() + "\n\n"

        delta_op = plot.plot_delta(
            np.around([(1.0 - p, p) for p in self.prevs], decimals=2),
            avg_dict,
            metric=metric,
            name=conf,
            train_prev=None,
            fit_scores={k: np.mean(vals) for k, vals in self.fit_scores.items()},
        )
        res += f"![plot_delta]({delta_op.relative_to(env.OUT_DIR).as_posix()})\n"

        if stdev:
            delta_stdev_op = plot.plot_delta(
                np.around([(1.0 - p, p) for p in self.prevs], decimals=2),
                avg_dict,
                metric=metric,
                name=conf,
                train_prev=None,
                fit_scores={k: np.mean(vals) for k, vals in self.fit_scores.items()},
                stdevs=stdev_dict,
            )
            res += f"![plot_delta_stdev]({delta_stdev_op.relative_to(env.OUT_DIR).as_posix()})\n"

        shift_op = plot.plot_shift(
            np.around([(1.0 - p, p) for p in self.s_prevs], decimals=2),
            s_avg_dict,
            metric=metric,
            name=conf,
            train_prev=None,
            fit_scores={k: np.mean(vals) for k, vals in self.fit_scores.items()},
        )
        res += f"![plot_shift]({shift_op.relative_to(env.OUT_DIR).as_posix()})\n"

        return res

    def __iter__(self):
        return (cr for cr in self.crs)
