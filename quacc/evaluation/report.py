from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from quacc import plot
from quacc.environ import env
from quacc.utils import fmt_line_md


class EvaluationReport:
    def __init__(self, name=None):
        self._prevs = []
        self._dict = {}
        self._g_prevs = None
        self._g_dict = None
        self.name = name if name is not None else "default"
        self.times = {}
        self.train_prev = None
        self.valid_prev = None
        self.target = "default"

    def append_row(self, base: np.ndarray | Tuple, **row):
        if isinstance(base, np.ndarray):
            base = tuple(base.tolist())
        self._prevs.append(base)
        for k, v in row.items():
            if (k, self.name) in self._dict:
                self._dict[(k, self.name)].append(v)
            else:
                self._dict[(k, self.name)] = [v]
        self._g_prevs = None

    @property
    def columns(self):
        return self._dict.keys()

    def group_by_prevs(self, metric: str = None):
        if self._g_dict is None:
            self._g_prevs = []
            self._g_dict = {k: [] for k in self._dict.keys()}

            for col, vals in self._dict.items():
                col_grouped = {}
                for bp, v in zip(self._prevs, vals):
                    if bp not in col_grouped:
                        col_grouped[bp] = []
                    col_grouped[bp].append(v)

                self._g_dict[col] = [
                    vs
                    for bp, vs in sorted(col_grouped.items(), key=lambda cg: cg[0][1])
                ]

            self._g_prevs = sorted(
                [(p0, p1) for [p0, p1] in np.unique(self._prevs, axis=0).tolist()],
                key=lambda bp: bp[1],
            )

            # last_end = 0
            # for ind, bp in enumerate(self._prevs):
            #     if ind < (len(self._prevs) - 1) and bp == self._prevs[ind + 1]:
            #         continue

            #     self._g_prevs.append(bp)
            #     for col in self._dict.keys():
            #         self._g_dict[col].append(
            #             stats.mean(self._dict[col][last_end : ind + 1])
            #         )

            #     last_end = ind + 1

        filtered_g_dict = self._g_dict
        if metric is not None:
            filtered_g_dict = {
                c1: ls for ((c0, c1), ls) in self._g_dict.items() if c0 == metric
            }

        return self._g_prevs, filtered_g_dict

    def avg_by_prevs(self, metric: str = None):
        g_prevs, g_dict = self.group_by_prevs(metric=metric)

        a_dict = {}
        for col, vals in g_dict.items():
            a_dict[col] = [np.mean(vs) for vs in vals]

        return g_prevs, a_dict

    def avg_all(self, metric: str = None):
        f_dict = self._dict
        if metric is not None:
            f_dict = {c1: ls for ((c0, c1), ls) in self._dict.items() if c0 == metric}

        a_dict = {}
        for col, vals in f_dict.items():
            a_dict[col] = [np.mean(vals)]

        return a_dict

    def get_dataframe(self, metric="acc"):
        g_prevs, g_dict = self.avg_by_prevs(metric=metric)
        a_dict = self.avg_all(metric=metric)
        for col in g_dict.keys():
            g_dict[col].extend(a_dict[col])
        return pd.DataFrame(
            g_dict,
            index=g_prevs + ["tot"],
            columns=g_dict.keys(),
        )

    def get_plot(self, mode="delta", metric="acc") -> Path:
        if mode == "delta":
            g_prevs, g_dict = self.group_by_prevs(metric=metric)
            return plot.plot_delta(
                g_prevs,
                g_dict,
                metric=metric,
                name=self.name,
                train_prev=self.train_prev,
            )
        elif mode == "diagonal":
            _, g_dict = self.avg_by_prevs(metric=metric + "_score")
            f_dict = {k: v for k, v in g_dict.items() if k != "ref"}
            referece = g_dict["ref"]
            return plot.plot_diagonal(
                referece,
                f_dict,
                metric=metric,
                name=self.name,
                train_prev=self.train_prev,
            )
        elif mode == "shift":
            g_prevs, g_dict = self.avg_by_prevs(metric=metric)
            return plot.plot_shift(
                g_prevs,
                g_dict,
                metric=metric,
                name=self.name,
                train_prev=self.train_prev,
            )

    def to_md(self, *metrics):
        res = ""
        res += fmt_line_md(f"train: {str(self.train_prev)}")
        res += fmt_line_md(f"validation: {str(self.valid_prev)}")
        for k, v in self.times.items():
            res += fmt_line_md(f"{k}: {v:.3f}s")
        res += "\n"
        for m in metrics:
            res += self.get_dataframe(metric=m).to_html() + "\n\n"
            op_delta = self.get_plot(mode="delta", metric=m)
            res += f"![plot_delta]({str(op_delta.relative_to(env.OUT_DIR))})\n"
            op_diag = self.get_plot(mode="diagonal", metric=m)
            res += f"![plot_diagonal]({str(op_diag.relative_to(env.OUT_DIR))})\n"
            op_shift = self.get_plot(mode="shift", metric=m)
            res += f"![plot_shift]({str(op_shift.relative_to(env.OUT_DIR))})\n"

        return res

    def merge(self, other):
        if not all(v1 == v2 for v1, v2 in zip(self._prevs, other._prevs)):
            raise ValueError("other has not same base prevalences of self")

        inters_keys = set(self._dict.keys()).intersection(set(other._dict.keys()))
        if len(inters_keys) > 0:
            raise ValueError(f"self and other have matching keys {str(inters_keys)}.")

        report = EvaluationReport()
        report._prevs = self._prevs
        report._dict = self._dict | other._dict
        return report

    @staticmethod
    def combine_reports(*args, name="default", train_prev=None, valid_prev=None):
        er = args[0]
        for r in args[1:]:
            er = er.merge(r)

        er.name = name
        er.train_prev = train_prev
        er.valid_prev = valid_prev
        return er


class DatasetReport:
    def __init__(self, name):
        self.name = name
        self.ers: List[EvaluationReport] = []

    def add(self, er: EvaluationReport):
        self.ers.append(er)

    def to_md(self, *metrics):
        res = f"{self.name}\n\n"
        for er in self.ers:
            res += f"{er.to_md(*metrics)}\n\n"

        return res

    def __iter__(self):
        return (er for er in self.ers)
