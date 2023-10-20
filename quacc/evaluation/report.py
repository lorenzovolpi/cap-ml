import math
import statistics as stats
from typing import List, Tuple

import numpy as np
import pandas as pd

from quacc import plot
from quacc.utils import fmt_line_md


class EvaluationReport:
    def __init__(self, prefix=None):
        self._prevs = []
        self._dict = {}
        self._g_prevs = None
        self._g_dict = None
        self.name = prefix if prefix is not None else "default"
        self.times = {}
        self.train_prevs = {}
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

    def groupby_prevs(self, metric: str = None):
        if self._g_dict is None:
            self._g_prevs = []
            self._g_dict = {k: [] for k in self._dict.keys()}

            last_end = 0
            for ind, bp in enumerate(self._prevs):
                if ind < (len(self._prevs) - 1) and bp == self._prevs[ind + 1]:
                    continue

                self._g_prevs.append(bp)
                for col in self._dict.keys():
                    self._g_dict[col].append(
                        stats.mean(self._dict[col][last_end : ind + 1])
                    )

                last_end = ind + 1

        filtered_g_dict = self._g_dict
        if metric is not None:
            filtered_g_dict = {
                c1: ls for ((c0, c1), ls) in self._g_dict.items() if c0 == metric
            }

        return self._g_prevs, filtered_g_dict

    def get_dataframe(self, metric="acc"):
        g_prevs, g_dict = self.groupby_prevs(metric=metric)
        return pd.DataFrame(
            g_dict,
            index=g_prevs,
            columns=g_dict.keys(),
        )

    def get_plot(self, mode="delta", metric="acc"):
        g_prevs, g_dict = self.groupby_prevs(metric=metric)
        t_prev = int(round(self.train_prevs["train"][0] * 100))
        title = f"{self.name}_{t_prev}_{metric}"
        plot.plot_delta(g_prevs, g_dict, metric, title)

    def to_md(self, *metrics):
        res = ""
        for k, v in self.train_prevs.items():
            res += fmt_line_md(f"{k}: {str(v)}")
        for k, v in self.times.items():
            res += fmt_line_md(f"{k}: {v:.3f}s")
        res += "\n"
        for m in metrics:
            res += self.get_dataframe(metric=m).to_html() + "\n\n"
            self.get_plot(metric=m)

        return res

    def merge(self, other):
        if not all(v1 == v2 for v1, v2 in zip(self._prevs, other._prevs)):
            raise ValueError("other has not same base prevalences of self")

        if len(set(self._dict.keys()).intersection(set(other._dict.keys()))) > 0:
            raise ValueError("self and other have matching keys")

        report = EvaluationReport()
        report._prevs = self._prevs
        report._dict = self._dict | other._dict
        return report

    @staticmethod
    def combine_reports(*args, name="default"):
        er = args[0]
        for r in args[1:]:
            er = er.merge(r)

        er.name = name
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
