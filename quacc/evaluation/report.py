from typing import Tuple
import statistics as stats
import numpy as np
import pandas as pd


def _fmt_line(s):
    return f"> {s}  \n"


class EvaluationReport:
    def __init__(self, prefix=None):
        self.base = []
        self.dict = {}
        self._grouped = False
        self._grouped_base = []
        self._grouped_dict = {}
        self._dataframe = None
        self.prefix = prefix if prefix is not None else "default"
        self._times = {}
        self._prevs = {}
        self._target = "default"

    def append_row(self, base: np.ndarray | Tuple, **row):
        if isinstance(base, np.ndarray):
            base = tuple(base.tolist())
        self.base.append(base)
        for k, v in row.items():
            if (k, self.prefix) in self.dict:
                self.dict[(k, self.prefix)].append(v)
            else:
                self.dict[(k, self.prefix)] = [v]
        self._grouped = False
        self._dataframe = None

    @property
    def columns(self):
        return self.dict.keys()

    @property
    def grouped(self):
        if self._grouped:
            return self._grouped_dict

        self._grouped_base = []
        self._grouped_dict = {k: [] for k in self.dict.keys()}

        last_end = 0
        for ind, bp in enumerate(self.base):
            if ind < (len(self.base) - 1) and bp == self.base[ind + 1]:
                continue

            self._grouped_base.append(bp)
            for col in self.dict.keys():
                self._grouped_dict[col].append(
                    stats.mean(self.dict[col][last_end : ind + 1])
                )

            last_end = ind + 1

        self._grouped = True
        return self._grouped_dict

    @property
    def gbase(self):
        self.grouped
        return self._grouped_base

    def get_dataframe(self, metrics=None):
        if self._dataframe is None:
            self_columns = sorted(self.columns, key=lambda c: c[0])
            self._dataframe = pd.DataFrame(
                self.grouped,
                index=self.gbase,
                columns=pd.MultiIndex.from_tuples(self_columns),
            )

        df = pd.DataFrame(self._dataframe)
        if metrics is not None:
            df = df.drop(
                [(c0, c1) for (c0, c1) in df.columns if c0 not in metrics], axis=1
            )

        if len(set(k0 for k0, k1 in df.columns)) == 1:
            df = df.droplevel(0, axis=1)

        return df

    def merge(self, other):
        if not all(v1 == v2 for v1, v2 in zip(self.base, other.base)):
            raise ValueError("other has not same base prevalences of self")

        if len(set(self.dict.keys()).intersection(set(other.dict.keys()))) > 0:
            raise ValueError("self and other have matching keys")

        report = EvaluationReport()
        report.base = self.base
        report.dict = self.dict | other.dict
        return report

    @property
    def times(self):
        return self._times

    @times.setter
    def times(self, val):
        self._times = val

    @property
    def prevs(self):
        return self._prevs

    @prevs.setter
    def prevs(self, val):
        self._prevs = val

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, val):
        self._target = val

    def to_md(self, *metrics):
        res = _fmt_line("target: " + self.target)
        for k, v in self.prevs.items():
            res += _fmt_line(f"{k}: {str(v)}")
        for k, v in self.times.items():
            res += _fmt_line(f"{k}: {v:.3f}s")
        res += "\n"
        for m in metrics:
            res += self.get_dataframe(metrics=m).to_html() + "\n\n"

        return res

    @staticmethod
    def combine_reports(*args):
        er = args[0]
        for r in args[1:]:
            er = er.merge(r)

        return er
