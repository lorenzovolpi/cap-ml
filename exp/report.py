import itertools
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import zscore

import quacc as qc
from quacc.utils.commons import get_results_path, get_shift, load_json_file, save_json_file


class TestReport:
    def __init__(
        self,
        rootdir,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        train_prev,
        val_prev,
        method_name,
    ):
        self.rootdir = rootdir
        self.basedir = basedir
        self.cls_name = cls_name
        self.acc_name = acc_name
        self.dataset_name = dataset_name
        self.train_prev = train_prev
        self.val_prev = val_prev
        self.method_name = method_name

    def get_path(self):
        return get_results_path(
            self.rootdir, self.basedir, self.cls_name, self.acc_name, self.dataset_name, self.method_name
        )

    def exists(self):
        return os.path.exists(self.get_path())

    @classmethod
    def build_path(cls, rootdir, basedir, cls_name, acc_name, dataset_name, method_name):
        return get_results_path(rootdir, basedir, cls_name, acc_name, dataset_name, method_name)

    def add_result(self, test_prevs, true_accs, estim_accs, t_train, t_test_ave):
        self.test_prevs = test_prevs
        self.estim_accs = estim_accs
        self.true_accs = true_accs
        self.t_train = t_train
        self.t_test_ave = t_test_ave

        return self

    def save_json(self):
        result = {
            "basedir": self.basedir,
            "cls_name": self.cls_name,
            "acc_name": self.acc_name,
            "dataset_name": self.dataset_name,
            "train_prev": self.train_prev,
            "val_prev": self.val_prev,
            "method_name": self.method_name,
            "test_prevs": self.test_prevs,
            "true_accs": self.true_accs,
            "estim_accs": self.estim_accs,
            "t_train": self.t_train,
            "t_test_ave": self.t_test_ave,
        }

        save_json_file(self.get_path(), result)

    @classmethod
    def load_json(cls, path, rootdir) -> "TestReport":
        def _test_report_hook(_dict):
            return TestReport(
                rootdir=rootdir,
                basedir=_dict["basedir"],
                cls_name=_dict["cls_name"],
                acc_name=_dict["acc_name"],
                dataset_name=_dict["dataset_name"],
                train_prev=_dict["train_prev"],
                val_prev=_dict["val_prev"],
                method_name=_dict["method_name"],
            ).add_result(
                test_prevs=_dict["test_prevs"],
                true_accs=_dict["true_accs"],
                estim_accs=_dict["estim_accs"],
                t_train=_dict["t_train"],
                t_test_ave=_dict["t_test_ave"],
            )

        return load_json_file(path, object_hook=_test_report_hook)


class Report:
    def __init__(self, results: dict[str, list[TestReport]]):
        self.results = results

    @classmethod
    def load_results(
        cls, rootdir, basedir, cls_name, acc_name, datasets: str | list[str] = "*", methods: str | list[str] = "*"
    ) -> "Report":
        _results = defaultdict(lambda: [])
        if isinstance(methods, str):
            methods = [methods]
        if isinstance(datasets, str):
            datasets = [datasets]
        for dataset_, method_ in itertools.product(datasets, methods):
            path = get_results_path(rootdir, basedir, cls_name, acc_name, dataset_, method_)
            for file in glob(path):
                if file.endswith(".json"):
                    method = Path(file).stem
                    _res = TestReport.load_json(file, rootdir)
                    _results[method].append(_res)
        return Report(_results)

    def filter_by_method(self, methods=None):
        if methods is None:
            return Report(self.results)

        if isinstance(methods, str):
            methods = [methods]
        return Report({_m: _rs for _m, _rs in self.results.items() if _m in methods})

    def time_data(self):
        dfs = []
        for _method, _results in self.results.items():
            _dataset_map = defaultdict(lambda: [])
            for r in _results:
                _dataset_map[r.dataset_name].append(r)

            for _dataset, _res in _dataset_map.items():
                _t_train = np.array([_r.t_train for _r in _res])
                _t_test_ave = np.array([_r.t_test_ave for _r in _res])
                report_df = pd.DataFrame(np.vstack([_t_train, _t_test_ave]).T, columns=["t_train", "t_test_ave"])
                report_df["method"] = _method
                report_df["dataset"] = _dataset
                dfs.append(report_df)

        all_df = pd.concat(dfs, axis=0, ignore_index=True)
        return all_df

    def table_data(self, mean=True, error=qc.error.ae):
        assert error in qc.error.ACCURACY_ERROR_SINGLE, "Unknown error function"
        dfs = []
        for _method, _results in self.results.items():
            _dataset_map = defaultdict(lambda: [])
            for r in _results:
                _dataset_map[r.dataset_name].append(r)

            for _dataset, _res in _dataset_map.items():
                _true_acc = np.hstack([_r.true_accs for _r in _res])
                _estim_acc = np.hstack([_r.estim_accs for _r in _res])
                _acc_err = error(_true_acc, _estim_acc)
                report_df = pd.DataFrame(
                    np.vstack([_true_acc, _estim_acc, _acc_err]).T,
                    columns=["true_accs", "estim_accs", "acc_err"],
                )
                report_df.loc[:, "method"] = np.tile(_method, (len(report_df),))
                report_df.loc[:, "dataset"] = np.tile(_dataset, (len(report_df),))
                dfs.append(report_df)

        all_df = pd.concat(dfs, axis=0, ignore_index=True)

        # compute zscore by dataset
        for _dataset in all_df["dataset"].unique():
            _acc_err = all_df.loc[all_df["dataset"] == _dataset, "acc_err"]
            all_df.loc[all_df["dataset"] == _dataset, "z_score"] = zscore(_acc_err)

        if mean:
            all_df = all_df.groupby(["method", "dataset"]).mean().reset_index()
        return all_df

    def diagonal_plot_data(self):
        dfs = []
        for _method, _results in self.results.items():
            _true_acc = np.hstack([_r.true_accs for _r in _results])
            _estim_acc = np.hstack([_r.estim_accs for _r in _results])
            _prev = np.hstack([_r.test_prevs for _r in _results])
            method_df = pd.DataFrame(
                np.vstack([_true_acc, _estim_acc, _prev]).T, columns=["true_accs", "estim_accs", "prev"]
            )
            method_df.loc[:, "method"] = np.tile(_method, (len(method_df),))
            dfs.append(method_df)

        return pd.concat(dfs, axis=0, ignore_index=True)

    # def delta_plot_data(self):
    #     dfs = []
    #     for _method, _results in self.results.items():
    #         if isinstance(_results[0].test_prevs[0], float):
    #             _prev = np.hstack([_r.test_prevs for _r in _results])
    #         else:
    #             _prev = np.vstack([_r.test_prevs for _r in _results])
    #             _prev = np.fromiter((tuple(tp) for tp in _prev), dtype="object")

    #         _true_accs = np.hstack([_r.true_accs for _r in _results])
    #         _estim_accs = np.hstack([_r.estim_accs for _r in _results])
    #         _acc_err = np.abs(_true_accs - _estim_accs)
    #         method_df = pd.DataFrame(
    #             np.vstack([_prev, _acc_err]).T, columns=["prevs", "acc_err"]
    #         )
    #         method_df.loc[:, "method"] = np.tile(_method, (len(method_df),))
    #         dfs.append(method_df.sort_values(by="prevs"))

    #     return pd.concat(dfs, axis=0, ignore_index=True)

    def shift_plot_data(self, error=qc.error.ae):
        assert error in qc.error.ACCURACY_ERROR_SINGLE, "Unknown error function"

        dfs = []
        for _method, _results in self.results.items():
            _shift = np.hstack([get_shift(np.array(_r.test_prevs), _r.train_prev) for _r in _results])

            _true_accs = np.hstack([_r.true_accs for _r in _results])
            _estim_accs = np.hstack([_r.estim_accs for _r in _results])
            _acc_err = error(_true_accs, _estim_accs)
            method_df = pd.DataFrame(np.vstack([_shift, _acc_err]).T, columns=["shifts", "acc_err"])
            method_df.loc[:, "method"] = np.tile(_method, (len(method_df),))
            dfs.append(method_df.sort_values(by="shifts"))

        return pd.concat(dfs, axis=0, ignore_index=True)
