import functools
import json
import os
from abc import ABC
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Callable, Literal
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import quapy as qp
from joblib import Parallel, delayed
from quapy.data.base import LabelledCollection
from scipy.sparse import coo_array
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import cap


class NaNError(Exception):
    def __init__(self, dataset, method, *args: object) -> None:
        super().__init__(*args)
        self.dataset = dataset
        self.method = method


class SparseMatrixBuilder(ABC):
    def __init__(self):
        self.data = []
        self.rows = []
        self.cols = []

    def add(self, data: np.ndarray, rows: np.ndarray, cols: np.ndarray):
        assert len(data) == len(rows) == len(cols), "Inconsistent array lengths; cannot add to SparseMatrixBuilder"
        self.data.append(data)
        self.rows.append(rows)
        self.cols.append(cols)

    def build_csr(self, shape):
        data = np.hstack(self.data)
        rows = np.hstack(self.rows)
        cols = np.hstack(self.cols)
        coo = coo_array((data, (rows, cols)), shape=shape)
        return coo.tocsr()


def combine_dataframes(dfs, df_index=[]) -> pd.DataFrame:
    if len(dfs) < 1:
        raise ValueError
    if len(dfs) == 1:
        return dfs[0]
    df = dfs[0]
    for ndf in dfs[1:]:
        df = df.join(ndf.set_index(df_index), on=df_index)

    return df


def avg_group_report(df: pd.DataFrame) -> pd.DataFrame:
    def _reduce_func(s1, s2):
        return {(n1, n2): v + s2[(n1, n2)] for ((n1, n2), v) in s1.items()}

    lst = df.to_dict(orient="records")[1:-1]
    summed_series = functools.reduce(_reduce_func, lst)
    idx = df.columns.drop([("base", "T"), ("base", "F")])
    avg_report = {(n1, n2): (v / len(lst)) for ((n1, n2), v) in summed_series.items() if n1 != "base"}
    return pd.DataFrame([avg_report], columns=idx)


def fmt_line_md(s):
    return f"> {s}  \n"


def get_quacc_home():
    home = cap.env["QUACC_DATA"]
    os.makedirs(home, exist_ok=True)
    return home


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, downloaded_path: Path):
    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=downloaded_path.name,
    ) as t:
        urlretrieve(url, filename=downloaded_path, reporthook=t.update_to)


def save_json_file(path, data):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_json_file(path, object_hook=None):
    if not os.path.exists(path):
        raise ValueError("Ivalid path for json file")
    with open(path, "r") as f:
        return json.load(f, object_hook=object_hook)


def get_results_path(rootdir, basedir, cls_name, acc_name, dataset_name, method_name):
    return os.path.join(
        rootdir,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        method_name + ".json",
    )


def get_plots_path(basedir, problem, cls_name, acc_name, dataset_name, plot_type, ext="svg"):
    return os.path.join(
        cap.env["OUT_DIR"],
        basedir,
        "plots",
        problem,
        f"{cls_name}_{acc_name}_{dataset_name}_{plot_type}.{ext}",
    )


def get_njobs(n_jobs):
    return cap.env["N_JOBS"] if n_jobs is None else n_jobs


def true_acc(h: BaseEstimator, acc_fn: Callable, U: LabelledCollection):
    y_pred = h.predict(U.X)
    y_true = U.y
    conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=U.classes_)
    return acc_fn(conf_table)


def save_dataset_stats(dataset_name, test_prot, L, V):
    path = os.path.join(cap.env["OUT_DIR"], "dataset_stats", f"{dataset_name}.json")
    test_prevs = [Ui.prevalence() for Ui in test_prot()]
    shifts = [qp.error.ae(L.prevalence(), Ui_prev) for Ui_prev in test_prevs]
    info = {
        "n_classes": L.n_classes,
        "n_train": len(L),
        "n_val": len(V),
        "train_prev": L.prevalence().tolist(),
        "val_prev": V.prevalence().tolist(),
        "test_prevs": [x.tolist() for x in test_prevs],
        "shifts": [x.tolist() for x in shifts],
        "sample_size": test_prot.sample_size,
        "num_samples": test_prot.total(),
    }
    save_json_file(path, info)


@contextmanager
def temp_force_njobs(force):
    if force:
        openblas_nt_was_set = "OPENBLAS_NUM_THREADS" in os.environ
        if openblas_nt_was_set:
            openblas_nt_old = os.getenv("OPENBLAS_NUM_THREADS")

        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        if force:
            if openblas_nt_was_set:
                os.environ["OPENBLAS_NUM_THREADS"] = openblas_nt_old
            else:
                os.environ.pop("OPENBLAS_NUM_THREADS")


def parallel(
    func,
    args_list,
    n_jobs,
    seed=None,
    return_as: Literal["list"] | Literal["array"] | Literal["generator"] | Literal["generator_unordered"] = "list",
    backend="loky",
    verbose=0,
    batch_size="auto",
    max_nbytes="1M",
):
    """
    A wrapper of multiprocessing:

    >>> Parallel(n_jobs=n_jobs)(
    >>>      delayed(func)(args_i) for args_i in args
    >>> )

    that takes the `quapy.environ` variable as input silently.
    Seeds the child processes to ensure reproducibility when n_jobs>1.

    :param func: callable
    :param args: args of func
    :param seed: the numeric seed
    :param asarray: set to True to return a np.ndarray instead of a list
    :param backend: indicates the backend used for handling parallel works
    """

    def func_dec(qp_environ, cap_environ, seed, *args):
        qp.environ = qp_environ.copy()
        qp.environ["N_JOBS"] = 1
        cap.env = cap_environ.copy()
        cap.env["N_JOBS"] = 1
        # set a context with a temporal seed to ensure results are reproducibles in parallel
        with ExitStack() as stack:
            if seed is not None:
                stack.enter_context(qp.util.temp_seed(seed))
            return func(*args)

    _returnas = "list" if return_as == "array" else return_as
    with ExitStack() as stack:
        stack.enter_context(cap.commons.temp_force_njobs(cap.env["FORCE_NJOBS"]))
        out = Parallel(
            n_jobs=n_jobs,
            return_as=_returnas,
            backend=backend,
            verbose=verbose,
            batch_size=batch_size,
            max_nbytes=max_nbytes,
        )(
            delayed(func_dec)(qp.environ, cap.env, None if seed is None else seed + i, args_i)
            for i, args_i in enumerate(args_list)
        )

    if return_as == "array":
        out = np.asarray(out)
    return out


def get_shift(test_prevs: np.ndarray, train_prev: np.ndarray | float, decimals=2):
    """
    Computes the shift of an array of prevalence values for a set of test sample in
    relation to the prevalence value of the training set.

    :param test_prevs: prevalence values for the test samples
    :param train_prev: prevalence value for the training set
    :param decimals: rounding decimals for the result (default=2)
    :return: an ndarray with the shifts for each test sample, shaped as (n,1) (ndim=2)
    """
    if test_prevs.ndim == 1:
        test_prevs = test_prevs[:, np.newaxis]
    train_prevs = np.tile(train_prev, (test_prevs.shape[0], 1))
    # _shift = nae(test_prevs, train_prevs)
    _shift = qp.error.ae(test_prevs, train_prevs)
    return np.around(_shift, decimals=decimals)


def contingency_table(y, y_hat, n_classes):
    ct = np.zeros((n_classes, n_classes))
    for _c in range(n_classes):
        _idx = y == _c
        for _c1 in range(n_classes):
            ct[_c, _c1] = np.sum(y_hat[_idx] == _c1)

    return ct / y.shape[0]
