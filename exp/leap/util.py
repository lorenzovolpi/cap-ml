import glob
import itertools as IT
import os
from pathlib import Path

import pandas as pd

import exp.leap.env as env
from exp.leap.config import gen_acc_measure, get_method_names, is_excluded
from quacc.models.cont_table import LEAP


def local_path(dataset_name, cls_name, method_name, acc_name, subproject=None):
    if subproject is None:
        parent_dir = os.path.join(env.root_dir, env.PROBLEM, cls_name, acc_name, dataset_name)
    else:
        parent_dir = os.path.join(env.root_dir, subproject, env.PROBLEM, cls_name, acc_name, dataset_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.json")


def all_exist_pre_check(dataset_name, cls_name, subproject=None, method_names=None):
    method_names = get_method_names() if method_names is None else method_names
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        if is_excluded(cls_name, dataset_name, method, acc):
            continue
        path = local_path(dataset_name, cls_name, method, acc, subproject=subproject)
        all_exist = os.path.exists(path)
        if not all_exist:
            print(path)
            break

    return all_exist


def load_results(*, base_dir=None, classifier="*", acc="*", dataset="*", filter_methods=None) -> pd.DataFrame:
    base_dir = env.root_dir if base_dir is None else base_dir
    dfs = []
    _methods = get_method_names() if filter_methods is None else filter_methods
    for path in glob.glob(os.path.join(base_dir, env.PROBLEM, classifier, acc, dataset, "*.json"), recursive=True):
        if Path(path).stem in _methods:
            dfs.append(pd.read_json(path))

    return pd.concat(dfs, axis=0)


def rename_datasets(mapping, datasets: str | list[str], df=None):
    if isinstance(datasets, str):
        _datasets = [mapping.get(d, d) for d in [datasets]]
    else:
        _datasets = [mapping.get(d, d) for d in datasets]

    if df is not None:
        for d, rd in mapping.items():
            df.loc[df["dataset"] == d, "dataset"] = rd

    res_datasets = _datasets[0] if isinstance(datasets, str) else _datasets
    if df is None:
        return res_datasets
    else:
        return res_datasets, df


def decorate_datasets(datasets: str | list[str], df=None):
    def _decorate(d):
        return r"\textsf{" + d + r"}"

    if isinstance(datasets, str):
        _datasets = [_decorate(d) for d in [datasets]]
    else:
        _datasets = [_decorate(d) for d in datasets]

    if df is not None:
        for d in df["dataset"].unique():
            df.loc[df["dataset"] == d, "dataset"] = _decorate(d)

    res_datasets = _datasets[0] if isinstance(datasets, str) else _datasets
    if df is None:
        return res_datasets
    else:
        return res_datasets, df


def rename_methods(mapping, methods, df=None, baselines=None):
    _methods = [mapping.get(m, m) for m in methods]

    res = [_methods]

    if df is not None:
        for m, rm in mapping.items():
            df.loc[df["method"] == m, "method"] = rm
        res.append(df)

    if baselines is not None:
        _baselines = [mapping.get(b, b) for b in baselines]
        res.append(_baselines)

    return tuple(res) if len(res) > 1 else res[0]


def get_extra_from_method(df, method):
    if isinstance(method, LEAP):
        df["true_solve"] = method._true_solve_log[-1]


def gen_method_df(df_len, **data):
    data = data | {k: [v] * df_len for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(data, orient="columns")
