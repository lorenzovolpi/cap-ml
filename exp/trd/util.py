import glob
import os

import pandas as pd

from exp.trd.config import PROBLEM, root_dir


def local_path(dataset_name, cls_name, method_name, acc_name):
    parent_dir = os.path.join(root_dir, PROBLEM, acc_name, dataset_name, method_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{cls_name}.json")


def load_results(acc_name="*", dataset="*") -> pd.DataFrame:
    dfs = []
    for path in glob.glob(os.path.join(root_dir, PROBLEM, acc_name, dataset, "**", "*.json"), recursive=True):
        dfs.append(pd.read_json(path))

    return pd.concat(dfs, axis=0)


def rename_datasets(mapping, df, datasets: str | list[str]):
    if isinstance(datasets, str):
        _datasets = [mapping.get(d, d) for d in [datasets]]
    else:
        _datasets = [mapping.get(d, d) for d in datasets]
    for d, rd in mapping.items():
        df.loc[df["dataset"] == d, "dataset"] = rd

    if isinstance(datasets, str):
        return df, _datasets[0]
    else:
        return df, _datasets


def rename_methods(mapping, df, methods, baselines=None):
    _methods = [mapping.get(m, m) for m in methods]
    for m, rm in mapping.items():
        df.loc[df["method"] == m, "method"] = rm

    if baselines is None:
        return df, _methods
    else:
        _baselines = [mapping.get(b, b) for b in baselines]
        return df, _methods, _baselines


def decorate_datasets(df, datasets: str | list[str]):
    def _decorate(d):
        return r"\textsf{" + d + r"}"

    if isinstance(datasets, str):
        _datasets = [_decorate(d) for d in [datasets]]
    else:
        _datasets = [_decorate(d) for d in datasets]
    for d in df["dataset"].unique():
        df.loc[df["dataset"] == d, "dataset"] = _decorate(d)

    if isinstance(datasets, str):
        return df, _datasets[0]
    else:
        return df, _datasets
