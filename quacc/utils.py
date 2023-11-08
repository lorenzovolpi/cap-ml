import functools
import os
import shutil
from pathlib import Path

import pandas as pd

from quacc.environment import env


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
    avg_report = {
        (n1, n2): (v / len(lst))
        for ((n1, n2), v) in summed_series.items()
        if n1 != "base"
    }
    return pd.DataFrame([avg_report], columns=idx)


def fmt_line_md(s):
    return f"> {s}  \n"


def create_dataser_dir(dir_name, update=False):
    base_out_dir = Path(env.OUT_DIR_NAME)
    if not base_out_dir.exists():
        os.mkdir(base_out_dir)

    dataset_dir = base_out_dir / dir_name
    env.OUT_DIR = dataset_dir
    if update:
        if not dataset_dir.exists():
            os.mkdir(dataset_dir)
    else:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        os.mkdir(dataset_dir)

    plot_dir_path = dataset_dir / "plot"
    env.PLOT_OUT_DIR = plot_dir_path
    if not plot_dir_path.exists():
        os.mkdir(plot_dir_path)
