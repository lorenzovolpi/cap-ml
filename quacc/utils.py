
import functools
import pandas as pd

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
        return {
            (n1, n2): v + s2[(n1, n2)] for ((n1, n2), v) in s1.items()
        }

    lst = df.to_dict(orient="records")[1:-1]
    summed_series = functools.reduce(_reduce_func, lst)
    idx = df.columns.drop([("base", "T"), ("base", "F")])
    avg_report = {
        (n1, n2): (v / len(lst))
        for ((n1, n2), v) in summed_series.items()
        if n1 != "base"
    }
    return pd.DataFrame([avg_report], columns=idx)