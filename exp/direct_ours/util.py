import numpy as np
import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier as MLP


def kdey(h: BaseEstimator = None):
    h = h if h is not None else MLP()
    return KDEyML(h)


def split_validation(V: LabelledCollection, ratio=0.6, repeats=100, sample_size=None, random_state=None):
    random_state = qp.environ["_R_SEED"] if random_state is None else random_state
    v_train, v_val = V.split_stratified(ratio, random_state=random_state)
    val_prot = UPP(
        v_val,
        repeats=repeats,
        sample_size=sample_size,
        return_type="labelled_collection",
        random_state=random_state,
    )
    return v_train, val_prot


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return np.around(prev[1:], decimals=4).tolist()
    else:
        return float(np.around(prev, decimals=4)[-1])


def gen_method_df(df_len, **data):
    data = data | {k: [v] * df_len for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(data, orient="columns")
