
from statistics import mean
from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_validate
from quapy.data import LabelledCollection


def kfcv(c_model: BaseEstimator, train: LabelledCollection) -> Dict:
    scoring = ["f1_macro"]
    scores = cross_validate(c_model, train.X, train.y, scoring=scoring)
    return {
        "f1_score": mean(scores["test_f1_macro"])
    }
