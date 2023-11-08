import numpy as np


def from_name(err_name):
    assert err_name in ERROR_NAMES, f"unknown error {err_name}"
    callable_error = globals()[err_name]
    return callable_error


# def f1(prev):
#     # https://github.com/dice-group/gerbil/wiki/Precision,-Recall-and-F1-measure
#     if prev[0] == 0 and prev[1] == 0 and prev[2] == 0:
#         return 1.0
#     elif prev[0] == 0 and prev[1] > 0 and prev[2] == 0:
#         return 0.0
#     elif prev[0] == 0 and prev[1] == 0 and prev[2] > 0:
#         return float('NaN')
#     else:
#         recall = prev[0] / (prev[0] + prev[1])
#         precision = prev[0] / (prev[0] + prev[2])
#         return 2 * (precision * recall) / (precision + recall)


def f1(prev):
    den = (2 * prev[3]) + prev[1] + prev[2]
    if den == 0:
        return 0.0
    else:
        return (2 * prev[3]) / den


def f1e(prev):
    return 1 - f1(prev)


def acc(prev: np.ndarray) -> float:
    return (prev[0] + prev[3]) / np.sum(prev)


def accd(true_prevs: np.ndarray, estim_prevs: np.ndarray) -> np.ndarray:
    vacc = np.vectorize(acc, signature="(m)->()")
    a_tp = vacc(true_prevs)
    a_ep = vacc(estim_prevs)
    return np.abs(a_tp - a_ep)


def maccd(true_prevs: np.ndarray, estim_prevs: np.ndarray) -> float:
    return accd(true_prevs, estim_prevs).mean()


ACCURACY_ERROR = {maccd}
ACCURACY_ERROR_SINGLE = {accd}
ACCURACY_ERROR_NAMES = {func.__name__ for func in ACCURACY_ERROR}
ACCURACY_ERROR_SINGLE_NAMES = {func.__name__ for func in ACCURACY_ERROR_SINGLE}
ERROR_NAMES = ACCURACY_ERROR_NAMES | ACCURACY_ERROR_SINGLE_NAMES
