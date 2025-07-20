import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def from_name(err_name):
    assert err_name in ERROR_NAMES, f"unknown error {err_name}"
    callable_error = globals()[err_name]
    return callable_error


def is_from_cont_table(param1, param2):
    if param2 is None and isinstance(param1, np.ndarray) and param1.ndim == 2 and (param1.shape[0] == param1.shape[1]):
        return True
    elif isinstance(param1, np.ndarray) and isinstance(param2, np.ndarray) and param1.shape == param2.shape:
        return False
    else:
        raise ValueError("parameters for evaluation function not understood")


def vanilla_acc(param1, param2=None):
    if is_from_cont_table(param1, param2):
        return _vanilla_acc_from_ct(param1)
    else:
        return accuracy_score(param1, param2)


def f1(param1, param2=None, average="binary"):
    _warning = False
    if is_from_cont_table(param1, param2):
        if param1.shape[0] > 2 and average == "binary":
            _warning = True
            average = "macro"
        _f1_score = _f1_from_ct(param1, average=average)
    else:
        if len(np.unique(np.hstack([param1, param2]))) > 2 and average == "binary":
            _warning = True
            average = "macro"
        _f1_score = f1_score(param1, param2, average=average, zero_division=1.0)

    if _warning:
        print("Warning: 'binary' average is not available for multiclass F1. Defaulting to 'macro' F1.")
    return _f1_score


def f1_macro(param1, param2=None):
    return f1(param1, param2, average="macro")


def f1_micro(param1, param2=None):
    return f1(param1, param2, average="micro")


def _vanilla_acc_from_ct(cont_table):
    return np.diag(cont_table).sum() / cont_table.sum()


def _f1_from_ct(cont_table, average):
    n = cont_table.shape[0]
    if average == "binary":
        tp = cont_table[1, 1]
        fp = cont_table[0, 1]
        fn = cont_table[1, 0]
        return _f1_bin(tp, fp, fn)
    elif average == "macro":
        f1_per_class = []
        for i in range(n):
            tp = cont_table[i, i]
            fp = cont_table[:, i].sum() - tp
            fn = cont_table[i, :].sum() - tp
            f1_per_class.append(_f1_bin(tp, fp, fn))
        return np.mean(f1_per_class)
    elif average == "micro":
        tp, fp, fn = 0, 0, 0
        tp = np.diag(cont_table).sum()
        fp = fn = cont_table.sum() - tp
        return _f1_bin(tp, fp, fn)
    else:
        raise ValueError(f"Unknown F1 average {average}")


def _f1_bin(tp, fp, fn):
    if 2 * tp + fp + fn == 0:
        return 1
    else:
        return (2 * tp) / (2 * tp + fp + fn)


def ae(true_accs, estim_accs):
    """Computes the absolute error between true and estimated accuracy value pairs.

    :param true_accs: array-like of shape `(n_samples,)` with the true accuracy values
    :param estim_accs: array-like of shape `(n_samples,)` with the estimated accuracy values
    :return: absolute error
    """
    assert true_accs.shape == estim_accs.shape, f"wrong shape {true_accs.shape} vs. {estim_accs.shape}"
    return np.abs(true_accs - estim_accs)


def mae(true_accs, estim_accs):
    """Computes the mean absolute error between true and estimated accuracy value pairs.

    :param true_accs: array-like of shape `(n_samples,)` with the true accuracy values
    :param estim_accs: array-like of shape `(n_samples,)` with the estimated accuracy values
    :return: mean absolute error
    """
    return ae(true_accs, estim_accs).mean()


def se(true_accs, estim_accs):
    """Computes the squared error between true and estimated accuracy value pairs.

    :param true_accs: array-like of shape `(n_samples,)` with the true accuracy values
    :param estim_accs: array-like of shape `(n_samples,)` with the estimated accuracy values
    :return: absolute error
    """
    return (true_accs - estim_accs) ** 2


def mse(true_accs, estim_accs):
    """Computes the mean squared error between true and estimated accuracy value pairs.

    :param true_accs: array-like of shape `(n_samples,)` with the true accuracy values
    :param estim_accs: array-like of shape `(n_samples,)` with the estimated accuracy values
    :return: mean squared error
    """
    return se(true_accs, estim_accs).mean()


def _reshape_for_error(true_accs, estim_accs):
    _true_accs = np.array(true_accs).reshape(-1, 1)
    _estim_accs = np.array(estim_accs).reshape(-1, 1)
    return _true_accs, _estim_accs


ACCURACY_MEASURE = {vanilla_acc, f1}
ACCURACY_ERROR = {mae, mse}
ACCURACY_ERROR_SINGLE = {ae, se}
ACCURACY_MEASURE_NAMES = {acc_fn.__name__ for acc_fn in ACCURACY_MEASURE}
ACCURACY_ERROR_NAMES = {err.__name__ for err in ACCURACY_ERROR}
ACCURACY_ERROR_SINGLE_NAMES = {err.__name__ for err in ACCURACY_ERROR_SINGLE}
ERROR_NAMES = ACCURACY_MEASURE_NAMES | ACCURACY_ERROR_NAMES | ACCURACY_ERROR_SINGLE_NAMES
