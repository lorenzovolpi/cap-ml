import json
import math
import os

import numpy as np
import quapy as qp
import scipy.special as sp
from quapy.data.datasets import UCI_MULTICLASS_DATASETS
from quapy.functional import get_nprevpoints_approximation, num_prevalence_combinations
from quapy.protocol import APP
from sklearn import base

from exp.trd.config import root_dir
from quacc.data.datasets import fetch_UCIMulticlassDataset

qp.environ["_R_SEED"] = 0
qp.environ["SAMPLE_SIZE"] = 100


def app_grid_multi():
    res = {}
    budget = 20
    for dataset_name in UCI_MULTICLASS_DATASETS:
        L, V, U = fetch_UCIMulticlassDataset(dataset_name)
        n_classes = L.n_classes
        temp = 1.0 / math.log(n_classes)
        budget = budget if budget >= n_classes else n_classes + 1
        _approx = get_nprevpoints_approximation(budget, n_classes=n_classes)
        _n_prevs = num_prevalence_combinations(_approx, n_classes)
        print(f"{dataset_name} {n_classes=} {_approx=} {_n_prevs=}")

        _prot = APP(L, n_prevalences=_approx, repeats=1, random_state=qp.environ["_R_SEED"])
        _grid = _prot.prevalence_grid()
        _prevs = np.hstack([_grid, (1 - _grid.sum(axis=-1))[:, np.newaxis]])
        # _apt_prevs = _prevs + 0.1
        # _apt_prevs = _prevs + 1.0 / n_classes
        _apt_prevs = _prevs / temp
        _s_prevs = sp.softmax(_apt_prevs, axis=-1)
        assert _n_prevs == _prevs.shape[0], "estimated n_prevs not matching with true value"
        _prev_zip = [{"orig": p.tolist(), "scaled": sp.tolist()} for p, sp in zip(_prevs, _s_prevs)]
        res[dataset_name] = dict(
            n_classes=n_classes,
            n_prevpoints=_approx,
            n_prevs=_n_prevs,
            prevs=_prev_zip,
        )

    with open(os.path.join(root_dir, "app_grid.json"), "w") as f:
        json.dump(res, f, indent=2)


def app_grid_multi2(alpha=2.0):
    res = {}
    for dataset_name in UCI_MULTICLASS_DATASETS:
        L, V, U = fetch_UCIMulticlassDataset(dataset_name)
        n_classes = L.n_classes
        _unif = 1.0 / n_classes
        _high = _unif * alpha
        _low = (1.0 - _high) / (n_classes - 1)
        _mask = np.eye(n_classes)
        _prevs = np.where(_mask == 1, _high, _low)
        _n_prevs = _prevs.shape[0]
        print(f"{dataset_name} {n_classes=} {_n_prevs=}")
        res[dataset_name] = dict(
            n_classes=n_classes,
            n_prevs=_n_prevs,
            alpha=alpha,
            prevs=_prevs.tolist(),
        )

    with open(os.path.join(root_dir, "app_grid.json"), "w") as f:
        json.dump(res, f, indent=2)


def lin_try():
    ncls_pool = [3, 4, 5, 6, 7, 8, 10, 11, 15]
    _alpha = 1.8
    print(_alpha)
    for n in ncls_pool:
        print(f"n_classes={n}", end="  ")
        _delta = _alpha / (n - 1)
        y = (1 - _delta) / n
        x = y + _delta
        prevs = [x] + ([y] * (n - 1))
        print(prevs)


def lin_try2():
    ncls_pool = [3, 4, 5, 6, 7, 8, 10, 11, 15]
    _alpha = 0.03
    _high = 0.7
    print(f"alpha={_alpha}")
    for n in ncls_pool:
        _U = 1.0 / n
        print(f"n_classes={n} U={_U}", end="  ")
        x = _high - (_alpha * n)
        y = (1 - x) / (n - 1)
        prevs = [x] + ([y] * (n - 1))
        print(prevs)


def lin_try3():
    ncls_pool = [3, 4, 5, 6, 7, 8, 10, 11, 15]
    _alpha = 2
    print(f"alpha={_alpha}")
    for n in ncls_pool:
        _U = 1.0 / n
        print(f"n_classes={n} U={_U}", end="  ")
        x = _U * _alpha
        y = (1 - x) / (n - 1)
        prevs = [x] + ([y] * (n - 1))
        print(prevs)


if __name__ == "__main__":
    app_grid_multi2()
