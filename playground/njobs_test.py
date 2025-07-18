from time import time

import joblib
import numpy as np
import quapy as qp
from quapy.method.aggregative import PACC
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from quacc.data.datasets import fetch_UCIMulticlassDataset
from quacc.error import f1_macro
from quacc.experiments.util import split_validation
from quacc.models.cont_table import QuAcc1xN2
from quacc.models.model_selection import GridSearchCAP

qp.environ["_R_SEED"] = 0
qp.environ["SAMPLE_SIZE"] = 250


class fastPACC(PACC):
    def __init__(self, classifier: BaseEstimator, val_split=5, n_jobs=None, solver="minimize"):
        super().__init__(classifier, val_split, n_jobs, solver)
        self.classifier = classifier
        self.val_split = val_split
        self.n_jobs = qp._get_njobs(n_jobs)
        self.solver = solver

    @classmethod
    def getPteCondEstim(cls, classes, y, y_):
        n_classes = len(classes)

        tinit = time()
        idxs = (y == np.arange(n_classes).reshape((-1, 1))).astype(int)
        idxs = idxs / idxs.sum(axis=-1, keepdims=True)

        confusion = np.matmul(idxs, y_)

        tmul = time()

        _zeros_idx = np.nonzero(confusion.sum(axis=-1) == 0)[0]
        confusion[_zeros_idx] = np.eye(n_classes)[_zeros_idx]

        confusion = confusion.T

        tfix = time()

        print(f"{tmul - tinit:.3f}s; {tfix - tmul:.3f}s")

        return confusion


if __name__ == "__main__":
    L, V, U = fetch_UCIMulticlassDataset("letter")
    V, val_prot = split_validation(V)

    h = LogisticRegression().fit(*L.Xy)

    pacc_lr_params = {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
        # "add_X": [True, False],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }

    joblib.parallel_config(verbose=60)
    model = GridSearchCAP(
        QuAcc1xN2(h, f1_macro, fastPACC(LogisticRegression())),
        pacc_lr_params,
        val_prot,
        f1_macro,
        refit=True,
    ).fit(V)
