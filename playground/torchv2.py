import os
from collections import defaultdict
from time import time

import numpy as np
import pandas as pd
import quapy as qp
import torch
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from scipy.special import softmax
from sklearn.base import BaseEstimator

import quacc as qc
from exp.leap.config import kdey
from exp.util import get_ct_predictions, split_validation
from quacc.error import vanilla_acc
from quacc.models.cont_table import LEAP, OCE, PHD
from quacc.models.direct import DoC

qp.environ["_R_SEED"] = 0
qp.environ["SAMPLE_SIZE"] = 1000

base_dir = os.path.join(qc.env["OUT_DIR"], "transformers", "embeds")


class BaseEstimatorAdapter(BaseEstimator):
    def __init__(self, V_hidden_states, U_hidden_states, V_logits, U_logits):
        self.hs = np.vstack([V_hidden_states, U_hidden_states])
        self.logits = np.vstack([V_logits, U_logits])

        hashes = self._hash(self.hs)
        self._dict = defaultdict(lambda: [])
        for i, hash in enumerate(hashes):
            self._dict[hash].append(i)

    def _hash(self, X):
        return np.around(np.abs(X).sum(axis=-1) * self.hs.shape[0])

    def predict_proba(self, X: np.ndarray):
        def f(data, hash):
            _ids = np.array(self._dict[hash])
            _m = self.hs[_ids, :]
            _eq_idx = np.nonzero((_m == data).all(axis=-1))[0][0]
            return _ids[_eq_idx]

        hashes = self._hash(X)
        logits_idx = np.vectorize(f, signature="(m),()->()")(X, hashes)
        _logits = self.logits[logits_idx, :]
        return softmax(_logits, axis=-1)

    def decision_function(self, X: np.ndarray):
        return self.predict_proba(X)


def load_model_dataset(dataset_name, model_name):
    parent_dir = os.path.join(base_dir, dataset_name, model_name)

    V_X = torch.load(os.path.join(parent_dir, "hidden_states.validation.pt")).numpy()
    V_logits = torch.load(os.path.join(parent_dir, "logits.validation.pt")).numpy()
    V_labels = torch.load(os.path.join(parent_dir, "labels.validation.pt")).numpy()
    U_X = torch.load(os.path.join(parent_dir, "hidden_states.test.pt")).numpy()
    U_logits = torch.load(os.path.join(parent_dir, "logits.test.pt")).numpy()
    U_labels = torch.load(os.path.join(parent_dir, "labels.test.pt")).numpy()

    V = LabelledCollection(V_X, V_labels, classes=np.unique(V_labels))
    U = LabelledCollection(U_X, U_labels, classes=np.unique(U_labels))

    model = BaseEstimatorAdapter(V_X, U_X, V_logits, U_logits)

    return model, (V, U)


if __name__ == "__main__":
    dataset_name = "imdb"
    model_name = "bert-base-uncased"

    model, (V, U) = load_model_dataset(dataset_name, model_name)

    test_prot = UPP(U, repeats=1000, random_state=0, return_type="labelled_collection")

    V1, V2_prot = split_validation(V)

    V_posteriors = model.predict_proba(V.X)
    V1_posteriors = model.predict_proba(V1.X)
    V2_prot_posteriors = [model.predict_proba(Vi.X) for Vi in V2_prot()]

    test_prot_posteriors = [model.predict_proba(Ui.X) for Ui in test_prot()]
    test_prot_yhat = [np.argmax(Ui_P, axis=-1) for Ui_P in test_prot_posteriors]
    test_prot_y = [Ui.y for Ui in test_prot()]

    true_accs = [vanilla_acc(y, yhat) for y, yhat in zip(test_prot_y, test_prot_yhat)]

    methods = [
        ("leap", LEAP(vanilla_acc, kdey(), reuse_h=model)),
        ("phd", PHD(vanilla_acc, kdey(), reuse_h=model)),
        ("oce", OCE(vanilla_acc, kdey(), reuse_h=model, optim_method="SLSQP")),
        ("doc", DoC(vanilla_acc, V2_prot, V2_prot_posteriors)),
    ]

    dfs = []
    for method_name, method in methods:
        if method_name in ["leap", "phd", "oce"]:
            method.fit(V, V_posteriors)
        else:
            method.fit(V1, V1_posteriors)
        estim_accs, estim_cts, t_test_ave = get_ct_predictions(method, test_prot, test_prot_posteriors)
        ae = qc.error.ae(np.array(true_accs), np.array(estim_accs))
        method_df = pd.DataFrame(
            np.vstack([true_accs, estim_accs, ae]).T, columns=["true_accs", "estim_accs", "acc_err"]
        )
        method_df["method"] = method_name
        dfs.append(method_df)

    df = pd.concat(dfs, axis=0)
    pivot = pd.pivot_table(df, values="acc_err", index=["method"])

    print(pivot)
