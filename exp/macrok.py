from collections import defaultdict

import numpy as np
import pandas as pd
import quapy as qp
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from sklearn import clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP

from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import k_bin, k_macro, vanilla_acc
from cap.models.cont_table import O_LEAP
from cap.utils.commons import contingency_table

qp.environ["_R_SEED"] = 42
qp.environ["SAMPLE_SIZE"] = 1000

PROBLEM = "multiclass"


def gen_datasets():
    if PROBLEM == "binary":
        for dn in ["spambase"]:
            L, V, U = fetch_UCIBinaryDataset(dn)
            yield dn, L, V, U
    elif PROBLEM == "multiclass":
        for dn in ["poker_hand", "connect-4", "wine-quality", "academic-success", "hcv", "abalone", "mhr"]:
            L, V, U = fetch_UCIMulticlassDataset(dn)
            yield dn, L, V, U


if __name__ == "__main__":
    base_h = LogisticRegression()
    acc_fn = k_bin if PROBLEM == "binary" else k_macro
    results = defaultdict(list)
    for dn, L, V, U in gen_datasets():
        h = clone(base_h)
        h.fit(*L.Xy)
        V_posteriors = h.predict_proba(V.X)
        leap = O_LEAP(acc_fn, KDEyML(MLP())).fit(V, V_posteriors)
        test_prot = UPP(
            U,
            repeats=100,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )
        estim_accs, true_accs = [], []
        for i, Ui in enumerate(test_prot(), start=1):
            Ui_post = h.predict_proba(Ui.X)
            estim_ct = leap.predict_ct(Ui.X, Ui_post)
            true_ct = contingency_table(Ui.y, np.argmax(Ui_post, axis=1), n_classes=Ui.n_classes)
            results[(dn, "estim")].append(acc_fn(estim_ct))
            results[(dn, "true")].append(acc_fn(true_ct))

    df = pd.DataFrame.from_dict(results)
    print(df.to_string())
