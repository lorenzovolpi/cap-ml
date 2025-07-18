import pdb

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from exp.util import get_predictions, split_validation
from quacc.data.datasets import fetch_UCIBinaryDataset
from quacc.error import ae, vanilla_acc
from quacc.models.cont_table import LEAP, OCE
from quacc.utils.commons import true_acc

qp.environ["_R_SEED"] = 0
qp.environ["SAMPLE_SIZE"] = 100
NUM_TEST = 100


def kdey():
    return KDEyML(LogisticRegression())


acc_name, acc_fn = "vanilla_accuracy", vanilla_acc

if __name__ == "__main__":
    dfs = []
    for dataset_name in [d for d in UCI_BINARY_DATASETS if d not in ["acute.a", "acute.b", "balance.2", "iris.1"]]:
        L, V, U = fetch_UCIBinaryDataset(dataset_name)
        h = LogisticRegression()
        h.fit(*L.Xy)

        test_prot = UPP(
            U,
            repeats=NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

        # V1, V2_prot = split_validation(V)

        V_posteriors = h.predict_proba(V.X)
        # V1_posteriors = h.predict_proba(V1.X)
        # V2_prot_posteriors = []
        # for sample in V2_prot():
        #     V2_prot_posteriors.append(h.predict_proba(sample.X))

        test_prot_posteriors, test_prot_y_hat = [], []
        for sample in test_prot():
            P = h.predict_proba(sample.X)
            test_prot_posteriors.append(P)
            test_prot_y_hat.append(np.argmax(P, axis=-1))

        true_accs = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

        for method_name, method in [
            ("OCE(KDEy)-L-BFGS-B", OCE(acc_fn, kdey(), reuse_h=h, optim_method="L-BFGS-B").fit(V, V_posteriors))
        ]:
            estim_accs = []
            for Ui, Ui_post in zip(test_prot(), test_prot_posteriors):
                _ct = method.predict_ct(Ui.X, Ui_post)
                _ct = _ct / _ct.sum()
                estim_accs.append(method.acc_fn(_ct))
            acc_err = [ae(np.array(ta), np.array(ea)) for ta, ea in zip(true_accs, estim_accs)]
            method_df = pd.DataFrame(np.vstack([estim_accs, acc_err]).T, columns=["estim_acc", "acc_err"])
            method_df["method"] = method_name
            method_df["dataset"] = dataset_name
            dfs.append(method_df)

        for method_name, method in [
            ("LEAP", LEAP(acc_fn, kdey(), reuse_h=h)),
            ("OCE(KDEy)-SLSQP", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP")),
        ]:
            method.fit(V, V_posteriors)
            estim_accs = method.batch_predict(test_prot, test_prot_posteriors)
            acc_err = [ae(np.array(ta), np.array(ea)) for ta, ea in zip(true_accs, estim_accs)]
            method_df = pd.DataFrame(np.vstack([estim_accs, acc_err]).T, columns=["estim_acc", "acc_err"])
            method_df["method"] = method_name
            method_df["dataset"] = dataset_name
            dfs.append(method_df)

        print(f"{dataset_name} done.")

    df = pd.concat(dfs, axis=0)
    pivot = pd.pivot_table(df, index=["dataset"], columns=["method"], values="acc_err")
    print(pivot.mean(axis=0))
    # acc_err = ae(np.array(true_accs), np.array(estim_accs))
