import os
from contextlib import redirect_stdout
from glob import glob
from time import time

import numpy as np
import pandas as pd
import quapy as qp
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

import quacc as qc
import quacc.error
from quacc.data.datasets import RCV1_MULTICLASS_DATASETS, fetch_RCV1BinaryDataset, fetch_RCV1MulticlassDataset
from quacc.error import vanilla_acc
from quacc.experiments.util import split_validation
from quacc.models.cont_table import LEAP, QuAcc1xN2, QuAcc1xNp1, QuAccNxN
from quacc.utils.commons import true_acc

NUM_TEST = 1000
qp.environ["_R_SEED"] = 0


CSV_SEP = ","
LOCAL_DIR = os.path.join(qc.env["OUT_DIR"], "pg_results", "n2e")
CONFIG = "binary"
VERBOSE = True


class cleanEMQ(EMQ):
    @classmethod
    def EM(cls, tr_prev, posterior_probabilities, epsilon=...):
        with open(os.devnull, "w") as f:
            with redirect_stdout(f):
                return super().EM(tr_prev, posterior_probabilities, epsilon)


def sld():
    return cleanEMQ(LogisticRegression(), val_split=5)


def kdey():
    return KDEyML(LogisticRegression())


def get_bin_quaccs(h, acc_fn, q_class):
    return [
        QuAcc1xN2(h, acc_fn, q_class),
        QuAcc1xNp1(h, acc_fn, q_class),
        QuAccNxN(h, acc_fn, q_class),
    ]


def get_multi_quaccs(h, acc_fn, q_class):
    return [
        QuAcc1xN2(h, acc_fn, q_class),
        QuAccNxN(h, acc_fn, q_class),
    ]


def gen_bin_datasets():
    for dataset_name in ["CCAT", "GCAT", "MCAT", "ECAT"]:
        yield dataset_name, fetch_RCV1BinaryDataset(dataset_name)


def gen_multi_datasets():
    for dataset_name in RCV1_MULTICLASS_DATASETS:
        yield dataset_name, fetch_RCV1MulticlassDataset(dataset_name)


if CONFIG == "multiclass":
    get_quaccs = get_multi_quaccs
    gen_datasets = gen_multi_datasets
    qp.environ["SAMPLE_SIZE"] = 250
    basedir = CONFIG
elif CONFIG == "binary":
    get_quaccs = get_bin_quaccs
    gen_datasets = gen_bin_datasets
    qp.environ["SAMPLE_SIZE"] = 1000
    basedir = CONFIG


# fmt: off

def gen_methods(h, acc_fn):
    yield "N2E(SLD)", LEAP(h, acc_fn, sld(), verbose=True)
    yield "N2E(SLD)-optim", LEAP(h, acc_fn, sld(), always_optimize=True, verbose=True)

# fmt: on


def get_local_path(dataset, method_name, acc_name):
    return os.path.join(LOCAL_DIR, basedir, f"{dataset}_{method_name}_{acc_name}.csv")


def main():
    os.makedirs(LOCAL_DIR, exist_ok=True)
    dfs = []
    for dataset_name, (L, V, U) in gen_datasets():
        V, val_prot = split_validation(V)
        # h = LogisticRegression()
        h_param_grid = {"C": np.logspace(-4, -4, 9), "class_weight": ["balanced", None]}
        h_name, h = "LR-OPT", GridSearchCV(LogisticRegression(), h_param_grid, cv=5, n_jobs=-1)
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)
        accs = [
            ("vanilla_acc", vanilla_acc),
            # ("f1", f1_macro),
        ]

        h.fit(*L.Xy)
        print(f"trained {h_name} trained over {dataset_name}")
        for acc_name, acc_fn in accs:
            for method_name, method in gen_methods(h, acc_fn):
                local_path = get_local_path(dataset_name, method_name, acc_name)

                if os.path.exists(local_path):
                    method_df = pd.read_csv(local_path, sep=CSV_SEP)
                    dfs.append(method_df)
                    print(f"method {method_name} for {acc_name} exists, skipping")
                    continue

                t_init = time()
                method.fit(V)
                true_accs = np.array([true_acc(h, acc_fn, Ui) for Ui in test_prot()])
                estim_accs = method.batch_predict(test_prot)
                estim_accs = np.asarray(estim_accs)

                ae = quacc.error.ae(true_accs, estim_accs)
                t_method = time() - t_init
                print(f"method {method_name} for {acc_name} took {t_method:.3f}s")

                method_df = pd.DataFrame(
                    np.vstack([true_accs, estim_accs, ae]).T,
                    columns=["true_accs", "estim_accs", "ae"],
                )
                method_df["method"] = method_name
                method_df["acc_name"] = acc_name
                method_df["dataset"] = dataset_name
                method_df.to_csv(local_path, sep=CSV_SEP)
                dfs.append(method_df)

    results = pd.concat(dfs)

    print(results.pivot_table(values="ae", index="dataset", columns=["method", "acc_name"]))


def clean_results(path="*.csv"):
    glob_path = os.path.join(LOCAL_DIR, basedir, path)
    for f in glob(glob_path):
        os.remove(f)


if __name__ == "__main__":
    # clean_results()
    main()
