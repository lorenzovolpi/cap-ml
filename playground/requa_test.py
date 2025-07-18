import itertools as IT
import os
from contextlib import redirect_stdout
from time import time
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression, Ridge

import quacc as qc
import quacc.error
from quacc.data.datasets import (
    RCV1_MULTICLASS_DATASETS,
    fetch_RCV1BinaryDataset,
    fetch_RCV1MulticlassDataset,
    fetch_UCIBinaryDataset,
)
from quacc.error import vanilla_acc
from quacc.experiments.util import fit_or_switch, get_logger, get_predictions, split_validation
from quacc.models.cont_table import QuAcc1xN2, QuAcc1xNp1, QuAccNxN
from quacc.models.regression import ReQua
from quacc.utils.commons import true_acc

NUM_TEST = 1000
qp.environ["_R_SEED"] = 0


CSV_SEP = ","
CONFIG = "binary"
VERBOSE = True

log = get_logger(id="requa_test")


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


def get_bin_quaccs(q_class):
    _, acc_fn = next(gen_accs())
    return [
        QuAcc1xN2(acc_fn, q_class),
        QuAcc1xNp1(acc_fn, q_class),
        QuAccNxN(acc_fn, q_class),
    ]


def get_multi_quaccs(q_class):
    _, acc_fn = next(gen_accs())
    return [
        QuAcc1xN2(acc_fn, q_class),
        QuAccNxN(acc_fn, q_class),
    ]


def gen_bin_datasets():
    _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
    _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
    for dn in _uci_names:
        dval = fetch_UCIBinaryDataset(dn)
        yield dn, dval


def gen_multi_datasets():
    for dataset_name in RCV1_MULTICLASS_DATASETS:
        yield dataset_name, fetch_RCV1MulticlassDataset(dataset_name)


def gen_classifiers():
    yield "LR", LogisticRegression()


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

def gen_methods(h, vprot, vprot_posteriors):
    _, acc_fn = next(gen_accs())
    quacc_params = {
        # "q_class__classifier__C": np.logspace(-3, 3, 7),
        # "q_class__classifier__class_weight": [None, "balanced"],
        "add_X": [True, False],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }
    yield "ReQua(SLD-LinReg)", ReQua(acc_fn, LinReg(), get_quaccs(sld()), quacc_params, vprot, vprot_posteriors, verbose=True)
    yield "ReQua(SLD-Ridge)", ReQua(acc_fn, Ridge(), get_quaccs(sld()), quacc_params, vprot, vprot_posteriors, verbose=True)
    yield "ReQua(SLD-KRR)", ReQua(acc_fn, KRR(), get_quaccs(sld()), quacc_params, vprot, vprot_posteriors, verbose=True)

# fmt: on


def get_method_names():
    mock_h = LogisticRegression()
    return [m for m, _ in gen_methods(mock_h)]


def gen_accs():
    yield "vanilla_acc", vanilla_acc
    # yield "f1", f1_macro


def get_local_path(cls_name, acc_name, dataset, method_name):
    parent_dir = os.path.join(qc.env["OUT_DIR"], "requa_test", basedir, cls_name, acc_name, dataset)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.csv")


def all_exist_pre_check(dataset_name, cls_name, L):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_accs()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        path = get_local_path(dataset_name, cls_name, method, acc, L)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def preload_existing(dataset_name, cls_name, L):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_accs()]

    dfs = []
    for method, acc in IT.product(method_names, acc_names):
        path = get_local_path(dataset_name, cls_name, method, acc, L)
        method_df = pd.read_csv(path, sep=CSV_SEP)
        dfs.append(method_df)

    return dfs


def experiments():
    log.info("-" * 31 + "  start  " + "-" * 31)

    dfs = []
    for cls_name, h in gen_classifiers():
        for dataset_name, (L, V, U) in gen_datasets():
            V1, V2_prot = split_validation(V)
            test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)

            log.info(f"Training {cls_name} over {dataset_name}")
            h.fit(*L.Xy)

            # generate posteriors
            V1_posteriors = h.predict_proba(V1.X)
            V2_prot_posteriors = [h.predict_proba(sample.X) for sample in V2_prot()]
            test_prot_posteriors, test_prot_y_hat = [], []
            for sample in test_prot():
                P = h.predict_proba(sample.X)
                test_prot_posteriors.append(P)
                test_prot_y_hat.append(np.argmax(P, axis=-1))

            # precompute the actual accuracy values
            true_accs = {}
            for acc_name, acc_fn in gen_accs():
                true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

            for method_name, method in gen_methods(h, V2_prot, V2_prot_posteriors):
                t_train = None
                for acc_name, acc_fn in gen_accs():
                    local_path = get_local_path(cls_name, acc_name, dataset_name, method_name)

                    if os.path.exists(local_path):
                        method_df = pd.read_csv(local_path, sep=CSV_SEP)
                        dfs.append(method_df)
                        log.info(f"{method_name} on {acc_name} exists, skipping")
                        continue

                    try:
                        method, _t_train = fit_or_switch(method, V1, V1_posteriors, acc_fn, t_train is not None)
                        t_train = t_train if _t_train is None else _t_train
                        estim_accs, t_test_ave = get_predictions(method, test_prot, test_prot_posteriors, False)
                    except Exception as e:
                        print_exception(e)
                        log.warning(f"{method_name}: {acc_name} gave error '{e}' - skipping")
                        continue

                    ae = quacc.error.ae(np.array(true_accs[acc_name]), np.array(estim_accs))

                    method_df = pd.DataFrame(
                        np.vstack([true_accs[acc_name], estim_accs, ae]).T,
                        columns=["true_accs", "estim_accs", "ae"],
                    )
                    method_df["method"] = method_name
                    method_df["acc_name"] = acc_name
                    method_df["dataset"] = dataset_name
                    method_df["cls"] = cls_name
                    log.info(f"{method_name} on {acc_name} done")
                    method_df.to_csv(local_path, sep=CSV_SEP)
                    dfs.append(method_df)

    log.info("-" * 32 + "  end  " + "-" * 32)

    results = pd.concat(dfs)

    print(results.pivot_table(values="ae", index="dataset", columns=["method", "acc_name"]))


# def clean_results(path="*.csv"):
#     glob_path = os.path.join(LOCAL_DIR, basedir, path)
#     for f in glob(glob_path):
#         os.remove(f)


if __name__ == "__main__":
    # clean_results()
    try:
        experiments()
    except Exception as e:
        log.error(e)
        print_exception(e)
