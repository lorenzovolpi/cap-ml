import itertools as IT
import os
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP

import cap
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import vanilla_acc
from cap.models.cont_table import O_LEAP
from cap.models.direct import RQBS, PrediQuant
from cap.utils.commons import contingency_table, true_acc_from_posteriors

PROBLEM = "multiclass"
NUM_TEST = 1000
qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0
basedir = os.path.join("output", "tests")
json_path = os.path.join(basedir, f"results_direct_ours_{PROBLEM}.json")


def kdey(h: BaseEstimator = None):
    h = h if h is not None else MLP()
    return KDEyML(h)


def split_validation(V: LabelledCollection, ratio=0.6, repeats=100, sample_size=None, random_state=None):
    random_state = qp.environ["_R_SEED"] if random_state is None else random_state
    v_train, v_val = V.split_stratified(ratio, random_state=random_state)
    val_prot = UPP(
        v_val,
        repeats=repeats,
        sample_size=sample_size,
        return_type="labelled_collection",
        random_state=random_state,
    )
    return v_train, val_prot


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return np.around(prev[1:], decimals=4).tolist()
    else:
        return float(np.around(prev, decimals=4)[-1])


def gen_method_df(df_len, **data):
    data = data | {k: [v] * df_len for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(data, orient="columns")


def gen_acc_measures():
    yield "vanilla_accuracy", vanilla_acc


class Clsf:
    def __init__(self, name: str, h: BaseEstimator) -> None:
        self.name = name
        self.h = clone(h)

    def fit(self, X, y):
        self.h.fit(X, y)
        return self


@dataclass
class DatasetBundle:
    name: str
    L: LabelledCollection
    V: LabelledCollection
    U: LabelledCollection
    L_prevalence: np.ndarray = None
    V1: LabelledCollection = None
    V2_prot: AbstractStochasticSeededProtocol = None
    test_prot: AbstractStochasticSeededProtocol = None
    V_posteriors: np.ndarray = None
    V1_posteriors: np.ndarray = None
    V2_prot_posteriors: np.ndarray = None
    test_prot_posteriors: np.ndarray = None
    test_prot_y_hat: np.ndarray = None
    test_prot_true_cts: np.ndarray = None
    true_accs: dict = None
    n_classes: int = -1

    def create_sets(self):
        self.L_prevalence = self.L.prevalence()
        self.n_classes = self.L_prevalence.shape[0]

        # generate test protocol
        self.test_prot = UPP(
            self.U,
            repeats=NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

        # split validation set
        self.V1, self.V2_prot = split_validation(self.V, random_state=qp.environ["_R_SEED"])

        return self

    def get_posteriors(self, h: BaseEstimator):
        # precomumpute model posteriors for validation sets
        self.V_posteriors = h.predict_proba(self.V.X)
        self.V1_posteriors = h.predict_proba(self.V1.X)
        self.V2_prot_posteriors = []
        for sample in self.V2_prot():
            self.V2_prot_posteriors.append(h.predict_proba(sample.X))

        # precomumpute model posteriors for test samples
        self.test_prot_posteriors, self.test_prot_y_hat, self.test_prot_true_cts = [], [], []
        for sample in self.test_prot():
            P = h.predict_proba(sample.X)
            self.test_prot_posteriors.append(P)
            y_hat = np.argmax(P, axis=-1)
            self.test_prot_y_hat.append(y_hat)
            self.test_prot_true_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))

        return self

    def get_true_accs(self, accs: List[Tuple[str, Callable[[np.ndarray, np.ndarray], float]]]):
        # compute true accs for h on dataset
        self.true_accs = {} if self.true_accs is None else self.true_accs
        missing_accs = [(acc_name, acc_fn) for acc_name, acc_fn in accs if acc_name not in self.true_accs]
        for acc_name, acc_fn in missing_accs:
            self.true_accs[acc_name] = [
                true_acc_from_posteriors(acc_fn, Ui, Ui_P)
                for Ui, Ui_P in IT.zip_longest(self.test_prot(), self.test_prot_posteriors)
            ]

        if len(missing_accs) > 0:
            self.updated = True

        return self

    @classmethod
    def mock(cls, dataset_name="mock"):
        return DatasetBundle(dataset_name, None, None, None, test_prot=lambda: [])


# fmt: off
def gen_methods(acc_fn: Callable, cl: Clsf, D: DatasetBundle):
    yield "PrediQuant", PrediQuant(acc_fn, kdey(cl.h), D.V2_prot, D.V2_prot_posteriors, alpha=0.3), D.V1, D.V1_posteriors
    yield "PrediQuantMM", PrediQuant(acc_fn, kdey(cl.h), D.V2_prot, D.V2_prot_posteriors, alpha=0.3), D.V1, D.V1_posteriors,
    yield "RQBS", RQBS(acc_fn, kdey()), D.V, D.V_posteriors
    yield "O-LEAP", O_LEAP(acc_fn, kdey()), D.V, D.V_posteriors

# fmt: on


def gen_datasets():
    if PROBLEM == "binary":
        uci_binary = [
            "breast-cancer",
            "german",
            "haberman",
            "ionosphere",
            "mammographic",
            "semeion",
            "sonar",
            "spambase",
            "spectf",
            "tictactoe",
            "transfusion",
            "wdbc",
        ]
        for dn in uci_binary:
            yield dn, fetch_UCIBinaryDataset(dn)
    elif PROBLEM == "multiclass":
        for dn in UCI_MULTICLASS_DATASETS:
            yield dn, fetch_UCIMulticlassDataset(dn)


def gen_classifiers():
    yield "LR", LogisticRegression()
    yield "MLP", MLP()


def train_predict(clsf_name, h, dataset_name, L, V, U):
    cl = Clsf(clsf_name, h).fit(*L.Xy)
    D = DatasetBundle(dataset_name, L, V, U).create_sets().get_posteriors(cl.h).get_true_accs(list(gen_acc_measures()))

    return cl, D


if __name__ == "__main__":
    os.makedirs(basedir, exist_ok=True)
    if not os.path.exists(json_path):
        cld: list[Tuple[Clsf, DatasetBundle]] = []
        for dataset_name, (L, V, U) in gen_datasets():
            for clsf_name, h in gen_classifiers():
                cld.append(train_predict(clsf_name, h, dataset_name, L, V, U))
                print(f"{clsf_name}@{dataset_name} trained")

        dfs = []
        for cl, D in cld:
            for acc_name, acc_fn in gen_acc_measures():
                for method_name, method, val, val_posteriors in gen_methods(acc_fn, cl, D):
                    val_prev = get_plain_prev(val.prevalence())
                    method.fit(val, val_posteriors)
                    estim_accs = method.batch_predict(D.test_prot, D.test_prot_posteriors)
                    true_accs = D.true_accs[acc_name]
                    aes = cap.error.ae(np.array(true_accs), np.array(estim_accs)).tolist()

                    df_len = D.test_prot.total()
                    method_df = gen_method_df(
                        df_len,
                        estim_accs=estim_accs,
                        true_accs=D.true_accs[acc_name],
                        aes=aes,
                        classifier=cl.name,
                        method=method_name,
                        dataset=D.name,
                        acc_name=acc_name,
                        train_prev=[D.L_prevalence] * df_len,
                        val_prev=[val_prev] * df_len,
                    )
                    dfs.append(method_df)
                    print(f"{method_name}@[{cl.name}-{D.name}-{acc_name}] done.")

        df = pd.concat(dfs, ignore_index=True, axis=0)
        df.to_json(json_path)

    df = pd.read_json(json_path)
    for acc_name, _ in gen_acc_measures():
        for c_name, h in gen_classifiers():
            sub_df = df[(df["classifier"] == c_name) & (df["acc_name"] == acc_name)]
            pivot = pd.pivot_table(
                sub_df,
                index=["dataset"],
                columns=["method"],
                values=["aes"],
            )
            pivot.loc["mean"] = pivot.mean(numeric_only=True, axis=0)
            print(f"\n{c_name} - {acc_name}\n", pivot)
