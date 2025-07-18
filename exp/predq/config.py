import os
from dataclasses import dataclass

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import quacc as qc
from exp.util import split_validation
from quacc.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset, sort_datasets_by_size
from quacc.error import f1, f1_macro, vanilla_acc
from quacc.models.cont_table import LEAP, OCE, PHD, NaiveCAP
from quacc.models.direct import ATC, DoC, PabloCAP, PrediQuant
from quacc.models.utils import OracleQuantifier
from quacc.utils.commons import contingency_table

PROJECT = "predq"
root_dir = os.path.join(qc.env["OUT_DIR"], PROJECT)
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0

PROBLEM = "multiclass"

_toggle = {
    "vanilla": True,
    "f1": False,
}


@dataclass
class DatasetBundle:
    L_prevalence: np.ndarray
    V: LabelledCollection
    U: LabelledCollection
    V1: LabelledCollection = None
    V2_prot: AbstractStochasticSeededProtocol = None
    test_prot: AbstractStochasticSeededProtocol = None
    V_posteriors: np.ndarray = None
    V1_posteriors: np.ndarray = None
    V2_prot_posteriors: np.ndarray = None
    test_prot_posteriors: np.ndarray = None
    test_prot_y_hat: np.ndarray = None
    test_prot_true_cts: np.ndarray = None

    def create_bundle(self, h: BaseEstimator):
        # generate test protocol
        self.test_prot = UPP(
            self.U,
            repeats=NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

        # split validation set
        self.V1, self.V2_prot = split_validation(self.V)

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

    @classmethod
    def mock(cls):
        return DatasetBundle(None, None, None, test_prot=lambda: [])


def kdey():
    return KDEyML(MLP())


def kdey_h(h: BaseEstimator):
    return KDEyML(h)


def gen_classifiers():
    yield "LR", LogisticRegression()
    yield "kNN", KNN(n_neighbors=10)
    yield "SVM", SVC(kernel="rbf", probability=True)
    yield "MLP", MLP()


def gen_datasets(only_names=False):
    if PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        for dn in _sorted_uci_names:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif PROBLEM == "multiclass":
        _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIMulticlassDataset)
        for dataset_name in _sorted_uci_names:
            dval = None if only_names else fetch_UCIMulticlassDataset(dataset_name)
            yield dataset_name, dval


def gen_acc_measure():
    multiclass = PROBLEM == "multiclass"
    if _toggle["vanilla"]:
        yield "vanilla_accuracy", vanilla_acc
    if _toggle["f1"]:
        yield "macro-F1", f1_macro if multiclass else f1


def gen_baselines(acc_fn):
    yield "Naive", NaiveCAP(acc_fn)
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")


def gen_baselines_vp(acc_fn, D: DatasetBundle):
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors)


def gen_CAP_cont_table(h, acc_fn):
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), log_true_solve=True)
    yield "S-LEAP(KDEy)", PHD(acc_fn, kdey())
    yield "O-LEAP(KDEy)", OCE(acc_fn, kdey(), optim_method="SLSQP")


def gen_CAP_direct_vp(h, acc_fn, D: DatasetBundle):
    yield "PrediQuant", PrediQuant(acc_fn, kdey_h(h), D.V2_prot, D.V2_prot_posteriors)
    yield "PrediQuant-a0", PrediQuant(acc_fn, kdey_h(h), D.V2_prot, D.V2_prot_posteriors, alpha=0)


def gen_CAP_direct(h, acc_fn):
    yield "PabloCAP", PabloCAP(acc_fn, kdey())


def gen_methods_with_oracle(h, acc_fn, D: DatasetBundle):
    oracle_q = OracleQuantifier([ui for ui in D.test_prot()])
    yield "LEAP(oracle)", LEAP(acc_fn, oracle_q)
    yield "S-LEAP(oracle)", PHD(acc_fn, oracle_q)
    yield "O-LEAP(oracle)", OCE(acc_fn, oracle_q, reuse_h=h, optim_method="SLSQP")


def gen_methods(h, D: DatasetBundle, with_oracle=False):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_baselines(acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_baselines_vp(acc_fn, D):
        yield name, method, D.V1, D.V1_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_CAP_direct(h, acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_CAP_direct_vp(h, acc_fn, D):
        yield name, method, D.V1, D.V1_posteriors
    if with_oracle:
        for name, method in gen_methods_with_oracle(h, acc_fn, D):
            yield name, method, D.V, D.V_posteriors


def get_classifier_names():
    return [name for name, _ in gen_classifiers()]


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def get_acc_names():
    return [acc_name for acc_name, _ in gen_acc_measure()]


def get_method_names(with_oracle=False):
    mock_h = LogisticRegression()
    mock_D = DatasetBundle.mock()

    names = [m for m, _, _, _ in gen_methods(mock_h, mock_D, with_oracle=with_oracle)]

    return names
