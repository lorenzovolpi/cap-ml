import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import quapy as qp
import torch
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import ACC, CC, EMQ, DistributionMatchingY, KDEyML
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import exp.leap.env as env
import quacc as qc
from exp.util import split_validation
from quacc.data.datasets import (
    fetch_RCV1WholeDataset,
    fetch_UCIBinaryDataset,
    fetch_UCIMulticlassDataset,
    sort_datasets_by_size,
)
from quacc.error import f1, f1_macro, f1_micro, vanilla_acc
from quacc.models._large_models import BaseEstimatorAdapter
from quacc.models.cont_table import CBPE, LEAP, OCE, PHD, NaiveCAP
from quacc.models.direct import ATC, COT, Q_COT, DispersionScore, DoC, NuclearNorm
from quacc.models.utils import OracleQuantifier
from quacc.utils.commons import contingency_table

_toggle = {
    "mlp": True,
    "same_h": True,
    "vanilla": True,
    "f1": True,
    "cc": True,
    "acc": True,
    "slsqp": False,
    "oracle": False,
}


@dataclass
class EXP:
    code: int
    cls_name: str
    dataset_name: str
    acc_name: str
    method_name: str
    df: pd.DataFrame = None
    t_train: float = None
    t_test_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


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

    def get_test_prot(self, sample_size=None):
        return UPP(
            self.U,
            repeats=env.NUM_TEST,
            sample_size=sample_size,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

    def create_bundle(self, h: BaseEstimator, sample_size=None):
        # generate test protocol
        self.test_prot = self.get_test_prot(sample_size=sample_size)
        # split validation set
        self.V1, self.V2_prot = split_validation(self.V, sample_size=sample_size)

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


def cc():
    return CC(MLP())


def acc():
    return ACC(MLP())


def kdey():
    return KDEyML(MLP())


def gen_classifiers():
    yield "LR", LogisticRegression()
    yield "kNN", KNN(n_neighbors=10)
    yield "SVM", SVC(kernel="rbf", probability=True)
    yield "MLP", MLP()
    yield "RFC", RFC()


def gen_datasets(only_names=False):
    if env.PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        for dn in _sorted_uci_names:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif env.PROBLEM == "multiclass":
        # _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_skip = []  # ["wine-quality"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIMulticlassDataset)
        for dataset_name in _sorted_uci_names:
            dval = None if only_names else fetch_UCIMulticlassDataset(dataset_name)
            yield dataset_name, dval


def gen_transformer_model_dataset(only_dataset_names=False, only_model_names=False):
    dataset_model = [
        ("imdb", "bert-base-uncased"),
    ]

    if only_dataset_names:
        return [d for d, _ in dataset_model]
    if only_model_names:
        return [m for _, m in dataset_model]

    for dataset_name, model_name in dataset_model:
        parent_dir = os.path.join(qc.env["OUT_DIR"], "trainsformers", "embeds", dataset_name, model_name)

        V_X = torch.load(os.path.join(parent_dir, "hidden_states.validation.pt")).numpy()
        V_logits = torch.load(os.path.join(parent_dir, "logits.validation.pt")).numpy()
        V_labels = torch.load(os.path.join(parent_dir, "labels.validation.pt")).numpy()
        U_X = torch.load(os.path.join(parent_dir, "hidden_states.test.pt")).numpy()
        U_logits = torch.load(os.path.join(parent_dir, "logits.test.pt")).numpy()
        U_labels = torch.load(os.path.join(parent_dir, "labels.test.pt")).numpy()

        V = LabelledCollection(V_X, V_labels, classes=np.unique(V_labels))
        U = LabelledCollection(U_X, U_labels, classes=np.unique(U_labels))

        model = BaseEstimatorAdapter(V_X, U_X, V_logits, U_logits)

        with open(os.path.join(parent_dir, "dataset_info.json")) as f:
            dataset_info = json.load(f)
            L_prev = np.array(dataset_info["L_prev"])

        yield (model_name, model), (dataset_name, (V, U), L_prev)


def gen_acc_measure():
    multiclass = env.PROBLEM == "multiclass"
    if _toggle["vanilla"]:
        yield "vanilla_accuracy", vanilla_acc
    if _toggle["f1"]:
        if env.PROBLEM == "binary":
            yield "f1", f1
        elif env.PROBLEM == "multiclass":
            yield "macro_f1", f1_macro
            # yield "micro_f1", f1_micro


def gen_baselines(acc_fn):
    yield "Naive", NaiveCAP(acc_fn)
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")
    yield "DS", DispersionScore(acc_fn)
    yield "COT", COT(acc_fn)
    yield "CBPE", CBPE(acc_fn)
    yield "NN", NuclearNorm(acc_fn)
    yield "Q-COT", Q_COT(acc_fn, kdey())
    # yield "QuAccNxN(KDEy)", QuAccNxN(acc_fn, kdey(), add_X=True, add_posteriors=True, add_maxinfsoft=True)
    # yield "QuAccNxN(KDEy-a)", QuAccNxN(acc_fn, kdey_auto(), add_X=True, add_posteriors=True, add_maxinfsoft=True)


def gen_baselines_vp(acc_fn, D):
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors)


# NOTE: the reason why mlp beats lr could be two-fold:
# (i) mlp uses a hidden layer of 100 ReLU which has proven to be more effective in practical applications than sigmoid function;
# (ii) while both mlp and lr use the same loss function (cross-entropy), lr corrects its predictions using a penalty function
# which is based on L2. This could affect negatively the performance of the KDEyML quantifier which uses the KL divergence as
# a loss function, which is strictly correlated to cross-entropy. The lr L2 regularization could possibly "ruin" the pure
# cross-entropy minimization towards which also KDEyML works.
def gen_CAP_cont_table(h, acc_fn):
    if _toggle["same_h"]:
        if _toggle["cc"]:
            yield "LEAP(CC)", LEAP(acc_fn, cc(), reuse_h=h, log_true_solve=True)
            yield "S-LEAP(CC)", PHD(acc_fn, cc(), reuse_h=h)
            yield "O-LEAP(CC)", OCE(acc_fn, cc(), reuse_h=h)
        if _toggle["acc"]:
            yield "LEAP(ACC)", LEAP(acc_fn, acc(), reuse_h=h, log_true_solve=True)
        yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True)
        yield "S-LEAP(KDEy)", PHD(acc_fn, kdey(), reuse_h=h)
        yield "O-LEAP(KDEy)", OCE(acc_fn, kdey(), reuse_h=h)

    if _toggle["mlp"]:
        if _toggle["cc"]:
            yield "LEAP(CC-MLP)", LEAP(acc_fn, cc(), log_true_solve=True)
            yield "S-LEAP(CC-MLP)", PHD(acc_fn, cc())
            yield "O-LEAP(CC-MLP)", OCE(acc_fn, cc())
        if _toggle["acc"]:
            yield "LEAP(ACC-MLP)", LEAP(acc_fn, acc(), log_true_solve=True)
        yield "LEAP(KDEy-MLP)", LEAP(acc_fn, kdey(), log_true_solve=True)
        yield "S-LEAP(KDEy-MLP)", PHD(acc_fn, kdey())
        yield "O-LEAP(KDEy-MLP)", OCE(acc_fn, kdey())

    if _toggle["slsqp"]:
        if _toggle["same_h"]:
            if _toggle["cc"]:
                yield (
                    "LEAP(CC)-SLSQP",
                    LEAP(acc_fn, cc(), reuse_h=h, log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
                )
                yield "O-LEAP(CC)-SLSQP", OCE(acc_fn, cc(), reuse_h=h, optim_method="SLSQP", sparse_matrix=False)
            if _toggle["acc"]:
                yield (
                    "LEAP(ACC)-SLSQP",
                    LEAP(acc_fn, acc(), reuse_h=h, log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
                )
            yield (
                "LEAP(KDEy)-SLSQP",
                LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
            )
            yield "O-LEAP(KDEy)-SLSQP", OCE(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP", sparse_matrix=False)

        if _toggle["mlp"]:
            if _toggle["cc"]:
                yield (
                    "LEAP(CC-MLP)-SLSQP",
                    LEAP(acc_fn, cc(), log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
                )
                yield "O-LEAP(CC-MLP)-SLSQP", OCE(acc_fn, cc(), optim_method="SLSQP", sparse_matrix=False)
            if _toggle["acc"]:
                yield (
                    "LEAP(ACC-MLP)-SLSQP",
                    LEAP(acc_fn, acc(), log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
                )
            yield (
                "LEAP(KDEy-MLP)-SLSQP",
                LEAP(acc_fn, kdey(), log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
            )
            yield "O-LEAP(KDEy-MLP)-SLSQP", OCE(acc_fn, kdey(), optim_method="SLSQP", sparse_matrix=False)


def gen_methods_with_oracle(h, acc_fn, D: DatasetBundle):
    if _toggle["oracle"]:
        oracle_q = OracleQuantifier([ui for ui in D.test_prot()])
        yield (
            "LEAP(oracle)",
            LEAP(acc_fn, oracle_q, reuse_h=h, log_true_solve=True),
        )
        yield "S-LEAP(oracle)", PHD(acc_fn, oracle_q, reuse_h=h)
        yield "O-LEAP(oracle)", OCE(acc_fn, oracle_q, reuse_h=h)
    else:
        return
        yield


def gen_methods(h, D):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_baselines(acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_baselines_vp(acc_fn, D):
        yield name, method, D.V1, D.V1_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_methods_with_oracle(h, acc_fn, D):
        yield name, method, D.V, D.V_posteriors


def get_classifier_names():
    return [name for name, _ in gen_classifiers()]


def get_tr_classifier_names():
    return gen_transformer_model_dataset(only_model_names=True)


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def get_tr_dataset_names():
    return gen_transformer_model_dataset(only_dataset_names=True)


def get_acc_names():
    return [acc_name for acc_name, _ in gen_acc_measure()]


def get_method_names(with_oracle=True):
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()

    baselines = [m for m, _ in gen_baselines(mock_acc_fn)] + [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_D)]
    CAP_ct = [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]

    names = baselines + CAP_ct

    if with_oracle:
        names += [m for m, _ in gen_methods_with_oracle(mock_h, mock_acc_fn, mock_D)]

    return names


def is_excluded(classifier, dataset, method, acc):
    # return (acc == "f1" or acc == "macro_f1") and method == "Q-COT"
    return False


def get_baseline_names():
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()
    baselines_names = [m for m, _ in gen_baselines(mock_acc_fn)]
    baselines_vp_names = [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_D)]
    return baselines_names[:2] + baselines_vp_names + baselines_names[2:]
