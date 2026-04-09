import os
from dataclasses import dataclass
from traceback import print_exception
from typing import Iterable, Literal, Tuple

import numpy as np
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import cap
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.utils.commons import parallel
from cap_exp.pretrain.data import ClassifierInfo, DatasetInfo, PretrainInfo
from cap_exp.pretrain.dataset import sort_datasets_by_size

EXPERIMENT = "pretrain"
DOMAIN = "classic"

NUM_TESTS = 1000
BATCH_SIZE = 8

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"


class TrainResult:
    def __init__(
        self,
        p_info: PretrainInfo,
        V_posteriors: np.ndarray,
        U_posteriors: np.ndarray,
        status: Literal["ok", "old"],
    ):
        self.p_info = p_info
        self.V_posteriors = V_posteriors
        self.U_posteriors = U_posteriors
        self.status = status

    @classmethod
    def ok(cls, p_info: PretrainInfo, V_posteriors: np.ndarray, U_posteriors: np.ndarray):
        return TrainResult(p_info, V_posteriors, U_posteriors, "ok")

    @classmethod
    def old(cls, p_info: PretrainInfo):
        return TrainResult(p_info, None, None, "old")

    @property
    def is_ok(self):
        return self.status == "ok"

    @property
    def is_old(self):
        return self.status == "old"


@dataclass()
class Posteriors:
    V_posteriors: np.ndarray
    U_posteriors: np.ndarray

    def asdict(self):
        return dict(
            V=self.V_posteriors,
            U=self.U_posteriors,
        )


def gen_classifiers() -> Iterable[Tuple[BaseEstimator, ClassifierInfo]]:
    yield LogisticRegression(), ClassifierInfo(class_name="LR", params={}, default=True)
    yield KNeighborsClassifier(), ClassifierInfo(class_name="kNN", params={}, default=True)
    svm_params = dict(kernel="rbf", probability=True)
    yield SVC(**svm_params), ClassifierInfo(class_name="SVM", params=svm_params, default=True)
    yield MLPClassifier(), ClassifierInfo(class_name="MLP", params={}, default=True)


def gen_datasets(
    only_names=False,
) -> Iterable[tuple[str, tuple[LabelledCollection, LabelledCollection, LabelledCollection] | None]]:
    _uci_bin_native = [
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
        # "yeast",
    ]
    _uci_bin_names = [d for d in UCI_BINARY_DATASETS if d in _uci_bin_native]
    coll = "uci_binary"
    _sorted_bin_names = sort_datasets_by_size(coll, _uci_bin_names, fetch_UCIBinaryDataset)
    for dn in _sorted_bin_names[:5]:
        dval = None if only_names else fetch_UCIBinaryDataset(dn)
        yield dn, coll, dval
    _uci_mul_names = [d for d in UCI_MULTICLASS_DATASETS]
    coll = "uci_multiclass"
    _sorted_mul_names = sort_datasets_by_size(coll, _uci_mul_names, fetch_UCIMulticlassDataset)
    for dn in _sorted_mul_names:
        dval = None if only_names else fetch_UCIMulticlassDataset(dn)
        yield dn, coll, dval


def train_variants(
    args: Tuple[
        str,
        str,
        LabelledCollection,
        LabelledCollection,
        LabelledCollection,
        Iterable[Tuple[BaseEstimator, ClassifierInfo]],
    ],
) -> list[Tuple[PretrainInfo, Posteriors]]:
    dataset_name, dataset_coll, L, V, U, h_batch = args
    n_classes = L.n_classes
    d_info = DatasetInfo(dataset_name, dataset_coll, n_classes)

    results = []
    for h, h_info in h_batch:
        p_info = PretrainInfo(DOMAIN, d_info, h_info)

        h.fit(*L.Xy)
        V_posteriors = h.predict_proba(V.X)
        U_posteriors = h.predict_proba(U.X)
        results.append((p_info, Posteriors(V_posteriors, U_posteriors)))

    return results


def pretrain():
    datasets_classifiers = []
    for dataset in gen_datasets():
        dataset_name, dataset_coll, (L, V, U) = dataset
        n_classes = L.n_classes
        d_info = DatasetInfo(dataset_name, dataset_coll, n_classes)
        i = 0
        clsf_batches = []
        for clsf in gen_classifiers(n_classes):
            _, h_info = clsf
            if PretrainInfo(DOMAIN, d_info, h_info).exists:
                print(f"Already exists: {h_info.name} on {d_info.name}, skipping.")
                continue
            if i % 8 == 0:
                clsf_batches.append([])
            clsf_batches[-1].append(clsf)
        for batch in clsf_batches:
            datasets_classifiers.append((dataset_name, dataset_coll, L, V, U, batch))

    results_gen: Iterable[list[Tuple[PretrainInfo, Posteriors]]] = parallel(
        func=train_variants,
        args_list=datasets_classifiers,
        n_jobs=cap.env["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    for results in results_gen:
        for p_info, post in results:
            print(f"Pretrained {p_info.h_info.name} on {p_info.d_info.name}.")
            p_info.dump(posteriors=post.asdict())


def main():
    try:
        pretrain()
    except Exception as e:
        print_exception(e)
