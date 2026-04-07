import hashlib
import os
import pickle
from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Self, Tuple

import numpy as np
import quapy as qp
from numba import njit
from pretrain.dataset import load_dataset
from quapy.data import LabelledCollection
from quapy.protocol import UPP
from sklearn.base import BaseEstimator, ClassifierMixin

from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.utils.commons import contingency_table
from cap_exp.util import split_validation

# FIX: REMOVE FROM HERE!!
BASEDIR = os.path.join("output", "tms", "pretrain")


class NotPretrainedError(Exception):
    pass


@njit
def _lookup_sample(X, S, order, left, right):
    res = []
    for j in range(S.shape[0]):
        row = S[j]
        cand = order[left[j] : right[j]]
        for i in cand:
            ok = True
            for k in range(X.shape[1]):
                if X[i, k] != row[k]:
                    ok = False
                    break
            if ok:
                res.append(i)
                break
    return res


class PreTrainedClassifier(BaseEstimator, ClassifierMixin):
    GOLDEN_RATIO_MULTIPLIER = 11400714819323198485
    MULTIPLIER_64 = 0x9E3779B97F4A7C15

    def __init__(
        self,
        U_X: np.ndarray,
        U_posteriors: np.ndarray,
        V_X: np.ndarray,
        V_posteriors: np.ndarray,
    ):
        self.X = np.ascontiguousarray(np.vstack([U_X, V_X]))
        self.P = np.vstack([U_posteriors, V_posteriors])
        self.hx_order, self.hx_sorted = self._build_hx(self.X)
        self._fitted = True
        self.classes_ = np.unique(np.argmax(self.P, axis=-1))

    def __getattribute__(self, name: str, /) -> Any:
        if name in ["U_X", "U_posteriors", "V_X", "V_posteriors"]:
            return False
        else:
            return super().__getattribute__(name)

    def _fp64_rows(self, X):
        row_bytes = X.dtype.itemsize * X.shape[1]
        if row_bytes % 8 != 0:
            # padding a multipli di 8 byte
            pad = 8 - (row_bytes % 8)
            Xb = np.ascontiguousarray(X.view(np.uint8))
            Xb = np.pad(Xb, ((0, 0), (0, pad)), mode="constant")
            U = Xb.view(np.uint64)
        else:
            U = X.view(np.uint64)

        U = U.reshape(X.shape[0], -1)

        # mixing semplice e veloce
        h = np.bitwise_xor.reduce(U * np.uint64(self.GOLDEN_RATIO_MULTIPLIER), axis=1)
        h ^= np.uint64(self.MULTIPLIER_64)
        return h

    def _build_hx(self, X: np.ndarray):
        hx = self._fp64_rows(X)
        hx_order = np.argsort(hx)
        hx_sorted = hx[hx_order]
        return hx_order, hx_sorted

    def fit(self, X, y):
        return self

    def predict_proba(self, S: np.ndarray) -> np.ndarray:
        Sc = np.ascontiguousarray(S)
        hs = self._fp64_rows(Sc)
        left = np.searchsorted(self.hx_sorted, hs, side="left")
        right = np.searchsorted(self.hx_sorted, hs, side="right")
        idx_list = _lookup_sample(self.X, Sc, self.hx_order, left, right)
        return self.P[idx_list, :]

    def predict(self, X: np.ndarray) -> np.ndarray:
        posteriors = self.predict_proba(X)
        return posteriors.argmax(axis=-1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.classes_.shape[0] == 2:
            return self.predict_proba(X)[:, 1].squeeze()
        else:
            return self.predict_proba(X)


@dataclass
class ClassifierInfo:
    class_name: str
    params: dict
    default: bool = False
    ms_ignore: bool = False

    @classmethod
    def _hash_params(cls, params: dict):
        params_str = cls._get_params_string(params)
        return hashlib.sha256(params_str.encode()).hexdigest()[:64]

    @classmethod
    def _get_params_string(cls, params: dict):
        params_str = "[" + ";".join([f"{k}={v}" for k, v in params.items()]) + "]"
        return params_str

    @classmethod
    def _get_name(cls, default: bool, class_name: str, params: dict):
        if default:
            return class_name
        params_str = cls._get_params_string(params)
        name = f"{class_name}_{params_str}"
        return name

    @classmethod
    def _get_full_name(cls, default: bool, class_name: str, params: dict):
        if default:
            return class_name
        full_name = f"{class_name}_{cls._hash_params(params)}"
        return full_name

    @property
    def name(self):
        return self._get_name(self.default, self.class_name, self.params)

    @property
    def full_name(self):
        return self._get_full_name(self.default, self.class_name, self.params)


@dataclass
class DatasetInfo:
    name: str
    collection: str
    n_classes: int


class DatasetBundle:
    def __init__(self, L_prevalence: np.ndarray, V: LabelledCollection, U: LabelledCollection):
        self.L_prevalence = L_prevalence
        self.V = V
        self.U = U

    def with_test_data(self, repeats: int) -> Self:
        self.test_prot = UPP(
            self.U,
            repeats=repeats,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )
        self.V1, self.V2_prot = split_validation(self.V, random_state=qp.environ["_R_SEED"])

        return self

    def with_posteriors(self, h: PreTrainedClassifier) -> Self:
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
            self.test_prot_true_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))

        return self


@dataclass
class PretrainInfo:
    domain: str
    d_info: DatasetInfo
    h_info: ClassifierInfo

    def _get_stem(self):
        return f"{self.h_info.full_name}_{self.d_info.name}_{self.d_info.n_classes}"

    @property
    def info_path(self):
        stem = self._get_stem()
        return os.path.join(BASEDIR, self.domain, f"{stem}_info.pkl")

    @property
    def posteriors_path(self):
        stem = self._get_stem()
        return os.path.join(BASEDIR, self.domain, f"{stem}_post.npz")

    @property
    def exists(self) -> bool:
        return os.path.exists(self.info_path)

    def dump(
        self,
        posteriors: Dict[str, np.ndarray] | None = None,
        logits: Dict[str, np.ndarray] | None = None,
    ):
        os.makedirs(os.path.dirname(self.info_path), exist_ok=True)
        with open(self.info_path, "wb") as f:
            pickle.dump(self, f)
        _posts = {}
        if posteriors:
            _posts["V_posteriors"] = posteriors["V"]
            _posts["U_posteriors"] = posteriors["U"]
        if logits:
            _posts["V_logits"] = logits["V"]
            _posts["U_logits"] = logits["U"]
        np.savez_compressed(self.posteriors_path, **_posts)

    def load_dataset_bundle(self) -> DatasetBundle:
        L_prevalence, V, U = load_from_collection(self)
        d_bundle = DatasetBundle(L_prevalence, V, U)
        return d_bundle

    def load_logits(self, npz=None):
        if npz is None:
            npz = np.load(self.posteriors_path)

        if "V_logits" not in npz or "U_logits" not in npz:
            raise NotPretrainedError(f"No logits found in {self.posteriors_path}")

        return npz["V_logits"], npz["U_logits"]

    def load_posteriors(self, npz=None):
        if npz is None:
            npz = np.load(self.posteriors_path)

        if "V_posteriors" not in npz or "U_posteriors" not in npz:
            raise NotPretrainedError(f"No posteriors found in {self.posteriors_path}")

        return npz["V_posteriors"], npz["U_posteriors"]

    def load_pretrained_classifier(self, d_bundle: DatasetBundle) -> PreTrainedClassifier:
        post_path = self.posteriors_path
        _npz = np.load(post_path)
        V_posteriors, U_posteriors = self.load_posteriors(npz=_npz)
        return PreTrainedClassifier(
            U_X=d_bundle.U.X, U_posteriors=U_posteriors, V_X=d_bundle.V.X, V_posteriors=V_posteriors
        )

    @classmethod
    def load(cls, path: str, fast: bool = False) -> Tuple[DatasetBundle, PreTrainedClassifier, Self] | Self:
        with open(path, "rb") as f:
            p_info = pickle.load(f)
        if fast:
            return p_info

        d_bundle = p_info.load_dataset_bundle()

        h = p_info.load_pretrained_classifier(d_bundle)

        return d_bundle, h, p_info


def load_from_collection(p_info: PretrainInfo):
    dataset_collection = p_info.d_info.collection
    dataset_name = p_info.d_info.name
    if dataset_collection == "uci_binary":
        L, V, U = fetch_UCIBinaryDataset(dataset_name)
        return L.prevalence(), V, U
    elif dataset_collection == "uci_multiclass":
        L, V, U = fetch_UCIMulticlassDataset(dataset_name)
        return L.prevalence(), V, U
    elif dataset_collection in ["text", "image"]:
        return load_dataset(dataset_collection, dataset_name, p_info.h_info.full_name)
    else:
        raise ValueError(f"Unknown dataset collection: {dataset_collection}")


def load_info_paths(basedir: str, domain: str):
    _dir = os.path.join(basedir, domain)
    paths = glob(os.path.join(_dir, "*_info.pkl"))

    return paths


@dataclass
class ClassifierDatasetBundle:
    dataset_name: str
    dataset_collection: str
    n_classes: int
    h_class_name: str
    h_params: dict
    h_default: bool
    h_ms_ignore: bool

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> Self:
        with open(path, "rb") as f:
            return pickle.load(f)
