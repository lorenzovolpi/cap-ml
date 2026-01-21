import itertools as IT
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator, clone

import exp.direct_ours.env as env
from cap.utils.commons import contingency_table, true_acc_from_posteriors
from exp.direct_ours.util import split_validation


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
            repeats=env.NUM_TEST,
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
