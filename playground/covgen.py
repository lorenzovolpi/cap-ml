import itertools as IT
from collections.abc import Iterator, Sequence
from time import time
from typing import Any, Callable, Literal

import numpy as np
import quapy as qp
import quapy.functional as F
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.protocol import UPP, AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.mixture import BayesianGaussianMixture

from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset

seed = 42
qp.environ["_R_SEED"] = seed


class generator(Sequence):
    def __init__(self, names: list[str], fn: Callable[[str], Any]) -> None:
        self.names = names
        self.fn = fn

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> str:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return generator(self.names[start:stop:step], self.fn)

        n = self.names[index]
        return n, self.fn(n)


def h_fetch(n: str):
    if n == "LR":
        return LogisticRegression()


def d_fetch(n: str):
    if n in UCI_BINARY_DATASETS:
        return fetch_UCIBinaryDataset(n)
    elif n in UCI_MULTICLASS_DATASETS:
        return fetch_UCIMulticlassDataset(n)


class BGMP(AbstractStochasticSeededProtocol, OnLabelledCollectionProtocol):
    def __init__(
        self,
        data: LabelledCollection,
        sample_size: int = None,
        repeats: int = 100,
        random_state: int = 0,
        return_type: Literal["sample_prev", "index", "labelled_collection"] = "sample_prev",
    ) -> None:
        super(BGMP, self).__init__(random_state)
        self.data = data
        self.sample_size = qp._get_sample_size(sample_size)
        self.repeats = repeats
        self.random_state = random_state
        self.collator = self.get_collator(return_type)
        bgm = BayesianGaussianMixture(
            n_components=64,
            weight_concentration_prior=1e-2,
            random_state=random_state,
        )
        bgm.fit(data.X)
        self.pi_original = bgm.weights_
        self.gamma = bgm.predict_proba(data.X)

    def samples_parameters(self):
        indexes = []
        for prevs in F.uniform_simplex_sampling(n_classes=self.pi_original.shape[0], size=self.repeats):
            ratios = prevs / self.pi_original
            weights_i = np.dot(self.gamma, ratios)
            sampling_probs = weights_i / np.sum(weights_i)
            index = np.random.choice(len(self.data), size=self.sample_size, replace=True, p=sampling_probs)
            indexes.append(index)
        return indexes

    def sample(self, index):
        return self.data.sampling_from_index(index)

    def total(self):
        return self.repeats


def main():
    datasets = generator(["spambase", "semeion", "isolet", "poker_hand", "molecular", "connect-4"], d_fetch)
    models = generator(["LR"], h_fetch)
    for d, h in IT.product(datasets, models):
        h_name, h = h
        d_name, (L, V, U) = d

        h.fit(*L.Xy)

        tinit = time()
        prot_cv = BGMP(
            V,
            sample_size=1000,
            repeats=10,
            random_state=seed,
            return_type="labelled_collection",
        )
        prot_lb = UPP(
            V,
            sample_size=1000,
            repeats=10,
            random_state=seed,
            return_type="labelled_collection",
        )
        tbgm = time() - tinit

        vp = h.predict_proba(V.X)
        vacc = accuracy_score(V.y, np.argmax(vp, axis=-1))
        print(f"{d_name} {h_name}:\n\tlen=[{len(L)}; {len(V)}; {len(U)}]\n\tt_time={tbgm}\n\taccs=\n\ttot: {vacc}")
        for i, Vi in enumerate(prot_cv()):
            vip = h.predict_proba(Vi.X)
            y_hat = np.argmax(vip, axis=-1)
            print(f"\t{i + 1}: {accuracy_score(Vi.y, y_hat)}")
        print("-" * 20)
        for i, Vi in enumerate(prot_lb()):
            vip = h.predict_proba(Vi.X)
            y_hat = np.argmax(vip, axis=-1)
            print(f"\t{i + 1}: {accuracy_score(Vi.y, y_hat)}")

        print()


if __name__ == "__main__":
    main()
