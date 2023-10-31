import math
from typing import List

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from sklearn.conftest import fetch_rcv1

TRAIN_VAL_PROP = 0.5


class DatasetSample:
    def __init__(
        self,
        train: LabelledCollection,
        validation: LabelledCollection,
        test: LabelledCollection,
    ):
        self.train = train
        self.validation = validation
        self.test = test

    @property
    def train_prev(self):
        return self.train.prevalence()

    @property
    def validation_prev(self):
        return self.validation.prevalence()

    @property
    def prevs(self):
        return {"train": self.train_prev, "validation": self.validation_prev}


class Dataset:
    def __init__(self, name, n_prevalences=9, prevs=None, target=None):
        self._name = name
        self._target = target

        self.prevs = None
        self.n_prevs = n_prevalences
        if prevs is not None:
            prevs = np.unique([p for p in prevs if p > 0.0 and p < 1.0])
            if prevs.shape[0] > 0:
                self.prevs = np.sort(prevs)
                self.n_prevs = self.prevs.shape[0]

    def __spambase(self):
        return qp.datasets.fetch_UCIDataset("spambase", verbose=False).train_test

    # provare min_df=5
    def __imdb(self):
        return qp.datasets.fetch_reviews("imdb", tfidf=True, min_df=3).train_test

    def __rcv1(self):
        n_train = 23149
        available_targets = ["CCAT", "GCAT", "MCAT"]

        if self._target is None or self._target not in available_targets:
            raise ValueError(f"Invalid target {self._target}")

        dataset = fetch_rcv1()
        target_index = np.where(dataset.target_names == self._target)[0]
        all_train_d = dataset.data[:n_train, :]
        test_d = dataset.data[n_train:, :]
        labels = dataset.target[:, target_index].toarray().flatten()
        all_train_l, test_l = labels[:n_train], labels[n_train:]
        all_train = LabelledCollection(all_train_d, all_train_l, classes=[0, 1])
        test = LabelledCollection(test_d, test_l, classes=[0, 1])

        return all_train, test

    def get_raw(self, validation=True) -> DatasetSample:
        all_train, test = {
            "spambase": self.__spambase,
            "imdb": self.__imdb,
            "rcv1": self.__rcv1,
        }[self._name]()

        train, val = all_train, None
        if validation:
            train, val = all_train.split_stratified(
                train_prop=TRAIN_VAL_PROP, random_state=0
            )

        return DatasetSample(train, val, test)

    def get(self) -> List[DatasetSample]:
        (all_train, test) = {
            "spambase": self.__spambase,
            "imdb": self.__imdb,
            "rcv1": self.__rcv1,
        }[self._name]()

        # resample all_train set to have (0.5, 0.5) prevalence
        at_positives = np.sum(all_train.y)
        all_train = all_train.sampling(
            min(at_positives, len(all_train) - at_positives) * 2, 0.5, random_state=0
        )

        # sample prevalences
        if self.prevs is not None:
            prevs = self.prevs
        else:
            prevs = np.linspace(0.0, 1.0, num=self.n_prevs + 1, endpoint=False)[1:]

        at_size = min(math.floor(len(all_train) * 0.5 / p) for p in prevs)
        datasets = []
        for p in 1.0 - prevs:
            all_train_sampled = all_train.sampling(at_size, p, random_state=0)
            train, validation = all_train_sampled.split_stratified(
                train_prop=TRAIN_VAL_PROP, random_state=0
            )
            datasets.append(DatasetSample(train, validation, test))

        return datasets

    def __call__(self):
        return self.get()

    @property
    def name(self):
        return (
            f"{self._name}_{self._target}_{self.n_prevs}prevs"
            if self._name == "rcv1"
            else f"{self._name}_{self.n_prevs}prevs"
        )


# >>> fetch_rcv1().target_names
# array(['C11', 'C12', 'C13', 'C14', 'C15', 'C151', 'C1511', 'C152', 'C16',
#        'C17', 'C171', 'C172', 'C173', 'C174', 'C18', 'C181', 'C182',
#        'C183', 'C21', 'C22', 'C23', 'C24', 'C31', 'C311', 'C312', 'C313',
#        'C32', 'C33', 'C331', 'C34', 'C41', 'C411', 'C42', 'CCAT', 'E11',
#        'E12', 'E121', 'E13', 'E131', 'E132', 'E14', 'E141', 'E142',
#        'E143', 'E21', 'E211', 'E212', 'E31', 'E311', 'E312', 'E313',
#        'E41', 'E411', 'E51', 'E511', 'E512', 'E513', 'E61', 'E71', 'ECAT',
#        'G15', 'G151', 'G152', 'G153', 'G154', 'G155', 'G156', 'G157',
#        'G158', 'G159', 'GCAT', 'GCRIM', 'GDEF', 'GDIP', 'GDIS', 'GENT',
#        'GENV', 'GFAS', 'GHEA', 'GJOB', 'GMIL', 'GOBIT', 'GODD', 'GPOL',
#        'GPRO', 'GREL', 'GSCI', 'GSPO', 'GTOUR', 'GVIO', 'GVOTE', 'GWEA',
#        'GWELF', 'M11', 'M12', 'M13', 'M131', 'M132', 'M14', 'M141',
#        'M142', 'M143', 'MCAT'], dtype=object)


def rcv1_info():
    dataset = fetch_rcv1()
    n_train = 23149

    targets = []
    for target in range(103):
        train_t_prev = np.average(dataset.target[:n_train, target].toarray().flatten())
        test_t_prev = np.average(dataset.target[n_train:, target].toarray().flatten())
        targets.append(
            (
                dataset.target_names[target],
                {
                    "train": (1.0 - train_t_prev, train_t_prev),
                    "test": (1.0 - test_t_prev, test_t_prev),
                },
            )
        )

    targets.sort(key=lambda t: t[1]["train"][1])
    for n, d in targets:
        print(f"{n}:")
        for k, (fp, tp) in d.items():
            print(f"\t{k}: {fp:.4f}, {tp:.4f}")


if __name__ == "__main__":
    rcv1_info()
