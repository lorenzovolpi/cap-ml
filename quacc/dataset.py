from typing import Tuple
import numpy as np
from quapy.data.base import LabelledCollection
import quapy as qp
from sklearn.conftest import fetch_rcv1

TRAIN_VAL_PROP = 0.5


def get_imdb() -> Tuple[LabelledCollection]:
    train, test = qp.datasets.fetch_reviews("imdb", tfidf=True).train_test
    train, validation = train.split_stratified(train_prop=TRAIN_VAL_PROP)
    return train, validation, test


def get_spambase():
    train, test = qp.datasets.fetch_UCIDataset("spambase", verbose=False).train_test
    train, validation = train.split_stratified(train_prop=TRAIN_VAL_PROP)
    return train, validation, test


def get_rcv1(sample_size=100):
    dataset = fetch_rcv1()

    target_labels = [
        (target, dataset.target[:, ind].toarray().flatten())
        for (ind, target) in enumerate(dataset.target_names)
    ]
    filtered_target_labels = filter(
        lambda _, labels: np.sum(labels) >= sample_size, target_labels
    )
    return {
        target: LabelledCollection(dataset.data, labels, classes=[0, 1])
        for (target, labels) in filtered_target_labels
    }
