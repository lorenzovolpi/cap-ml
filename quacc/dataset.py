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


def get_spambase() -> Tuple[LabelledCollection]:
    train, test = qp.datasets.fetch_UCIDataset("spambase", verbose=False).train_test
    train, validation = train.split_stratified(train_prop=TRAIN_VAL_PROP)
    return train, validation, test


def get_rcv1(sample_size=100):
    n_train = 23149
    dataset = fetch_rcv1()

    def dataset_split(data, labels, classes=[0, 1]) -> Tuple[LabelledCollection]:
        all_train_d, test_d = data[:n_train, :], data[n_train:, :]
        all_train_l, test_l = labels[:n_train], labels[n_train:]
        all_train = LabelledCollection(all_train_d, all_train_l, classes=classes)
        test = LabelledCollection(test_d, test_l, classes=classes)
        train, validation = all_train.split_stratified(train_prop=TRAIN_VAL_PROP)
        return train, validation, test

    target_labels = [
        (target, dataset.target[:, ind].toarray().flatten())
        for (ind, target) in enumerate(dataset.target_names)
    ]
    filtered_target_labels = filter(
        lambda _, labels: np.sum(labels[n_train:]) >= sample_size, target_labels
    )
    return {
        target: dataset_split(dataset.data, labels, classes=[0, 1])
        for (target, labels) in filtered_target_labels
    }
