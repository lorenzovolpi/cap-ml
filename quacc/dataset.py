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

def get_rcv1(target:str):
    sample_size = qp.environ["SAMPLE_SIZE"]
    n_train = 23149
    dataset = fetch_rcv1()

    if target not in dataset.target_names:
        raise ValueError("Invalid target")

    def dataset_split(data, labels, classes=[0, 1]) -> Tuple[LabelledCollection]:
        all_train_d, test_d = data[:n_train, :], data[n_train:, :]
        all_train_l, test_l = labels[:n_train], labels[n_train:]
        all_train = LabelledCollection(all_train_d, all_train_l, classes=classes)
        test = LabelledCollection(test_d, test_l, classes=classes)
        train, validation = all_train.split_stratified(train_prop=TRAIN_VAL_PROP)
        return train, validation, test

    target_index = np.where(dataset.target_names == target)[0]
    target_labels = dataset.target[:, target_index].toarray().flatten()

    if np.sum(target_labels[n_train:]) < sample_size:
        raise ValueError("Target has too few positive samples")

    d = dataset_split(dataset.data, target_labels, classes=[0, 1])

    return d

