from typing import List, Optional

import numpy as np
import quapy as qp
import scipy.sparse as sp
from quapy.data import LabelledCollection


class ExtendedCollection(LabelledCollection):
    def __init__(
        self,
        instances: np.ndarray | sp.csr_matrix,
        labels: np.ndarray,
        classes: Optional[List] = None,
    ):
        super().__init__(instances, labels, classes=classes)
        
def get_dataset(name):
    datasets = {
        "spambase": lambda: qp.datasets.fetch_UCIDataset(
            "spambase", verbose=False
        ).train_test,
        "hp": lambda: qp.datasets.fetch_reviews("hp", tfidf=True).train_test,
        "imdb": lambda: qp.datasets.fetch_reviews("imdb", tfidf=True).train_test,
    }

    try:
        return datasets[name]()
    except KeyError:
        raise KeyError(f"{name} is not available as a dataset")
