from typing import Any, List, Optional

import numpy as np
import math
import quapy as qp
import scipy.sparse as sp
from quapy.data import LabelledCollection


# Extended classes
#
# 0 ~ True 0
# 1 ~ False 1
# 2 ~ False 0
# 3 ~ True 1
#      _____________________
#     |          |          |
#     |  True 0  |  False 1 |
#     |__________|__________|
#     |          |          |
#     |  False 0 |  True 1  |
#     |__________|__________|
#
class ExClassManager:
    @staticmethod
    def get_ex(n_classes: int, true_class: int, pred_class: int) -> int:
        return true_class * n_classes + pred_class

    @staticmethod
    def get_pred(n_classes: int, ex_class: int) -> int:
        return ex_class % n_classes

    @staticmethod
    def get_true(n_classes: int, ex_class: int) -> int:
        return ex_class // n_classes


class ExtendedCollection(LabelledCollection):
    def __init__(
        self,
        instances: np.ndarray | sp.csr_matrix,
        labels: np.ndarray,
        classes: Optional[List] = None,
    ):
        super().__init__(instances, labels, classes=classes)

    def split_by_pred(self):
        _ncl = int(math.sqrt(self.n_classes))
        _indexes = ExtendedCollection.split_index_by_pred(_ncl, self.instances)
        return [
            ExtendedCollection(
                self.instances[ind] if len(ind) > 0 else np.asarray([], dtype=int),
                np.asarray(
                    [
                        ExClassManager.get_true(_ncl, lbl)
                        for lbl in (self.labels[ind] if len(ind) > 0 else [])
                    ],
                    dtype=int,
                ),
                classes=range(0, _ncl),
            )
            for ind in _indexes
        ]

    @classmethod
    def split_index_by_pred(
        cls, n_classes: int, instances: np.ndarray
    ) -> List[np.ndarray]:
        _pred_label = [np.argmax(inst[-n_classes:], axis=0) for inst in instances]
        return [
            np.asarray([j for (j, x) in enumerate(_pred_label) if x == i])
            for i in range(0, n_classes)
        ]

    @classmethod
    def extend_instances(
        cls, instances: np.ndarray, pred_proba: np.ndarray
    ) -> np.ndarray:
        if isinstance(instances, sp.csr_matrix):
            _pred_proba = sp.csr_matrix(pred_proba)
            n_x = sp.hstack([instances, _pred_proba])
        elif isinstance(instances, np.ndarray):
            n_x = np.concatenate((instances, pred_proba), axis=1)
        else:
            raise ValueError("Unsupported matrix format")

        return n_x

    @classmethod
    def extend_collection(cls, base: LabelledCollection, pred_proba: np.ndarray) -> Any:
        n_classes = base.n_classes

        # n_X = [ X | predicted probs. ]
        n_x = cls.extend_instances(base.X, pred_proba)

        # n_y = (exptected y, predicted y)
        pred = np.asarray([prob.argmax(axis=0) for prob in pred_proba])
        n_y = np.asarray(
            [
                ExClassManager.get_ex(n_classes, true_class, pred_class)
                for (true_class, pred_class) in zip(base.y, pred)
            ]
        )

        return ExtendedCollection(n_x, n_y, classes=[*range(0, n_classes * n_classes)])


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
