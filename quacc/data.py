import math
from typing import List, Optional

import numpy as np
import scipy.sparse as sp
from quapy.data import LabelledCollection
from sklearn.base import BaseEstimator


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
        _indexes = ExtendedCollection._split_index_by_pred(_ncl, self.instances)
        if isinstance(self.instances, np.ndarray):
            _instances = [
                self.instances[ind] if ind.shape[0] > 0 else np.asarray([], dtype=int)
                for ind in _indexes
            ]
        elif isinstance(self.instances, sp.csr_matrix):
            _instances = [
                self.instances[ind]
                if ind.shape[0] > 0
                else sp.csr_matrix(np.empty((0, 0), dtype=int))
                for ind in _indexes
            ]
        _labels = [
            np.asarray(
                [
                    ExClassManager.get_true(_ncl, lbl)
                    for lbl in (self.labels[ind] if len(ind) > 0 else [])
                ],
                dtype=int,
            )
            for ind in _indexes
        ]
        return [
            ExtendedCollection(inst, lbl, classes=range(0, _ncl))
            for (inst, lbl) in zip(_instances, _labels)
        ]

    @classmethod
    def split_inst_by_pred(
        cls, n_classes: int, instances: np.ndarray | sp.csr_matrix
    ) -> (List[np.ndarray | sp.csr_matrix], List[float]):
        _indexes = cls._split_index_by_pred(n_classes, instances)
        if isinstance(instances, np.ndarray):
            _instances = [
                instances[ind] if ind.shape[0] > 0 else np.asarray([], dtype=int)
                for ind in _indexes
            ]
        elif isinstance(instances, sp.csr_matrix):
            _instances = [
                instances[ind]
                if ind.shape[0] > 0
                else sp.csr_matrix(np.empty((0, 0), dtype=int))
                for ind in _indexes
            ]
        norms = [inst.shape[0] / instances.shape[0] for inst in _instances]
        return _instances, norms

    @classmethod
    def _split_index_by_pred(
        cls, n_classes: int, instances: np.ndarray | sp.csr_matrix
    ) -> List[np.ndarray]:
        if isinstance(instances, np.ndarray):
            _pred_label = [np.argmax(inst[-n_classes:], axis=0) for inst in instances]
        elif isinstance(instances, sp.csr_matrix):
            _pred_label = [
                np.argmax(inst[:, -n_classes:].toarray().flatten(), axis=0)
                for inst in instances
            ]
        else:
            raise ValueError("Unsupported matrix format")

        return [
            np.asarray([j for (j, x) in enumerate(_pred_label) if x == i], dtype=int)
            for i in range(0, n_classes)
        ]

    @classmethod
    def extend_instances(
        cls, instances: np.ndarray | sp.csr_matrix, pred_proba: np.ndarray
    ) -> np.ndarray | sp.csr_matrix:
        if isinstance(instances, sp.csr_matrix):
            _pred_proba = sp.csr_matrix(pred_proba)
            n_x = sp.hstack([instances, _pred_proba])
        elif isinstance(instances, np.ndarray):
            n_x = np.concatenate((instances, pred_proba), axis=1)
        else:
            raise ValueError("Unsupported matrix format")

        return n_x

    @classmethod
    def extend_collection(
        cls,
        base: LabelledCollection,
        classifier: BaseEstimator = None,
        pred_proba: np.ndarray = None,
    ):
        if classifier is None and pred_proba is None:
            raise AttributeError("classifier and pred_proba cannot be both None")

        if classifier is not None and pred_proba is not None:
            raise AttributeError(
                "Not needed parameters: just one of classifier or pred_proba is needed"
            )

        if classifier:
            pred_proba = classifier.predict_proba(base.X)

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
