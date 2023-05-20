import numpy as np
import scipy.sparse as sp
from quapy.data import LabelledCollection
from typing import List, Optional


class ExtendedCollection(LabelledCollection):
    def __init__(
        self,
        instances: np.ndarray | sp.csr_matrix,
        labels: np.ndarray,
        classes: Optional[List] = None,
    ):
        super().__init__(instances, labels, classes=classes)
