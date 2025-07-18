import itertools as IT
from abc import ABC, abstractmethod
from typing import List

from quapy.data.base import LabelledCollection
from quapy.protocol import AbstractProtocol
from sklearn.base import BaseEstimator


class ClassifierAccuracyPrediction(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, val: LabelledCollection, posteriors):
        """
        Trains a CAP method.

        :param val: training data
        :return: self
        """
        ...

    @abstractmethod
    def predict(self, X, posteriors, oracle_prev=None) -> float:
        """
        Predicts directly the accuracy using the accuracy function

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: float
        """
        ...

    def batch_predict(self, prot: AbstractProtocol, posteriors) -> list[float]:
        estim_accs = [self.predict(Ui.X, posteriors=P) for Ui, P in IT.zip_longest(prot(), posteriors)]
        return estim_accs


CAP = ClassifierAccuracyPrediction
