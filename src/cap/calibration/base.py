from abc import ABC, abstractmethod


class SimpleCalibratorFactory(ABC):
    @abstractmethod
    def __call__(self, valid_preacts, valid_labels): ...


class SourceTargetCalibratorFactory(ABC):
    @abstractmethod
    def __call__(self, Zsrc, ysrc, Ztgt): ...
