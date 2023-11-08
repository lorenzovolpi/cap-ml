from typing import Callable, Union

import numpy as np
from quapy.protocol import AbstractProtocol, OnLabelledCollectionProtocol

import quacc as qc

from ..method.base import BaseAccuracyEstimator


def evaluate(
    estimator: BaseAccuracyEstimator,
    protocol: AbstractProtocol,
    error_metric: Union[Callable | str],
) -> float:
    if isinstance(error_metric, str):
        error_metric = qc.error.from_name(error_metric)

    collator_bck_ = protocol.collator
    protocol.collator = OnLabelledCollectionProtocol.get_collator("labelled_collection")

    estim_prevs, true_prevs = [], []
    for sample in protocol():
        e_sample = estimator.extend(sample)
        estim_prev = estimator.estimate(e_sample.X, ext=True)
        estim_prevs.append(estim_prev)
        true_prevs.append(e_sample.prevalence())

    protocol.collator = collator_bck_

    true_prevs = np.array(true_prevs)
    estim_prevs = np.array(estim_prevs)

    return error_metric(true_prevs, estim_prevs)
