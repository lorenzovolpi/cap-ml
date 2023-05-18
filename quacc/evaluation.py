from quapy.method.base import BaseQuantifier
from quapy.protocol import OnLabelledCollectionProtocol, AbstractStochasticSeededProtocol

from .estimator import AccuracyEstimator, _extend_collection


def estimate(estimator: AccuracyEstimator, protocol: AbstractStochasticSeededProtocol):

    # ensure that the protocol returns a LabelledCollection for each iteration
    protocol.collator = OnLabelledCollectionProtocol.get_collator('labelled_collection')

    base_prevs, true_prevs, estim_prevs = [], [], []
    for sample in protocol():
       e_sample = estimator.extend(sample)
       estim_prev = estimator.estimate(e_sample.X, ext=True)
       base_prevs.append(sample.prevalence())
       true_prevs.append(e_sample.prevalence())
       estim_prevs.append(estim_prev)

    return base_prevs, true_prevs, estim_prevs
       

def evaluate():
    pass
