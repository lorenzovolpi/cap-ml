import itertools as IT
from typing import Callable, Iterable, Literal

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.protocol import AbstractProtocol, AbstractStochasticSeededProtocol
from sklearn.metrics import confusion_matrix

import cap
import cap.models.utils as utils
from cap.models.direct import CAPDirect
from cap.utils.commons import contingency_table


class PrediQuant(CAPDirect):
    def __init__(
        self,
        acc: Callable,
        quantifier: AggregativeQuantifier,
        protocol: AbstractStochasticSeededProtocol,
        prot_posteriors: np.ndarray,
        alpha=0.3,
        alpha_rate=1.2,
        error: str | Callable = cap.error.mae,
        predict_train_prev=True,
    ):
        super().__init__(acc)
        self.q = quantifier
        self.protocol = protocol
        self.prot_posteriors = prot_posteriors
        self.alpha = alpha
        self.alpha_rate = alpha_rate
        self.sample_size = qp.environ["SAMPLE_SIZE"]
        self.__check_error(error)
        self.predict_train_prev = predict_train_prev

    def __check_error(self, error):
        if error in cap.error.ACCURACY_ERROR_SINGLE:
            self.error = error
        elif isinstance(error, str) and error in cap.error.ACCURACY_ERROR_SINGLE_NAMES:
            self.error = cap.error.from_name(error)
        elif hasattr(error, "__call__"):
            self.error = error
        else:
            raise ValueError(
                f"unexpected error type; must either be a callable function or a str\n"
                f"representing the name of an error function in {cap.error.ACCURACY_ERROR_NAMES}"
            )

    def fit(self, val: LabelledCollection, posteriors):
        self.q.fit(val, fit_classifier=False, val_split=val)

        # precompute classifier predictions on samples
        self.sigma_acc = [
            self.true_acc(sigma_i, P) for sigma_i, P in IT.zip_longest(self.protocol(), self.prot_posteriors)
        ]

        # precompute prevalence predictions on samples
        if self.predict_train_prev:
            self.sigma_pred_prevs = [self.q.aggregate(P) for P in self.prot_posteriors]
        else:
            self.sigma_pred_prevs = [sigma_i.prevalence() for sigma_i in self.protocol()]

        return self

    def predict(self, X, posteriors):
        test_pred_prev = self.q.quantify(X)

        if self.alpha > 0:
            # select samples from V2 with predicted prevalence close to the predicted prevalence for U
            _first = True
            selected_accuracies = []
            _alpha = self.alpha
            while _first or len(selected_accuracies) == 0:
                _first = False
                for pred_prev_i, acc_i in zip(self.sigma_pred_prevs, self.sigma_acc):
                    max_discrepancy = np.max(self.error(pred_prev_i, test_pred_prev))
                    if max_discrepancy < _alpha:
                        selected_accuracies.append(acc_i)
                _alpha *= self.alpha_rate

            return np.median(selected_accuracies)
        else:
            # mean average, weights samples from V2 according to the closeness of predicted prevalence in U
            accum_weight = 0
            epsilon = 10e-4
            moving_mean = 0
            for pred_prev_i, acc_i in zip(self.sigma_pred_prevs, self.sigma_acc):
                max_discrepancy = np.max(self.error(pred_prev_i, test_pred_prev))
                weight = -np.log(max_discrepancy + epsilon)
                accum_weight += weight
                moving_mean += weight * acc_i

            # print("prediquant_check", moving_mean, accum_weight, moving_mean / accum_weight)
            return moving_mean / accum_weight


class RQBS(CAPDirect):
    """
    Reverse Quantification-Based Sampling
    a.k.a. PabloCAP
    """

    def __init__(
        self,
        acc: Callable,
        quantifier: AggregativeQuantifier,
        n_vsamples: int = 100,
        sample_size: int = None,
        aggr: Literal["mean", "median"] = "median",
    ):
        super().__init__(acc)
        self.q = quantifier
        self.n_vsamples = n_vsamples
        self.sample_size = qp.environ["SAMPLE_SIZE"] if sample_size is None else sample_size
        self.aggr = aggr
        if self.sample_size is None:
            raise ValueError(
                'sample_size cannot be None; it must be specified directly or by setting qp.environ["SAMPLE_SIZE"]'
            )
        if aggr not in ["mean", "median"]:
            raise ValueError(f"Unknown aggregation function {aggr}, use mean or median")

    def aggr_fun(self, accs: Iterable):
        if self.aggr == "mean":
            return np.mean(accs)
        elif self.aggr == "median":
            return np.median(accs)

    def fit(self, val: LabelledCollection, posteriors):
        self.q.fit(val)
        self.val_post = LabelledCollection(instances=posteriors, labels=val.y, classes=val.classes_)
        return self

    def predict(self, X, posteriors):
        q_hat = utils.smooth(self.q.quantify(X))
        val_samples_idx = [self.val_post.sampling_index(self.sample_size, *q_hat) for _ in range(self.n_vsamples)]

        val_sample_true_accs = []
        for idx in val_samples_idx:
            vali_yhat = self.val_post.X[idx, :].argmax(axis=1)
            vali_y = self.val_post.y[idx]
            vali_ct = contingency_table(vali_y, vali_yhat, self.val_post.n_classes)
            val_sample_true_accs.append(self.acc(vali_ct))

        return self.aggr_fun(val_sample_true_accs)
