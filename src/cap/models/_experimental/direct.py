import itertools as IT
from typing import Callable

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.protocol import AbstractStochasticSeededProtocol
from sklearn.metrics import confusion_matrix

import cap
import cap.models.utils as utils
from cap.models.direct import CAPDirect


class PrediQuant(CAPDirect):
    def __init__(
        self,
        acc_fn: Callable,
        quantifier: AggregativeQuantifier,
        protocol: AbstractStochasticSeededProtocol,
        prot_posteriors,
        alpha=0.3,
        alpha_rate=1.2,
        error: str | Callable = cap.error.mae,
        predict_train_prev=True,
    ):
        super().__init__(acc_fn)
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
        # v2, v1 = val.split_stratified(train_prop=0.5)
        # self.q.fit(v1, fit_classifier=False, val_split=v1)
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

    def predict(self, X, posteriors, oracle_prev=None):
        if oracle_prev is None:
            test_pred_prev = self.q.quantify(X)
        else:
            test_pred_prev = oracle_prev

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


class PabloCAP(CAPDirect):
    def __init__(
        self,
        acc_fn: Callable,
        quantifier: AggregativeQuantifier,
        n_val_samples=100,
        aggr="mean",
    ):
        super().__init__(acc_fn)
        self.q = quantifier
        self.n_val_samples = n_val_samples
        self.aggr = aggr
        assert aggr in [
            "mean",
            "median",
        ], "unknown aggregation function, use mean or median"

    def fit(self, val: LabelledCollection, posteriors):
        self.q.fit(val)
        label_predictions = np.argmax(posteriors, axis=-1)
        self.pre_classified = LabelledCollection(instances=label_predictions, labels=val.labels)
        return self

    def predict(self, X, posteriors, oracle_prev=None):
        if oracle_prev is None:
            pred_prev = utils.smooth(self.q.quantify(X))
        else:
            pred_prev = oracle_prev
        X_size = X.shape[0]
        acc_estim = []
        for _ in range(self.n_val_samples):
            sigma_i = self.pre_classified.sampling(X_size, *pred_prev[:-1])
            y_pred, y_true = sigma_i.Xy
            conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sigma_i.classes_)
            acc_i = self.acc(conf_table)
            acc_estim.append(acc_i)
        if self.aggr == "mean":
            return np.mean(acc_estim)
        elif self.aggr == "median":
            return np.median(acc_estim)
        else:
            raise ValueError("unknown aggregation function")
