import itertools as IT
import random
from copy import deepcopy
from typing import Callable

import numpy as np
import ot
import quapy as qp
import scipy as sp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier, BaseQuantifier
from quapy.protocol import UPP, AbstractProtocol
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

import quacc as qc
import quacc.models.utils as utils
from quacc.error import vanilla_acc
from quacc.models.base import ClassifierAccuracyPrediction
from quacc.models.utils import get_posteriors_from_h, max_conf, neg_entropy


def _one_hot(arr: np.ndarray, num_classes=None):
    assert arr.ndim == 1, "too many dimensions for input array"
    num_classes = num_classes if num_classes else np.max(arr)
    return np.eye(num_classes)[arr]


def _sample_label_dist(sample_size, val_prior, n_classes):
    labels = sum([[i] * int(val_prior[i] * sample_size) for i in range(n_classes)], start=[])
    rem = sample_size - len(labels)
    rem_labels = random.choices(list(range(n_classes)), weights=val_prior, k=rem)
    return np.asarray(labels + rem_labels)


class CAPDirect(ClassifierAccuracyPrediction):
    def __init__(self, acc: Callable):
        super().__init__()
        self.acc = acc

    def true_acc(self, sample: LabelledCollection, posteriors):
        y_pred = np.argmax(posteriors, axis=-1)
        y_true = sample.y
        conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sample.classes_)
        return self.acc(conf_table)

    def switch_and_fit(self, acc_fn, data, posteriors):
        self.acc = acc_fn
        return self.fit(data, posteriors)


class PrediQuant(CAPDirect):
    def __init__(
        self,
        acc_fn: Callable,
        quantifier: AggregativeQuantifier,
        protocol: AbstractProtocol,
        prot_posteriors,
        alpha=0.3,
        alpha_rate=1.2,
        error: str | Callable = qc.error.mae,
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
        if error in qc.error.ACCURACY_ERROR_SINGLE:
            self.error = error
        elif isinstance(error, str) and error in qc.error.ACCURACY_ERROR_SINGLE_NAMES:
            self.error = qc.error.from_name(error)
        elif hasattr(error, "__call__"):
            self.error = error
        else:
            raise ValueError(
                f"unexpected error type; must either be a callable function or a str\n"
                f"representing the name of an error function in {qc.error.ACCURACY_ERROR_NAMES}"
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


### baselines ###


class ATC(CAPDirect):
    VALID_FUNCTIONS = {"maxconf", "neg_entropy"}

    def __init__(self, acc_fn: Callable, scoring_fn="maxconf"):
        assert scoring_fn in ATC.VALID_FUNCTIONS, f"unknown scoring function, use any from {ATC.VALID_FUNCTIONS}"
        # assert acc_fn == 'vanilla_accuracy', \
        #    'use acc_fn=="vanilla_accuracy"; other metris are not yet tested in ATC'
        super().__init__(acc_fn)
        self.scoring_fn = scoring_fn

    def get_scores(self, P):
        if self.scoring_fn == "maxconf":
            scores = max_conf(P)
        else:
            scores = neg_entropy(P)
        return scores

    def fit(self, val: LabelledCollection, posteriors):
        pred_labels = np.argmax(posteriors, axis=1)
        true_labels = val.y
        scores = self.get_scores(posteriors)
        _, self.threshold = self.__find_ATC_threshold(scores=scores, labels=(pred_labels == true_labels))
        return self

    def predict(self, X, posteriors, oracle_prev=None):
        scores = self.get_scores(posteriors)
        # assert self.acc_fn == 'vanilla_accuracy', \
        #    'use acc_fn=="vanilla_accuracy"; other metris are not yet tested in ATC'
        return self.__get_ATC_acc(self.threshold, scores)

    def __find_ATC_threshold(self, scores, labels):
        # code copy-pasted from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py
        sorted_idx = np.argsort(scores)

        sorted_scores = scores[sorted_idx]
        sorted_labels = labels[sorted_idx]

        fp = np.sum(labels == 0)
        fn = 0.0

        min_fp_fn = np.abs(fp - fn)
        thres = 0.0
        for i in range(len(labels)):
            if sorted_labels[i] == 0:
                fp -= 1
            else:
                fn += 1

            if np.abs(fp - fn) < min_fp_fn:
                min_fp_fn = np.abs(fp - fn)
                thres = sorted_scores[i]

        return min_fp_fn, thres

    def __get_ATC_acc(self, thres, scores):
        # code copy-pasted from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py
        return np.mean(scores >= thres)


class DoC(CAPDirect):
    def __init__(self, acc_fn: Callable, protocol: AbstractProtocol, prot_posteriors, clip_vals=(0, 1)):
        super().__init__(acc_fn)
        self.protocol = protocol
        self.prot_posteriors = prot_posteriors
        self.clip_vals = clip_vals

    def _get_post_stats(self, X, y, posteriors):
        P = posteriors
        mc = max_conf(P)
        pred_labels = np.argmax(P, axis=-1)
        acc = self.acc(y, pred_labels)
        return mc, acc

    def _doc(self, mc1, mc2):
        return mc2.mean() - mc1.mean()

    def train_regression(self, prot_mcs, prot_accs):
        docs = [self._doc(self.val_mc, prot_mc_i) for prot_mc_i in prot_mcs]
        target = [self.val_acc - prot_acc_i for prot_acc_i in prot_accs]
        docs = np.asarray(docs).reshape(-1, 1)
        target = np.asarray(target)
        lin_reg = LinearRegression()
        return lin_reg.fit(docs, target)

    def predict_regression(self, test_mc):
        docs = np.asarray([self._doc(self.val_mc, test_mc)]).reshape(-1, 1)
        pred_acc = self.reg_model.predict(docs)
        return self.val_acc - pred_acc

    def fit(self, val: LabelledCollection, posteriors):
        self.val_mc, self.val_acc = self._get_post_stats(*val.Xy, posteriors)

        prot_stats = [
            self._get_post_stats(*sample.Xy, P) for sample, P in IT.zip_longest(self.protocol(), self.prot_posteriors)
        ]
        prot_mcs, prot_accs = list(zip(*prot_stats))

        self.reg_model = self.train_regression(prot_mcs, prot_accs)

        return self

    def predict(self, X, posteriors, oracle_prev=None):
        mc = max_conf(posteriors)
        acc_pred = self.predict_regression(mc)[0]
        if self.clip_vals is not None:
            acc_pred = float(np.clip(acc_pred, *self.clip_vals))
        return acc_pred


class DispersionScore(CAPDirect):
    def __init__(self, acc_fn: Callable, val_samples=100, clip_vals=(0, 1)):
        super().__init__(acc_fn)
        self.val_samples = 100
        self.clip_vals = clip_vals

    def _get_ds(self, X, post):
        y_hat = np.argmax(post, axis=-1)
        _mu = safe_mean(X, axis=0)

        def _get_mui(i):
            # check if any class has no datapoints; if so, set corresponding
            # value in _mus to _mu in order to make corresponding value in
            # _mus_l2sq to 0 to avoid nan values and mean errors
            X_i = X[y_hat == i, :]
            return safe_mean(X_i, axis=0) if X_i.shape[0] > 0 else _mu

        _mus = np.array([_get_mui(i) for i in self.classes_])
        _mus_l2sq = np.sum((_mu - _mus) ** 2, axis=-1)
        _weights = np.array([np.sum(y_hat == i) for i in self.classes_]) / y_hat.shape[0]
        dispersion_score = np.log(np.sum(_weights * _mus_l2sq) / (self.n_classes - 1) + 1e-5)
        return dispersion_score

    def _get_acc(self, y, post):
        y_hat = np.argmax(post, axis=-1)
        return vanilla_acc(y, y_hat)

    def train_reg_model(self, _dss, _accs):
        _dss = np.asarray(_dss).reshape(-1, 1)
        _accs = np.asarray(_accs)
        lin_reg = LinearRegression()
        return lin_reg.fit(_dss, _accs)

    def predict_reg_model(self, _ds):
        _ds = np.asarray([_ds]).reshape(-1, 1)
        return self.reg_model.predict(_ds)

    def fit(self, val: LabelledCollection, posteriors):
        val_prot = UPP(
            val,
            repeats=self.val_samples,
            random_state=qp.environ["_R_SEED"],
            return_type="index",
        )

        self.classes_ = val.classes_
        self.n_classes = val.n_classes
        _dss = [self._get_ds(val.X[idx, :], posteriors[idx, :]) for idx in val_prot()]
        _accs = [self._get_acc(val.y[idx], posteriors[idx, :]) for idx in val_prot()]
        self.reg_model = self.train_reg_model(_dss, _accs)

        return self

    def predict(self, X, posteriors):
        _ds = self._get_ds(X, posteriors)
        acc_pred = self.predict_reg_model(_ds)[0]
        if self.clip_vals is not None:
            acc_pred = float(np.clip(acc_pred, *self.clip_vals))
        return acc_pred


class NuclearNorm(CAPDirect):
    def __init__(self, acc_fn: Callable, val_samples=100, clip_vals=(0, 1)):
        super().__init__(acc_fn)
        self.val_samples = 100
        self.clip_vals = clip_vals

    def _get_nuc_norm(self, P):
        return np.linalg.norm(P, ord="nuc") * np.sqrt(min(P.shape[0], P.shape[1]) * P.shape[0])

    def _get_acc(self, y, post):
        y_hat = np.argmax(post, axis=-1)
        return vanilla_acc(y, y_hat)

    def train_reg_model(self, _nns, _accs):
        _nns = np.asarray(_nns).reshape(-1, 1)
        _accs = np.asarray(_accs)
        lin_reg = LinearRegression()
        return lin_reg.fit(_nns, _accs)

    def predict_reg_model(self, _ds):
        _ds = np.asarray([_ds]).reshape(-1, 1)
        return self.reg_model.predict(_ds)

    def fit(self, val: LabelledCollection, posteriors):
        val_prot = UPP(
            val,
            repeats=self.val_samples,
            random_state=qp.environ["_R_SEED"],
            return_type="index",
        )

        self.classes_ = val.classes_
        self.n_classes = val.n_classes
        _nns = [self._get_nuc_norm(posteriors[idx, :]) for idx in val_prot()]
        _accs = [self._get_acc(val.y[idx], posteriors[idx, :]) for idx in val_prot()]
        self.reg_model = self.train_reg_model(_nns, _accs)

        return self

    def predict(self, X, posteriors):
        _nn = self._get_nuc_norm(posteriors)
        acc_pred = self.predict_reg_model(_nn)[0]
        if self.clip_vals is not None:
            acc_pred = float(np.clip(acc_pred, *self.clip_vals))
        return acc_pred


class COT(CAPDirect):
    def __init__(self, acc_fn: Callable, emd_max_iter=1e8, exact_train_prev=True):
        super().__init__(acc_fn)
        self.emd_max_iter = emd_max_iter
        self.exact_train_prev = exact_train_prev

    def fit(self, val: LabelledCollection, posteriors):
        self.n_classes = val.n_classes
        self.classes = val.classes_

        # val_y = val.y
        # val_y_hat = np.argmax(posteriors, axis=-1)
        # print(val_y, val_y_hat)
        # print(val_y.shape, val_y_hat.shape)
        # self.val_labels = LabelledCollection(val_y_hat, val_y, classes=self.classes)

        self.val_prior = val.prevalence()

        return self

    def predict(self, X, posteriors):
        sample_size = X.shape[0]

        # val_y_hat, val_y = self.val_labels.uniform_sampling(sample_size).Xy
        # val_lbls = val_y if self.exact_train_prev else val_y_hat

        val_lbls = _sample_label_dist(sample_size, self.val_prior, self.n_classes)

        val_one_hot = _one_hot(val_lbls, num_classes=self.n_classes)

        M = sp.spatial.distance.cdist(val_one_hot, posteriors, "minkowski", p=1) / 2
        weights = np.asarray([])
        Pi = ot.emd(weights, weights, M, numItermax=self.emd_max_iter)
        costs = (Pi * M.shape[0] * M).sum(axis=1)
        return 1 - costs.mean()


class COTT(CAPDirect):
    def __init__(self, acc_fn: Callable, emd_max_iter=1e8, exact_train_prev=True):
        super().__init__(acc_fn)
        self.emd_max_iter = emd_max_iter
        self.exact_train_prev = exact_train_prev

    def _get_threshold(self, val: LabelledCollection, val_posteriors):
        val_y_hat = np.argmax(val_posteriors, axis=-1)
        val_y_oh = _one_hot(val.y, num_classes=self.n_classes)
        M = sp.spatial.distance.cdist(val_y_oh, val_posteriors, "minkowski", p=1)
        weights = np.asarray([])
        Pi = ot.emd(weights, weights, M, numItermax=self.emd_max_iter)
        costs = (Pi * M.shape[0] * M).sum(axis=1) * -1

        n_incorrect = (val.y != val_y_hat).sum()
        t = np.sort(costs)[n_incorrect - 1]
        return t

    def fit(self, val: LabelledCollection, posteriors):
        self.n_classes = val.n_classes
        self.classes = val.classes_

        # val_y = val.y
        # val_y_hat = np.argmax(posteriors, axis=-1)
        # self.val_labels = LabelledCollection(val_y_hat, val_y, classes=self.classes)

        self.val_prior = val.prevalence()
        self.threshold = self._get_threshold(val, posteriors)

        return self

    def predict(self, X, posteriors):
        sample_size = X.shape[0]

        # val_y_hat, val_y = self.val_labels.uniform_sampling(sample_size).Xy
        # val_lbls = val_y if self.exact_train_prev else val_y_hat

        val_lbls = _sample_label_dist(sample_size, self.val_prior, self.n_classes)
        val_one_hot = _one_hot(val_lbls, num_classes=self.n_classes)

        M = sp.spatial.distance.cdist(val_one_hot, posteriors, "minkowski", p=1)
        weights = np.asarray([])
        Pi = ot.emd(weights, weights, M, numItermax=self.emd_max_iter)
        costs = (Pi * M.shape[0] * M).sum(axis=1) * -1
        est_err = (costs < self.threshold).sum() / sample_size
        return 1 - est_err


class Q_COT(CAPDirect):
    def __init__(self, acc_fn: Callable, q: AggregativeQuantifier, emd_max_iter=1e8, exact_train_prev=True):
        super().__init__(acc_fn)
        self.q = q
        self.emd_max_iter = emd_max_iter
        self.exact_train_prev = exact_train_prev

    def fit(self, val: LabelledCollection, posteriors):
        self.n_classes = val.n_classes
        self.classes = val.classes_
        self.q.fit(val)

        return self

    def predict(self, X, posteriors):
        sample_size = X.shape[0]

        test_q_priors = self.q.quantify(X)
        test_q_labels = _sample_label_dist(sample_size, test_q_priors, self.n_classes)

        test_q_one_hot = _one_hot(test_q_labels, num_classes=self.n_classes)

        M = sp.spatial.distance.cdist(test_q_one_hot, posteriors, "minkowski", p=1) / 2
        weights = np.asarray([])
        Pi = ot.emd(weights, weights, M, numItermax=self.emd_max_iter)
        costs = (Pi * M.shape[0] * M).sum(axis=1)
        return 1 - costs.mean()


def safe_mean(X, axis=None):
    if issparse(X):
        return np.array(X.mean(axis=axis)).squeeze()
    else:
        return np.mean(X, axis=axis)
