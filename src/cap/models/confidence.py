import itertools as IT
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Callable, Literal, Self, override

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy as qp
from quapy.data import LabelledCollection
from quapy.functional import prevalence_from_labels
from quapy.method.aggregative import EMQ, AggregativeQuantifier
from quapy.method.confidence import AggregativeBootstrap
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import cap.models.utils
from cap.calib import BCTS
from cap.models import _bayes, utils
from cap.models._cbpe import PoiBin
from cap.models.cont_table import CAPContingencyTable
from cap.models.direct import CAPDirect
from cap.utils.commons import contingency_table

P_TEST_Y: str = "P_test(Y)"
P_TEST_C: str = "P_test(C)"
P_C_COND_Y: str = "P(C|Y)"


class ConfidenceInterval(ABC):
    """
    Instantiates a region based on (independent) Confidence Intervals.

    :param X: np.ndarray of shape (n_bootstrap_samples, n_classes)
    :param confidence_level: float, the confidence level (default 0.95)
    """

    def __init__(self, X, confidence_level=0.95):
        assert 0 < confidence_level < 1, f"{confidence_level=} must be in range(0,1)"

        if X is np.nan:
            self._samples = np.nan
            self._mean = np.nan
            self.low, self.high = np.nan, np.nan
        else:
            X = np.asarray(X)

            self._samples = X
            self._mean = X.mean()
            self.alpha = 1 - confidence_level

            low_perc = (self.alpha / 2.0) * 100
            high_perc = (1 - self.alpha / 2.0) * 100
            low, high = np.percentile(self._samples, q=[low_perc, high_perc])
            self.low, self.high = float(low), float(high)

    @property
    def samples(self):
        return self._samples

    @property
    def point_estimate(self):
        """
        Returns the point estimate, the class-wise average of the bootstrapped estimates

        :return: np.ndarray of shape (n_classes,)
        """
        return self._mean

    def coverage(self, true_value: float) -> float:
        """
        Checks whether a value, or a sets of values, are contained in the confidence region. The method computes the
        fraction of these that are contained in the region, if more than one value is passed. If only one value is
        passed, then it either returns 1.0 or 0.0, for indicating the value is in the region or not, respectively.

        :param true_value: a np.ndarray of shape (n_classes,) or shape (n_values, n_classes,)
        :return: float in [0,1]
        """
        return 1.0 if (self.low <= true_value) and (true_value <= self.high) else 0.0

    def amplitude(self) -> float:
        return self.high - self.low

    def winkler(self, true_acc) -> float:
        ampl = self.amplitude()
        low, high = self.interval()

        if true_acc < low:
            ampl += (2 / self.alpha) * (low - true_acc)
        elif true_acc > high:
            ampl += (2 / self.alpha) * (true_acc - high)
        return float(ampl)

    def interval(self) -> tuple[float, float]:
        return self.low, self.high


class PoissonConfidenceInterval(ConfidenceInterval):
    def __init__(self, X, confidence_level=0.95):
        assert 0 < confidence_level < 1, f"{confidence_level=} must be in range(0,1)"

        if X is np.nan:
            super().__init__(X, confidence_level=confidence_level)
        else:
            _distrib = {}
            n = len(X) - 1
            for k, pmf in enumerate(X):
                _distrib[k / n] = pmf

            X = np.asarray(_distrib.items())

            self._samples = X
            self._mean = self.__compute_expected_value(X)
            self.alpha = 1 - confidence_level
            self.low, self.high = self.__compute_ci(X)

    def __compute_expected_value(self, _samples: np.ndarray):
        expectation = 0.0
        for item in sorted(_samples.tolist()):
            x, p = item
            expectation += x * p
        return expectation

    def __compute_ci(self, _samples: np.ndarray):
        sorted_items = sorted(_samples.tolist())
        a = 0
        b = len(sorted_items) - 1
        tail_coverage = 0
        bounds_not_found = True

        while bounds_not_found is True:
            low_p = sorted_items[a][1]
            high_p = sorted_items[b][1]
            if low_p < high_p:
                if tail_coverage + low_p < self.alpha:
                    tail_coverage += low_p
                    a += 1
                else:
                    bounds_not_found = False
            else:
                if tail_coverage + high_p < self.alpha:
                    tail_coverage += high_p
                    b -= 1
                else:
                    bounds_not_found = False
        limits = (sorted_items[a][0], sorted_items[b][0])
        return limits


class CAPWithConfidence(ABC):
    @abstractmethod
    def predict_with_confidence(self, X: np.ndarray, posteriors: np.ndarray) -> ConfidenceInterval: ...

    @classmethod
    def ci_from_accs(cls, accs: np.ndarray, confidence_level: float = 0.95) -> ConfidenceInterval:
        return ConfidenceInterval(accs, confidence_level=confidence_level)


class CTCAPWithConfidence(CAPWithConfidence):
    @abstractmethod
    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray: ...

    def ci_from_cts(self, cts: np.ndarray) -> ConfidenceInterval:
        if cts is np.nan:
            return ConfidenceInterval(np.nan)

        accs = np.array([self.acc_fn(ct) for ct in cts])
        return self.ci_from_accs(accs)

    def predict_with_confidence(self, X: np.ndarray, posteriors: np.ndarray) -> ConfidenceInterval:
        cts = self.predict_ct_range(X, posteriors)
        return self.ci_from_cts(cts)


class DirectCAPWithConfidence(CAPWithConfidence):
    @abstractmethod
    def predict_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray: ...

    def predict_with_confidence(self, X: np.ndarray, posteriors: np.ndarray) -> ConfidenceInterval:
        accs = self.predict_range(X, posteriors)
        return self.ci_from_accs(accs)


class RQBS(CAPContingencyTable, CTCAPWithConfidence):
    """
    Reverse Quantification-Based Sampling
    a.k.a. PabloCAP
    """

    def __init__(
        self,
        acc: Callable,
        quantifier: AggregativeQuantifier,
        num_samples: int = 1000,
        sample_size: int = None,
        random_state=None,
    ):
        CAPContingencyTable.__init__(self, acc)
        self.q = AggregativeBootstrap(
            quantifier, n_train_samples=1, n_test_samples=num_samples, random_state=random_state
        )
        self.num_samples = num_samples
        self.sample_size = qp.environ["SAMPLE_SIZE"] if sample_size is None else sample_size
        self.random_state = qp.environ["_R_SEED"] if random_state is None else random_state
        if self.sample_size is None:
            raise ValueError(
                'sample_size cannot be None; it must be specified directly or by setting qp.environ["SAMPLE_SIZE"]'
            )

    def fit(self, val: LabelledCollection, posteriors):
        self.q.fit(*val.Xy)
        self.val_post = LabelledCollection(instances=posteriors, labels=val.y, classes=val.classes_)
        return self

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray):
        _, qhat_cr = self.q.predict_conf(X)
        # smooth prevalences to make them sum up to 1
        qhat_range = [cap.models.utils.smooth(prev_i) for prev_i in qhat_cr.samples]
        val_samples_idx = [self.val_post.sampling_index(self.sample_size, *q_hat) for q_hat in qhat_range]

        val_sample_cts = []
        for idx in val_samples_idx:
            vali_yhat = self.val_post.X[idx, :].argmax(axis=1)
            vali_y = self.val_post.y[idx]
            vali_ct = contingency_table(vali_y, vali_yhat, self.val_post.n_classes)
            val_sample_cts.append(vali_ct)

        return np.asarray(val_sample_cts)

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        return self.predict_ct_range(X, posteriors).mean(axis=0)


class PrediQuant(CAPDirect, DirectCAPWithConfidence):
    DISTANCES = ["l1", "hellinger", "jensen-shannon"]

    def __init__(
        self,
        acc: Callable,
        quantifier: AggregativeQuantifier,
        alpha=0.1,
        num_samples: int = 1000,
        sample_size: int = None,
        distance: Literal["l1", "hellinger", "jensen-shannon"] = "l1",
        reuse_h: BaseEstimator | None = None,
        predict_train_prev=True,
        random_state=None,
    ):
        super().__init__(acc)
        self.q = quantifier
        self.alpha = alpha
        self.num_samples = num_samples
        self.sample_size = qp.environ["SAMPLE_SIZE"] if sample_size is None else sample_size
        self.error = self.__get_error(distance)
        self.reuse_h = reuse_h
        self.predict_train_prev = predict_train_prev
        self.random_state = qp.environ["_R_SEED"] if random_state is None else random_state
        if self.sample_size is None:
            raise ValueError(
                'sample_size cannot be None; it must be specified directly or by setting qp.environ["SAMPLE_SIZE"]'
            )

    def __get_error(self, error_name: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if error_name == "l1":
            return utils.l1_dist
        elif error_name == "hellinger":
            return utils.hellinger_dist
        elif error_name == "jensen-shannon":
            return utils.jensen_shannon_dist
        else:
            raise ValueError(f"unexpected error type: {error_name}. Must be one of l1, hellinger, jensen-shannon")

    def __check_posteriors(self, n_classes: int, P: np.ndarray):
        if P.ndim == 1:
            P = P.reshape(-1, 1)
        if P.shape[1] != n_classes:
            P = np.hstack([P, 1.0 - P.sum(axis=1, keepdims=True)])

        return P

    def fit(self, val: LabelledCollection, posteriors):
        V1_idx, V2_idx = train_test_split(
            np.arange(len(val)), test_size=0.5, random_state=self.random_state, stratify=val.y
        )
        V1, V2 = val.sampling_from_index(V1_idx), val.sampling_from_index(V2_idx)
        V2_post = posteriors[V2_idx, :]
        sigma_idx = UPP(
            V2,
            sample_size=self.sample_size,
            repeats=int(self.num_samples / self.alpha),
            random_state=self.random_state,
            return_type="index",
        )
        sigma_indices = list(sigma_idx())
        sigma_y = [V2.y[idx] for idx in sigma_indices]
        sigma_post = [V2_post[idx, :] for idx in sigma_indices]

        if self.reuse_h is not None:
            self.q = deepcopy(self.q)
            self.q.set_params(classifier=self.reuse_h, fit_classifier=False, val_split=V1.Xy)
            self.q.fit(*val.Xy)
        else:
            self.q.fit(*val.Xy)

        # precompute classifier predictions on samples
        self.prot_posteriors = [self.__check_posteriors(val.n_classes, P_i) for P_i in sigma_post]
        self.sigma_ct = [
            contingency_table(y_i, np.argmax(P_i, axis=-1), val.n_classes)
            for y_i, P_i in IT.zip_longest(sigma_y, sigma_post)
        ]

        # precompute prevalence predictions on samples
        self.sigma_pred_prevs = [self.q.aggregate(P_i) for P_i in sigma_post]
        self.sigma_true_prevs = [prevalence_from_labels(y_i, val.classes_) for y_i in sigma_y]

        return self

    @property
    def sigma_prevs(self):
        if self.predict_train_prev:
            return self.sigma_pred_prevs
        else:
            return self.sigma_true_prevs

    def _predict_test_prior(self, X):
        return self.q.predict(X)

    def _predict_closest_idx(self, test_priors: np.ndarray):
        sigma_dists = self.error(np.array(self.sigma_prevs), test_priors)
        return np.argsort(sigma_dists)

    def predict_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        sigma_accs = np.array([self.acc(ct) for ct in self.sigma_ct])
        test_prior = self._predict_test_prior(X)
        closest_idx = self._predict_closest_idx(test_prior)
        return sigma_accs[closest_idx][: self.num_samples]

    def predict(self, X, posteriors):
        return float(self.predict_range(X, posteriors).mean())


class CBPE(CAPDirect, DirectCAPWithConfidence):
    VALID_ACCS = ["vanilla_accuracy"]

    def __init__(self, acc_name: str):
        self.acc_name = self.__check_acc(acc_name)

    def __check_acc(self, acc_name: str) -> Literal["vanilla_accuracy"]:
        if acc_name not in self.VALID_ACCS:
            # raise ValueError(f"acc_name must be one of {self.VALID_ACCS}")
            return None
        return acc_name

    def fit(self, val: LabelledCollection, posteriors) -> "CBPE":
        self.n_classes = val.n_classes
        if self.n_classes > 2 or self.acc_name is None:
            if hasattr(self, "calib"):
                del self.calib
            if hasattr(self, "val_prev"):
                del self.val_prev
            return self

        val_labels = np.eye(val.n_classes)[val.y]
        self.calib = BCTS()(posteriors, val_labels, posterior_supplied=True)
        self.val_prev = val.prevalence()
        return self

    @override
    def switch_and_fit(self, acc_fn, data, posteriors) -> "CAPDirect":
        acc_name = acc_fn.__name__
        if acc_name in ["vanilla_accuracy", "vanilla_acc"]:
            acc_name = "vanilla_accuracy"
        self.acc_name = self.__check_acc(acc_name)
        return self.fit(data, posteriors)

    @staticmethod
    def __vanilla_acc(confidences: np.ndarray) -> np.ndarray:
        n = len(confidences)
        if n == 0:
            raise ValueError(
                "\n\nEmpty list of confidence scores lead to zero division. Accuracy considered to be 0.0 in this case."
                + "\nYou may change this behaviour by setting the parameter 'zero_division' value to 1,"
                + "\nor suppress this warning by setting the parameter value to 0."
            )

        pos_confs = np.where(confidences >= 0.5, confidences, 1 - confidences)

        pb = PoiBin(pos_confs)

        k_values = list(range(n + 1))
        pmf = pb.pmf(k_values)
        accs = [pmf[k] for k in k_values]
        # accuracy_distribution = {}
        # for k in k_values:
        #     accuracy_distribution[k / n] = pmf[k]
        #
        # return accuracy_distribution
        return accs

    def predict_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        if self.n_classes > 2:
            return np.nan

        if self.acc_name is None:
            return np.nan

        posteriors_calib = self.calib(posteriors)
        _, posteriors_em = EMQ.EM(self.val_prev, posteriors_calib)
        confidences = posteriors_em.max(axis=1)

        if self.acc_name == "vanilla_accuracy":
            return np.asarray(list(self.__vanilla_acc(confidences).items()))

    @classmethod
    @override
    def ci_from_accs(cls, accs: np.ndarray, confidence_level: float = 0.95) -> ConfidenceInterval:
        return PoissonConfidenceInterval(accs, confidence_level=confidence_level)

    def predict(self, X, posteriors):
        accs = self.predict_range(X, posteriors)
        ci = self.ci_from_accs(accs)
        return ci.point_estimate


class BayesCAP: ...


class ACC_BayesCAP(CAPContingencyTable, CTCAPWithConfidence, BayesCAP):
    def __init__(self, acc_fn: Callable, num_warmup: int = 500, num_samples: int = 1000, random_state: int = None):
        CAPContingencyTable.__init__(self, acc_fn)
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.randm_state = qp.environ["_R_SEED"] if random_state is None else random_state

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        val_yhat = np.argmax(posteriors, axis=-1)
        self.classes = val.classes
        self.val_ct = confusion_matrix(val.y, val_yhat, labels=val.classes)
        return self

    def model(self, pred_posterior_count: np.ndarray, train_class_cond_count: np.ndarray):
        train_class_count = train_class_cond_count.sum(axis=1)

        K = len(pred_posterior_count)
        L = len(train_class_count)

        pi_ = numpyro.sample(P_TEST_Y, dist.Dirichlet(jnp.ones(L)))
        p_c_cond_y = numpyro.sample(P_C_COND_Y, dist.Dirichlet(jnp.ones(K).repeat(L).reshape(L, K)))

        with numpyro.plate("plate", L):
            numpyro.sample("F_yc", dist.Multinomial(train_class_count, p_c_cond_y), obs=train_class_cond_count)

        p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
        numpyro.sample("N_c", dist.Multinomial(jnp.sum(pred_posterior_count), p_c), obs=pred_posterior_count)

    def sample_posterior(self, posterior_count: np.ndarray) -> dict:
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(self.model),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            progress_bar=False,
        )
        rng_key = jax.random.PRNGKey(self.randm_state)
        mcmc.run(rng_key, pred_posterior_count=posterior_count, train_class_cond_count=self.val_ct)
        return mcmc.get_samples()

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        yhat = np.argmax(posteriors, axis=-1)
        posterior_count = qp.functional.counts_from_labels(yhat, self.classes)

        samples = self.sample_posterior(posterior_count)

        # compute P(Y,C) for all samples from P(Y) and P(C|Y)
        p_y_and_c_test = jnp.einsum("sy,syc->syc", samples[P_TEST_Y], samples[P_C_COND_Y])
        cts = np.array(jax.device_get(p_y_and_c_test))
        return cts

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        return self.predict_ct_range(X, posteriors).mean(axis=0)


class HD_BayesCAP(CAPContingencyTable, CTCAPWithConfidence, BayesCAP):
    def __init__(
        self,
        acc_fn: Callable,
        num_warmup: int = 500,
        num_samples: int = 1000,
        nbins: int = 4,
        fixed_bins: bool = False,
        prediction_threshold: float = 0.5,
        random_state: int = None,
    ):
        CAPContingencyTable.__init__(self, acc_fn)
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.randm_state = qp.environ["_R_SEED"] if random_state is None else random_state

        if not 0 <= prediction_threshold <= 1:
            raise ValueError(f"parameter {prediction_threshold=} must be in [0, 1]")

        self.nbins = nbins
        self.fixed_bins = fixed_bins
        self.prediction_threshold = prediction_threshold

        self.stan_code = _bayes.load_stan_file()

    def fit(self, val: LabelledCollection, posteriors: np.ndarray):
        self.n_classes = val.n_classes
        if self.n_classes > 2:
            return self

        self.pos_label = val.classes_[1]
        y_hat = posteriors[:, self.pos_label]

        if self.fixed_bins:
            bin_limits = np.linspace(0, 1, self.nbins + 1)
        else:
            bin_limits = np.quantile(y_hat, np.linspace(0, 1, self.nbins + 1))

        if bin_limits[0] < self.prediction_threshold < bin_limits[-1]:
            bin_limits = np.sort(np.append(bin_limits, self.prediction_threshold))
        self.bin_limits = np.unique(bin_limits)
        self.effective_nbins = len(self.bin_limits) - 1
        if self.effective_nbins < 1:
            raise ValueError("could not build valid bins from classifier predictions")

        bin_indices = np.digitize(y_hat, self.bin_limits[1:-1], right=True)

        pos_mask = val.y == self.pos_label
        neg_mask = ~pos_mask

        self.pos_hist = np.bincount(bin_indices[pos_mask], minlength=self.effective_nbins)
        self.neg_hist = np.bincount(bin_indices[neg_mask], minlength=self.effective_nbins)
        self.pred_pos_bins = (self.bin_limits[:-1] >= self.prediction_threshold).astype(float)
        return self

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray):
        if self.n_classes > 2:
            return np.nan

        Px_test = posteriors[:, self.pos_label]
        test_hist, _ = np.histogram(Px_test, bins=self.bin_limits)

        samples = _bayes.hd_bayes_stan(
            self.stan_code,
            self.effective_nbins,
            self.pos_hist,
            self.neg_hist,
            test_hist,
            self.num_samples,
            self.num_warmup,
            self.randm_state,
        )

        prevs = samples["prev"]
        p_pos_distrib = samples["p_pos"]
        p_neg_distrib = samples["p_neg"]
        tpr_distrib = p_pos_distrib @ self.pred_pos_bins
        fpr_distrib = p_neg_distrib @ self.pred_pos_bins
        tnr_distrib = 1 - fpr_distrib
        fnr_distrib = 1 - tpr_distrib
        ct_distrib = np.stack(
            [
                (1 - prevs) * tnr_distrib,
                (1 - prevs) * fpr_distrib,
                prevs * fnr_distrib,
                prevs * tpr_distrib,
            ],
            axis=1,
        ).reshape(-1, 2, 2)

        return ct_distrib

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        if self.n_classes > 2:
            return np.nan

        return self.predict_ct_range(X, posteriors).mean(axis=0)


class BootstrapCAP(ABC): ...


class BootstrapCTCAP(CAPContingencyTable, BootstrapCAP, CTCAPWithConfidence):
    def __init__(self, method: CAPContingencyTable, num_samples: int = 1000, random_state: int = None):
        CAPContingencyTable.__init__(self, method.acc_fn)
        self.base_method = method
        self.num_samples = num_samples
        self.randm_state = qp.environ["_R_SEED"] if random_state is None else random_state

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        self.base_method.fit(val, posteriors)
        return self

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        ng = np.random.RandomState(self.randm_state)
        boostraps = [ng.choice(len(X), len(X), replace=True) for _ in range(self.num_samples)]
        cts = []
        for idx in boostraps:
            X_ = X[idx]
            X_P = posteriors[idx]
            cts.append(self.base_method.predict_ct(X_, X_P))

        return np.asarray(cts)

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        return self.predict_ct_range(X, posteriors).mean(axis=0)

    def switch(self, acc_fn: Callable) -> Self:
        self.acc_fn = acc_fn
        self.base_method.switch(acc_fn)
        return self


class BootstrapDirectCAP(CAPDirect, BootstrapCAP, DirectCAPWithConfidence):
    def __init__(self, method: CAPDirect, num_samples: int = 1000, random_state: int = None):
        CAPDirect.__init__(self, method.acc)
        self.base_method = method
        self.num_samples = num_samples
        self.randm_state = qp.environ["_R_SEED"] if random_state is None else random_state

    def fit(self, val: LabelledCollection, posteriors: np.ndarray):
        self.base_method.fit(val, posteriors)
        return self

    def predict_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        ng = np.random.RandomState(self.randm_state)
        boostraps = [ng.choice(len(X), len(X), replace=True) for _ in range(self.num_samples)]
        accs = []
        for idx in boostraps:
            X_ = X[idx]
            X_P = posteriors[idx]
            accs.append(self.base_method.predict(X_, X_P))

        return np.asarray(accs)

    def predict(self, X: np.ndarray, posteriors: np.ndarray) -> float:
        return float(self.predict_range(X, posteriors).mean())

    def switch_and_fit(self, acc_fn, data, posteriors):
        self.acc = acc_fn
        self.base_method.acc = acc_fn
        return self.fit(data, posteriors)
