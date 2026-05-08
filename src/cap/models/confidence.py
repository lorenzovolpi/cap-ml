from abc import ABC, abstractmethod
from typing import Callable, Literal, Self

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from quapy.method.confidence import AggregativeBootstrap
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

import cap.models.utils as utils
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

        X = np.asarray(X)

        self._samples = X
        self._mean = X.mean()
        self.aplha = 1 - confidence_level

        low_perc = (self.alpha / 2.0) * 100
        high_perc = (1 - self.alpha / 2.0) * 100
        low, high = np.percentile(self.samples_, q=[low_perc, high_perc])
        self.I_low, self.I_high = float(low), float(high)

    def point_estimate(self):
        """
        Returns the point estimate, the class-wise average of the bootstrapped estimates

        :return: np.ndarray of shape (n_classes,)
        """
        return self.means_

    def coverage(self, true_value: float) -> float:
        """
        Checks whether a value, or a sets of values, are contained in the confidence region. The method computes the
        fraction of these that are contained in the region, if more than one value is passed. If only one value is
        passed, then it either returns 1.0 or 0.0, for indicating the value is in the region or not, respectively.

        :param true_value: a np.ndarray of shape (n_classes,) or shape (n_values, n_classes,)
        :return: float in [0,1]
        """
        return 1.0 if (self.I_low <= true_value) and (true_value <= self.I_high) else 0.0

    def interval(self) -> tuple[float, float]:
        return self.I_low, self.I_high


class CAPWithConfidence(ABC):
    @abstractmethod
    def predict_with_confidence(self, X: np.ndarray, posteriors: np.ndarray) -> ConfidenceInterval: ...


class CTCAPWithConfidence(CAPWithConfidence):
    @abstractmethod
    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray: ...

    def ci_from_cts(self, cts: np.ndarray) -> ConfidenceInterval:
        accs = np.array([self.acc_fn(ct) for ct in cts])
        return ConfidenceInterval(accs)

    def predict_with_confidence(self, X: np.ndarray, posteriors: np.ndarray) -> ConfidenceInterval:
        cts = self.predict_ct_range(X, posteriors)
        return self.ci_from_cts(cts)


class DirectCAPWithConfidence(CAPWithConfidence):
    @abstractmethod
    def predict_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray: ...

    def predict_with_confidence(self, X: np.ndarray, posteriors: np.ndarray) -> ConfidenceInterval:
        accs = self.predict_range(X, posteriors)
        return ConfidenceInterval(accs)


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
        super().__init__(acc)
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
        self.val_y = val.y
        self.val_post = posteriors
        self.classes_ = val.classes_
        return self

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray):
        _, qhat_cr = self.q.predict_conf(X)
        qhat_range = qhat_cr.samples
        val_samples_idx = [self.val_post.sampling_index(self.sample_size, *q_hat) for q_hat in qhat_range]

        val_sample_cts = []
        for idx in val_samples_idx:
            vali_yhat = self.val_post[idx, :].argmax(axis=1)
            vali_y = self.val_y[idx]
            vali_ct = contingency_table(vali_y, vali_yhat, self.val_post.n_classes)
            val_sample_cts.append(vali_ct)

        return np.asarray(val_sample_cts)

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        return self.predict_ct_range(X, posteriors).mean(axis=0)


class BayesCAP(CAPContingencyTable, CTCAPWithConfidence):
    def __init__(self, acc_fn: Callable, num_warmup: int = 500, num_samples: int = 1000, random_state: int = None):
        super(CAPContingencyTable, self).__init__(acc_fn)
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


class BootstrapCTCAP(CAPContingencyTable, CAPWithConfidence):
    def __init__(self, method: CAPContingencyTable, num_samples: int = 1000, random_state: int = None):
        super(CAPContingencyTable, self).__init__(method.acc_fn)
        self.method = method
        self.num_samples = num_samples
        self.randm_state = qp.environ["_R_SEED"] if random_state is None else random_state

    def fit(self, val: LabelledCollection, posteriors: np.ndarray) -> Self:
        self.method.fit(val, posteriors)
        return self

    def predict_ct_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        ng = np.random.RandomState(self.randm_state)
        boostraps = [ng.choice(len(X), len(X), replace=True) for _ in range(self.num_samples)]
        cts = []
        for idx in boostraps:
            X_ = X[idx]
            X_P = posteriors[idx]
            cts.append(self.method.predict_ct(X_, X_P))

        return np.asarray(cts)

    def predict_ct(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        return self.predict_ct_range(X, posteriors).mean(axis=0)

    def switch(self, acc_fn: Callable) -> Self:
        self.acc_fn = acc_fn
        self.method.switch(acc_fn)
        return self


class BootstrapDirectCAP(CAPDirect, DirectCAPWithConfidence):
    def __init__(self, method: CAPDirect, num_samples: int = 1000, random_state: int = None):
        super(CAPContingencyTable, self).__init__(method.acc)
        self.method = method
        self.num_samples = num_samples
        self.randm_state = qp.environ["_R_SEED"] if random_state is None else random_state

    def fit(self, val: LabelledCollection, posteriors: np.ndarray):
        self.method.fit(val, posteriors)
        return self

    def predict_range(self, X: np.ndarray, posteriors: np.ndarray) -> np.ndarray:
        ng = np.random.RandomState(self.randm_state)
        boostraps = [ng.choice(len(X), len(X), replace=True) for _ in range(self.num_samples)]
        accs = []
        for idx in boostraps:
            X_ = X[idx]
            X_P = posteriors[idx]
            accs.append(self.method.predict(X_, X_P))

        return np.asarray(accs)

    def predict(self, X: np.ndarray, posteriors: np.ndarray) -> float:
        return float(self.predict_range(X, posteriors).mean())

    def switch_and_fit(self, acc_fn, data, posteriors):
        self.acc = acc_fn
        self.method.acc = acc_fn
        return self.fit(data, posteriors)
