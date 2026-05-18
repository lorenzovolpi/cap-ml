import contextlib
import importlib
import logging
import os
import sys

import numpy as np
import stan


def load_stan_file():
    return importlib.resources.files("cap.models").joinpath("stan/hd_bayes_cap.stan").read_text(encoding="utf-8")


@contextlib.contextmanager
def _suppress_stan_logging():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


def hd_bayes_stan(
    stan_code,
    n_bins: int,
    pos_hist: np.ndarray,
    neg_hist: np.ndarray,
    test_hist: np.ndarray,
    number_of_samples: int,
    num_warmup: int,
    random_state: int,
):

    logging.getLogger("stan.common").setLevel(logging.ERROR)

    stan_data = {
        "n_bucket": n_bins,
        "train_neg": np.asarray(neg_hist, dtype=int).tolist(),
        "train_pos": np.asarray(pos_hist, dtype=int).tolist(),
        "test": np.asarray(test_hist, dtype=int).tolist(),
        "posterior": 1,
    }

    with _suppress_stan_logging():
        stan_model = stan.build(stan_code, data=stan_data, random_seed=random_state)
        fit = stan_model.sample(num_chains=1, num_samples=number_of_samples, num_warmup=num_warmup)

    def simplex_samples(name):
        samples = np.asarray(fit[name])
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        if samples.shape[-1] == n_bins:
            return samples.reshape(-1, n_bins)
        if samples.shape[0] == n_bins:
            samples = np.moveaxis(samples, 0, -1)
        return samples.reshape(-1, n_bins)

    return {
        "prev": np.asarray(fit["prev"]).reshape(-1),
        "p_pos": simplex_samples("p_pos"),
        "p_neg": simplex_samples("p_neg"),
    }
