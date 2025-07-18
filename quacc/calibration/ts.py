import numpy as np
import scipy

from quacc.calibration.base import SimpleCalibratorFactory
from quacc.calibration.util import compute_nll, inverse_softmax, softmax, t_gradients


def do_tempscale_optimization(labels, preacts, verbose, lbfgs_kwargs, epsilon=5e-2):
    def eval_func(x):
        t = x[0]
        nll = compute_nll(labels, preacts, t)
        # multiply by -1 because we care about *negative* log likelihood
        grads_t = t_gradients(labels, preacts, t)
        neg_mean_grad_t = 0.0 if np.any(np.isnan(grads_t) | (grads_t == np.inf)) else -np.mean(grads_t)

        return nll, np.array([neg_mean_grad_t])

    if verbose:
        original_nll = compute_nll(labels=labels, preacts=preacts, t=1.0, bs=np.zeros(labels.shape[1]))
        print("Original NLL is: ", original_nll)

    max_preact = np.max(np.abs(preacts))
    # compute the min value for T based on the max preact value and the max
    # float64 value
    min_t = max_preact / np.log((np.finfo(np.float64).max - 1) / max_preact)

    optimization_result = scipy.optimize.minimize(
        fun=eval_func,
        # fun=lambda x: eval_func(x)[0],
        x0=np.array([1.0]),
        bounds=[(epsilon, None)],
        jac=True,
        method="L-BFGS-B",
        tol=1e-07,
        **lbfgs_kwargs,
    )

    if verbose:
        print(optimization_result)

    assert optimization_result.success, optimization_result

    optimal_t = optimization_result.x

    if verbose:
        final_nll = compute_nll(labels=labels, preacts=preacts, t=optimal_t)
        print("Final NLL & grad is: ", final_nll)

    return optimal_t


class TemperatureScaling(SimpleCalibratorFactory):
    def __init__(self, lbfgs_kwargs={}, verbose=False):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose

    def __call__(self, valid_preacts, valid_labels, posterior_supplied=False):
        if posterior_supplied:
            valid_preacts = inverse_softmax(valid_preacts)
        assert np.max(np.sum(valid_labels, axis=1) == 1.0)

        optimal_t = do_tempscale_optimization(
            valid_labels,
            valid_preacts,
            self.verbose,
            self.lbfgs_kwargs,
        )

        return lambda preact: softmax(
            preact=inverse_softmax(preact) if posterior_supplied else preact,
            temp=optimal_t,
        )


TS = TemperatureScaling
