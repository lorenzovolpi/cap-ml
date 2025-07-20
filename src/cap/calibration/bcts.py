import numpy as np
import scipy

from cap.calibration.base import SimpleCalibratorFactory
from cap.calibration.util import b_gradients, compute_nll, inverse_softmax, softmax, t_gradients


def do_tempscale_and_bias_optimization(labels, preacts, bias_positions, verbose, lbfgs_kwargs, epsilon=5e-2):
    if bias_positions == "all":
        bias_positions = np.arange(labels.shape[1])

    def eval_func(x):
        t = x[0]
        bs = np.zeros(labels.shape[1])
        bs[bias_positions] = x[1:]

        nll = compute_nll(labels, preacts, t)
        # multiply by -1 because we care about *negative* log likelihood
        grads_t = t_gradients(labels, preacts, t)
        grads_b = b_gradients(labels, preacts, t, bs)
        neg_mean_grad_t = 0.0 if np.any(np.isnan(grads_t) | (grads_t == np.inf)) else -np.mean(grads_t)
        neg_mean_grad_b = np.where(
            np.any(np.isnan(grads_b) | (grads_b == np.inf), axis=0),
            np.zeros(labels.shape[1]),
            -np.mean(grads_b, axis=0),
        )

        return nll, np.hstack([[neg_mean_grad_t], neg_mean_grad_b[bias_positions]])

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
        x0=np.array([1.0] + [0.0 for _ in bias_positions]),
        bounds=[(epsilon, None)] + [(None, None) for _ in bias_positions],
        jac=True,
        method="L-BFGS-B",
        tol=1e-07,
        **lbfgs_kwargs,
    )

    if verbose:
        print(optimization_result)

    assert optimization_result.success, optimization_result

    optimal_t = optimization_result.x[0]
    biases = np.zeros(labels.shape[1])
    biases[bias_positions] = optimization_result.x[1:]

    if verbose:
        final_nll = compute_nll(labels=labels, preacts=preacts, t=optimal_t)
        print("Final NLL & grad is: ", final_nll)

    return optimal_t, biases


class BiasCorrectedTemperatureScaling(SimpleCalibratorFactory):
    def __init__(self, lbfgs_kwargs={}, verbose=False, bias_positions="all"):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        self.bias_positions = bias_positions

    def __call__(self, valid_preacts, valid_labels, posterior_supplied=False):
        if posterior_supplied:
            valid_preacts = inverse_softmax(valid_preacts)
        assert np.max(np.sum(valid_labels, axis=1) == 1.0)

        optimal_t, biases = do_tempscale_and_bias_optimization(
            valid_labels,
            valid_preacts,
            self.bias_positions,
            self.verbose,
            self.lbfgs_kwargs,
        )

        return lambda preact: softmax(
            preact=inverse_softmax(preact) if posterior_supplied else preact,
            temp=optimal_t,
            biases=biases,
        )


BCTS = BiasCorrectedTemperatureScaling
