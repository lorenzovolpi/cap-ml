from abc import abstractmethod

import numpy as np
import scipy
import scipy.optimize


def smooth(prevalences, epsilon=1e-12, axis=None):
    prevalences = np.asarray(prevalences, dtype=float) + epsilon
    prevalences /= prevalences.sum(axis=axis, keepdims=axis is not None)
    return prevalences


def inverse_softmax(preds):
    preds = smooth(preds, axis=1)
    log_preds = np.log(preds)
    return log_preds - np.mean(log_preds, axis=1, keepdims=True)


def softmax(preact, temp, biases):
    if biases is None:
        biases = np.zeros(preact.shape[1])
    logits = preact / temp + biases[None, :]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exponents = np.exp(logits)
    sum_exponents = np.sum(exponents, axis=1)
    return exponents / sum_exponents[:, None]


def vector_scaled_softmax(preact, ws, biases):
    logits = preact * ws[None, :] + biases[None, :]
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exponents = np.exp(logits)
    sum_exponents = np.sum(exponents, axis=1)
    return exponents / sum_exponents[:, None]


def check_calibrated_posteriors(posteriors):
    if not np.all(np.isfinite(posteriors)):
        num_nan = int(np.isnan(posteriors).sum())
        num_inf = int(np.isinf(posteriors).sum())
        raise ValueError(f"calibrator returned non-finite posteriors: {num_nan=} {num_inf=}")
    return posteriors


class CalibratorFactory(object):
    @abstractmethod
    def __call__(self, valid_preacts, valid_labels): ...


def compute_nll(labels, preacts, t, bs):
    tsb_preacts = preacts / float(t) + bs[None, :]
    return compute_nll_given_preacts(labels=labels, preacts=tsb_preacts)


def compute_nll_given_preacts(labels, preacts):
    log_sum_exp = scipy.special.logsumexp(a=preacts, axis=1)
    tsb_logits_trueclass = np.sum(preacts * labels, axis=1)
    log_likelihoods = tsb_logits_trueclass - log_sum_exp
    nll = -np.mean(log_likelihoods)
    return nll


def do_regularized_tempscale_optimization(labels, preacts, beta, verbose, lbfgs_kwargs):
    # beta is the regularization parameter
    def eval_func(x):
        t = x[0]
        bs = np.array(x[1:])
        # tsb = temp_scaled_biased
        tsb_preacts = preacts / float(t) + bs[None, :]
        log_sum_exp = scipy.special.logsumexp(a=tsb_preacts, axis=1)

        exp_tsb_logits = np.exp(tsb_preacts)
        sum_exp = np.sum(exp_tsb_logits, axis=1)
        sum_preact_times_exp = np.sum(preacts * exp_tsb_logits, axis=1)

        notsb_logits_trueclass = np.sum(preacts * labels, axis=1)
        tsb_logits_trueclass = np.sum(tsb_preacts * labels, axis=1)

        log_likelihoods = tsb_logits_trueclass - log_sum_exp
        objective = -np.mean(log_likelihoods) + beta * np.sum(np.square(bs))
        grads_t = (sum_preact_times_exp / sum_exp - notsb_logits_trueclass) / (float(t) ** 2)
        grads_b = labels - (exp_tsb_logits / (sum_exp[:, None]))
        # multiply by -1 because we care about *negative* log likelihood
        mean_grad_t = -np.mean(grads_t)
        mean_grads_b = (-np.mean(grads_b, axis=0)) + (2 * bs * beta)
        return objective, np.array([mean_grad_t] + list(mean_grads_b))

    if verbose:
        original_nll = compute_nll(labels=labels, preacts=preacts, t=1.0, bs=np.zeros(labels.shape[1]))
        print("Original NLL is: ", original_nll)

    optimization_result = scipy.optimize.minimize(
        fun=eval_func,
        # fun=lambda x: eval_func(x)[0],
        x0=np.array([1.0] + [0.0 for x in range(labels.shape[1])]),
        bounds=[(0, None)] + [(None, None) for x in range(labels.shape[1])],
        jac=True,
        method="L-BFGS-B",
        tol=1e-07,
        **lbfgs_kwargs,
    )
    if verbose:
        print("Optimization Result:")
        print(optimization_result)
    assert optimization_result.success, optimization_result
    optimal_t = optimization_result.x[0]
    biases = np.array(optimization_result.x[1:])
    final_nll = compute_nll(labels=labels, preacts=preacts, t=optimal_t, bs=biases)
    if verbose:
        print("Final NLL & grad is: ", final_nll)

    return (optimal_t, biases)


def do_tempscale_optimization(labels, preacts, bias_positions, verbose, lbfgs_kwargs, min_temp=5e-2):
    if bias_positions == "all":
        bias_positions = np.arange(labels.shape[1])

    def eval_func(x):
        t = x[0]
        bs = np.zeros(labels.shape[1])
        for bias_pos_idx, bias_pos in enumerate(bias_positions):
            bs[bias_pos] = x[1 + bias_pos_idx]
        # tsb = temp_scaled_biased
        tsb_preacts = preacts / float(t) + bs[None, :]
        log_sum_exp = scipy.special.logsumexp(a=tsb_preacts, axis=1)

        stable_tsb_preacts = tsb_preacts - np.max(tsb_preacts, axis=1, keepdims=True)
        exp_tsb_logits = np.exp(stable_tsb_preacts)
        sum_exp = np.sum(exp_tsb_logits, axis=1)
        sum_preact_times_exp = np.sum(preacts * exp_tsb_logits, axis=1)

        notsb_logits_trueclass = np.sum(preacts * labels, axis=1)
        tsb_logits_trueclass = np.sum(tsb_preacts * labels, axis=1)

        log_likelihoods = tsb_logits_trueclass - log_sum_exp
        nll = -np.mean(log_likelihoods)
        grads_t = (sum_preact_times_exp / sum_exp - notsb_logits_trueclass) / (float(t) ** 2)
        grads_b = labels - (exp_tsb_logits / (sum_exp[:, None]))
        # multiply by -1 because we care about *negative* log likelihood
        mean_grad_t = -np.mean(grads_t)
        mean_grads_b = -np.mean(grads_b, axis=0)
        # only supply the gradients for the bias positions that
        # we are allowed to optimize for
        mean_grads_b_masked = []
        for bias_pos_idx, bias_pos in enumerate(bias_positions):
            mean_grads_b_masked.append(mean_grads_b[bias_pos])
        return nll, np.array([mean_grad_t] + mean_grads_b_masked)

    if verbose:
        original_nll = compute_nll(labels=labels, preacts=preacts, t=1.0, bs=np.zeros(labels.shape[1]))
        print("Original NLL is: ", original_nll)

    optimization_result = scipy.optimize.minimize(
        fun=eval_func,
        # fun=lambda x: eval_func(x)[0],
        x0=np.array([1.0] + [0.0 for x in bias_positions]),
        bounds=[(min_temp, None)] + [(None, None) for x in bias_positions],
        jac=True,
        method="L-BFGS-B",
        tol=1e-07,
        **lbfgs_kwargs,
    )
    if verbose:
        print(optimization_result)
    assert optimization_result.success, optimization_result
    biases = np.zeros(labels.shape[1])
    if hasattr(optimization_result.x, "__iter__"):
        optimal_t = optimization_result.x[0]
        for bias_pos_idx, bias_pos in enumerate(bias_positions):
            biases[bias_pos] = optimization_result.x[1 + bias_pos_idx]
        final_nll = compute_nll(labels=labels, preacts=preacts, t=optimal_t, bs=biases)
    else:
        optimal_t = optimization_result.x
        final_nll = compute_nll(labels=labels, preacts=preacts, t=optimal_t, bs=np.zeros(labels.shape[1]))
    if verbose:
        print("Final NLL & grad is: ", final_nll)

    return (optimal_t, biases)


class VectorScaling(CalibratorFactory):
    def __init__(self, lbfgs_kwargs={}, verbose=False):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose

    def _get_optimal_ws_and_biases(self, preacts, labels):

        def eval_func(x):
            ws = np.array(x[: int(len(x) / 2)])
            bs = np.array(x[int(len(x) / 2) :])

            vs_logits = preacts * ws[None, :] + bs[None, :]
            log_sum_exp = scipy.special.logsumexp(a=vs_logits, axis=1)
            exp_vs_logits = np.exp(vs_logits)
            sum_exp = np.sum(exp_vs_logits, axis=1)

            log_likelihoods = np.sum(vs_logits * labels, axis=1) - log_sum_exp
            nll = -np.mean(log_likelihoods)

            grads_ws = preacts * (labels - (exp_vs_logits / sum_exp[:, None]))
            grads_b = labels - (exp_vs_logits / sum_exp[:, None])

            # multiply by -1 because we care about *negative* log likelihood
            mean_grads_ws = -np.mean(grads_ws, axis=0)
            mean_grads_b = -np.mean(grads_b, axis=0)

            return nll, np.array(list(mean_grads_ws) + list(mean_grads_b))

        if self.verbose:
            original_nll = compute_nll(labels=labels, preacts=preacts, t=1.0, bs=np.zeros(labels.shape[1]))
            print("Original NLL is: ", original_nll)

        optimization_result = scipy.optimize.minimize(
            fun=eval_func,
            # fun=lambda x: eval_func(x)[0],
            x0=np.array([1.0 for x in range(preacts.shape[1])] + [0.0 for x in range(preacts.shape[1])]),
            bounds=[(0, None) for x in range(preacts.shape[1])] + [(None, None) for x in range(preacts.shape[1])],
            jac=True,
            method="L-BFGS-B",
            tol=1e-07,
            **self.lbfgs_kwargs,
        )
        if self.verbose:
            print(optimization_result)
        assert optimization_result.success, optimization_result

        ws = optimization_result.x[: preacts.shape[1]]
        bs = optimization_result.x[preacts.shape[1] :]
        return ws, bs

    def __call__(self, valid_preacts, valid_labels, posterior_supplied=False):
        if posterior_supplied:
            valid_preacts = inverse_softmax(valid_preacts)
        assert np.max(np.sum(valid_labels, axis=1) == 1.0)

        (ws, biases) = self._get_optimal_ws_and_biases(preacts=valid_preacts, labels=valid_labels)

        def calibrate(preact):
            posteriors = vector_scaled_softmax(
                preact=(inverse_softmax(preact) if posterior_supplied else preact), ws=ws, biases=biases
            )
            return check_calibrated_posteriors(posteriors)

        return calibrate


class TempScaling(CalibratorFactory):
    def __init__(self, ece_bins=15, lbfgs_kwargs={}, verbose=False, bias_positions=[]):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        self.ece_bins = ece_bins
        # the subset of bias positions that we are allowed to optimize for
        self.bias_positions = bias_positions

    def _get_optimal_t_and_biases(self, valid_preacts, valid_labels):
        (optimal_t, biases) = do_tempscale_optimization(
            labels=valid_labels,
            preacts=valid_preacts,
            bias_positions=self.bias_positions,
            verbose=self.verbose,
            lbfgs_kwargs=self.lbfgs_kwargs,
        )
        return (optimal_t, biases)

    def __call__(self, valid_preacts, valid_labels, posterior_supplied=False):

        if posterior_supplied:
            valid_preacts = inverse_softmax(valid_preacts)
        assert np.max(np.sum(valid_labels, axis=1) == 1.0)

        (optimal_t, biases) = self._get_optimal_t_and_biases(valid_preacts=valid_preacts, valid_labels=valid_labels)

        def calibrate(preact):
            posteriors = softmax(
                preact=(inverse_softmax(preact) if posterior_supplied else preact), temp=optimal_t, biases=biases
            )
            return check_calibrated_posteriors(posteriors)

        return calibrate


class BCTS(TempScaling):
    def __init__(self, ece_bins=15, lbfgs_kwargs={}, verbose=False):
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        super().__init__(ece_bins=ece_bins, lbfgs_kwargs=lbfgs_kwargs, verbose=verbose, bias_positions="all")


class CrossValidatedBCTS(TempScaling):
    def __init__(
        self,
        num_crossvalidation_splits=10,
        # frac_to_split_with=0.5,
        # seed=1234,
        betas_to_try=[0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        lbfgs_kwargs={},
        verbose=False,
        max_num_bias=None,
    ):
        self.num_crossvalidation_splits = num_crossvalidation_splits
        # self.frac_to_split_with = frac_to_split_with
        # self.rng = np.random.RandomState(seed)
        self.betas_to_try = betas_to_try
        self.lbfgs_kwargs = lbfgs_kwargs
        self.verbose = verbose
        self.max_num_bias = max_num_bias

    def _get_optimal_t_and_biases(self, valid_preacts, valid_labels):

        heldout_biasdiffs_at_different_betas = []
        for split_num in range(self.num_crossvalidation_splits):
            # get the CV split
            training_preacts = []
            training_labels = []
            cv_heldout_preacts = []
            cv_heldout_labels = []
            for idx in range(len(valid_preacts)):
                if (idx % self.num_crossvalidation_splits) == split_num:
                    cv_heldout_preacts.append(valid_preacts[idx])
                    cv_heldout_labels.append(valid_labels[idx])
                else:
                    training_preacts.append(valid_preacts[idx])
                    training_labels.append(valid_labels[idx])
            training_preacts = np.array(training_preacts)
            training_labels = np.array(training_labels)
            cv_heldout_preacts = np.array(cv_heldout_preacts)
            cv_heldout_labels = np.array(cv_heldout_labels)

            thissplit_heldout_biasdiff_at_different_betas = []
            for beta in self.betas_to_try:
                (_t, _biases) = do_regularized_tempscale_optimization(
                    labels=training_labels,
                    preacts=training_preacts,
                    beta=beta,
                    verbose=False,
                    lbfgs_kwargs=self.lbfgs_kwargs,
                )
                heldout_postsoftmax_preds = softmax(preact=cv_heldout_preacts, temp=_t, biases=_biases)
                thissplit_heldout_biasdiff_at_different_betas.append(
                    scipy.spatial.distance.jensenshannon(
                        p=np.mean(heldout_postsoftmax_preds, axis=0), q=np.mean(cv_heldout_labels, axis=0)
                    )
                )
            heldout_biasdiffs_at_different_betas.append(thissplit_heldout_biasdiff_at_different_betas)

        avgacrosssplits_heldout_biasdiffs_at_different_betas = np.mean(
            np.array(heldout_biasdiffs_at_different_betas), axis=0
        )

        if self.verbose:
            print("Avg heldout biasdiff history", avgacrosssplits_heldout_biasdiffs_at_different_betas)

        best_beta = self.betas_to_try[np.argmin(avgacrosssplits_heldout_biasdiffs_at_different_betas)]
        if self.verbose:
            print("Best beta", best_beta)

        (optimal_t, biases) = do_regularized_tempscale_optimization(
            labels=valid_labels, preacts=valid_preacts, beta=best_beta, verbose=False, lbfgs_kwargs=self.lbfgs_kwargs
        )

        return (optimal_t, biases)
