import numpy as np
import scipy


def compute_nll(labels, preacts, t, bs=None):
    if bs is None:
        bs = np.zeros(labels.shape[1])
    tsb_preacts = preacts / float(t) + bs[None, :]
    log_sum_exp = scipy.special.logsumexp(a=tsb_preacts, axis=1)
    tsb_logits_trueclass = np.sum(tsb_preacts * labels, axis=1)
    log_likelihoods = tsb_logits_trueclass - log_sum_exp
    nll = -np.mean(log_likelihoods)
    return nll


def t_gradients(labels, preacts, t):
    ts_preacts = preacts / float(t)
    nots_logits_trueclass = np.sum(preacts * labels, axis=1)
    exp_ts_logits = np.exp(ts_preacts)
    sum_exp = np.sum(exp_ts_logits, axis=1)
    sum_preact_times_exp = np.sum(preacts * exp_ts_logits, axis=1)
    return (sum_preact_times_exp / sum_exp - nots_logits_trueclass) / (float(t) ** 2)


def b_gradients(labels, preacts, t, bs):
    tsb_preacts = preacts / float(t) + bs[None, :]
    exp_tsb_logits = np.exp(tsb_preacts)
    sum_exp = np.sum(exp_tsb_logits, axis=1)
    return labels - (exp_tsb_logits / (sum_exp[:, None]))


def softmax(preact, temp, biases=None):
    if biases is None:
        biases = np.zeros(preact.shape[1])
    exponents = np.exp(preact / temp + biases[None, :])
    sum_exponents = np.sum(exponents, axis=1)
    return exponents / sum_exponents[:, None]


def inverse_softmax(preds):
    preds = smooth(preds, epsilon=1e-12, axis=1)
    lgP = np.log(preds)
    return lgP - lgP.mean(axis=1, keepdims=True)


def smooth(prevalences, epsilon=1e-5, axis=None):
    """
    Smooths a prevalence vector.

    :param prevalences: np.ndarray
    :param epsilon: float, a small quantity (default 1E-5)
    :return: smoothed prevalence vector
    """
    prevalences = prevalences + epsilon
    prevalences /= prevalences.sum(axis=axis, keepdims=axis is not None)
    return prevalences
