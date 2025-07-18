import numpy as np


def adaptive_bins(confs, n_bins):
    assert n_bins > 0, f"Invalid number of bins: {n_bins}"
    _, npb_i = np.modf(confs.shape[0] / n_bins)
    npb_r = int(confs.shape[0] - npb_i * n_bins)
    histo = np.full(n_bins, npb_i) + np.hstack([np.ones(npb_r), np.zeros(n_bins - npb_r)])
    bin_idx = np.cumsum(histo, dtype=np.int64) - 1
    # add an epsilon to make bin limits non overlapping to actual values
    sorted_confs = np.sort(confs)
    epsilon = (sorted_confs[1:] - sorted_confs[:-1]).min() / 10
    bins = sorted_confs[bin_idx] + epsilon
    # make the last limit equal to 1
    bins[-1] = 1.0
    # add 0 as the first limit
    bins = np.insert(bins, 0, 0)
    return bins


def calibration_error(P, y, n_bins=15, adaptive_binning=True, norm="l1"):
    """
    Computes the calibration from the posteriors obtained from a classification model given the true labels.

    :param P: posteriors obtained from the model on a dataset
    :param y: true labels of the dataset
    :param n_bins: number of bins used to compute the error; default value=15
    :param adataptive_binning: if true all bins contain the same number of datapoints, otherwise all bins have the same width; default value=True
    :param norm: norm used to compute the error; available values are 'l1', 'l2' and 'inf'; default value='l1'
    :return: calibration error
    """
    y_hat = np.argmax(P, axis=1)
    top_P = P[np.arange(P.shape[0]), y_hat]

    if adaptive_binning:
        bins = adaptive_bins(top_P, n_bins)
        bin_weights = bins[1:] - bins[:-1]
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_weights = np.histogram(top_P, bins)[0] / P.shape[0]

    binning = np.digitize(top_P, bins, right=True)

    bin_ce = np.zeros(n_bins)
    for b in np.unique(binning):
        b_idx = np.nonzero(binning == b)[0]
        bin_conf_err = (y[b_idx] == y_hat[b_idx]).mean() - top_P[b_idx].mean()
        bin_ce[b - 1] = bin_conf_err

    if norm == "l1":
        return np.sum(bin_weights * np.abs(bin_ce))
    elif norm == "l2":
        return np.sqrt(np.sum(bin_weights * bin_ce**2))
    elif norm == "inf":
        return np.max(bin_ce)


if __name__ == "__main__":
    a = np.random.rand(100)
    n_bins = 15
    print(np.sort(a))
    bins = adaptive_bins(a, n_bins)
    print(bins)
    histo, _ = np.histogram(a, bins)
    print(histo)
    bin_weights = bins[1:] - bins[:-1]
    print(bin_weights)
    assert np.isclose(bin_weights.sum(), 1)
