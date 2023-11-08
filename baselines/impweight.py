import numpy as np
from scipy.sparse import issparse, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from baselines import densratio
from baselines.pykliep import DensityRatioEstimator


def kliep(Xtr, ytr, Xte):
    kliep = DensityRatioEstimator()
    kliep.fit(Xtr, Xte)
    return kliep.predict(Xtr)


def usilf(Xtr, ytr, Xte, alpha=0.0):
    dense_ratio_obj = densratio(Xtr, Xte, alpha=alpha, verbose=False)
    return dense_ratio_obj.compute_density_ratio(Xtr)


def logreg(Xtr, ytr, Xte):
    # check "Direct Density Ratio Estimation for
    # Large-scale Covariate Shift Adaptation", Eq.28

    if issparse(Xtr):
        X = vstack([Xtr, Xte])
    else:
        X = np.concatenate([Xtr, Xte])

    y = [0] * Xtr.shape[0] + [1] * Xte.shape[0]

    logreg = GridSearchCV(
        LogisticRegression(),
        param_grid={"C": np.logspace(-3, 3, 7), "class_weight": ["balanced", None]},
        n_jobs=-1,
    )
    logreg.fit(X, y)
    probs = logreg.predict_proba(Xtr)
    prob_train, prob_test = probs[:, 0], probs[:, 1]
    prior_train = Xtr.shape[0]
    prior_test = Xte.shape[0]
    w = (prior_train / prior_test) * (prob_test / prob_train)
    return w


kdex2_params = {"bandwidth": np.logspace(-1, 1, 20)}


def kdex2_lltr(Xtr):
    if issparse(Xtr):
        Xtr = Xtr.toarray()
    return GridSearchCV(KernelDensity(), kdex2_params).fit(Xtr).score_samples(Xtr)


def kdex2_weights(Xtr, Xte, log_likelihood_tr):
    log_likelihood_te = (
        GridSearchCV(KernelDensity(), kdex2_params).fit(Xte).score_samples(Xtr)
    )
    likelihood_tr = np.exp(log_likelihood_tr)
    likelihood_te = np.exp(log_likelihood_te)
    return likelihood_te / likelihood_tr


def get_acc(tr_preds, ytr, w):
    return np.sum((1.0 * (tr_preds == ytr)) * w) / np.sum(w)
