# import itertools
# from typing import Iterable

# import quapy as qp
# import quapy.functional as F
# from densratio import densratio
# from quapy.method.aggregative import *
# from quapy.protocol import (
#     AbstractStochasticSeededProtocol,
#     OnLabelledCollectionProtocol,
# )
# from scipy.sparse import issparse, vstack
# from scipy.spatial.distance import cdist
# from scipy.stats import multivariate_normal
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KernelDensity

import time

import numpy as np
import sklearn.metrics as metrics
from pykliep import DensityRatioEstimator
from quapy.protocol import APP
from scipy.sparse import issparse, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

import baselines.impweight as iw
from baselines.densratio import densratio
from quacc.dataset import Dataset


# ---------------------------------------------------------------------------------------
# Methods of "importance weight", e.g., by ratio density estimation (KLIEP, SILF, LogReg)
# ---------------------------------------------------------------------------------------
class ImportanceWeight:
    def weights(self, Xtr, ytr, Xte):
        ...


class KLIEP(ImportanceWeight):
    def __init__(self):
        pass

    def weights(self, Xtr, ytr, Xte):
        kliep = DensityRatioEstimator()
        kliep.fit(Xtr, Xte)
        return kliep.predict(Xtr)


class USILF(ImportanceWeight):
    def __init__(self, alpha=0.0):
        self.alpha = alpha

    def weights(self, Xtr, ytr, Xte):
        dense_ratio_obj = densratio(Xtr, Xte, alpha=self.alpha, verbose=False)
        return dense_ratio_obj.compute_density_ratio(Xtr)


class LogReg(ImportanceWeight):
    def __init__(self):
        pass

    def weights(self, Xtr, ytr, Xte):
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


class KDEx2(ImportanceWeight):
    def __init__(self):
        pass

    def weights(self, Xtr, ytr, Xte):
        params = {"bandwidth": np.logspace(-1, 1, 20)}
        log_likelihood_tr = (
            GridSearchCV(KernelDensity(), params).fit(Xtr).score_samples(Xtr)
        )
        log_likelihood_te = (
            GridSearchCV(KernelDensity(), params).fit(Xte).score_samples(Xtr)
        )
        likelihood_tr = np.exp(log_likelihood_tr)
        likelihood_te = np.exp(log_likelihood_te)
        return likelihood_te / likelihood_tr


if __name__ == "__main__":
    # d = Dataset("rcv1", target="CCAT").get_raw()
    d = Dataset("imdb", n_prevalences=1).get()[0]

    tstart = time.time()
    lr = LogisticRegression()
    lr.fit(*d.train.Xy)
    val_preds = lr.predict(d.validation.X)
    protocol = APP(
        d.test,
        n_prevalences=21,
        repeats=1,
        sample_size=100,
        return_type="labelled_collection",
    )

    results = []
    for sample in protocol():
        wx = iw.logreg(d.validation.X, d.validation.y, sample.X)
        test_preds = lr.predict(sample.X)
        estim_acc = np.sum((1.0 * (val_preds == d.validation.y)) * wx) / np.sum(wx)
        true_acc = metrics.accuracy_score(sample.y, test_preds)
        results.append((sample.prevalence(), estim_acc, true_acc))

    tend = time.time()

    for r in results:
        print(*r)

    print(f"logreg finished [took {tend-tstart:.3f}s]")
    import win11toast

    win11toast.notify("models.py", "Completed")
