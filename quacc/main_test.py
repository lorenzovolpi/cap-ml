from copy import deepcopy
from time import time

import numpy as np
import win11toast
from quapy.method.aggregative import SLD
from quapy.protocol import APP, UPP
from sklearn.linear_model import LogisticRegression

from quacc.dataset import Dataset
from quacc.error import acc
from quacc.evaluation.baseline import ref
from quacc.evaluation.method import mulmc_sld
from quacc.evaluation.report import CompReport, EvaluationReport
from quacc.method.base import BinaryQuantifierAccuracyEstimator
from quacc.method.model_selection import GridSearchAE


def test_gs():
    d = Dataset(name="rcv1", target="CCAT", n_prevalences=1).get_raw()

    classifier = LogisticRegression()
    classifier.fit(*d.train.Xy)

    quantifier = SLD(LogisticRegression())
    # estimator = MultiClassAccuracyEstimator(classifier, quantifier)
    estimator = BinaryQuantifierAccuracyEstimator(classifier, quantifier)

    v_train, v_val = d.validation.split_stratified(0.6, random_state=0)
    gs_protocol = UPP(v_val, sample_size=1000, repeats=100)
    gs_estimator = GridSearchAE(
        model=deepcopy(estimator),
        param_grid={
            "q__classifier__C": np.logspace(-3, 3, 7),
            "q__classifier__class_weight": [None, "balanced"],
            "q__recalib": [None, "bcts", "ts"],
        },
        refit=False,
        protocol=gs_protocol,
        verbose=True,
    ).fit(v_train)

    estimator.fit(d.validation)

    tstart = time()
    erb, ergs = EvaluationReport("base"), EvaluationReport("gs")
    protocol = APP(
        d.test,
        sample_size=1000,
        n_prevalences=21,
        repeats=100,
        return_type="labelled_collection",
    )
    for sample in protocol():
        e_sample = gs_estimator.extend(sample)
        estim_prev_b = estimator.estimate(e_sample.X, ext=True)
        estim_prev_gs = gs_estimator.estimate(e_sample.X, ext=True)
        erb.append_row(
            sample.prevalence(),
            acc=abs(acc(e_sample.prevalence()) - acc(estim_prev_b)),
        )
        ergs.append_row(
            sample.prevalence(),
            acc=abs(acc(e_sample.prevalence()) - acc(estim_prev_gs)),
        )

    cr = CompReport(
        [erb, ergs],
        "test",
        train_prev=d.train_prev,
        valid_prev=d.validation_prev,
    )

    print(cr.table())
    print(f"[took {time() - tstart:.3f}s]")
    win11toast.notify("Test", "completed")


def test_mc():
    d = Dataset(name="rcv1", target="CCAT", prevs=[0.9]).get()[0]
    classifier = LogisticRegression().fit(*d.train.Xy)
    protocol = APP(
        d.test,
        sample_size=1000,
        repeats=100,
        n_prevalences=21,
        return_type="labelled_collection",
    )

    ref_er = ref(classifier, d.validation, protocol)
    mulmc_er = mulmc_sld(classifier, d.validation, protocol)

    cr = CompReport(
        [mulmc_er, ref_er],
        name="test_mc",
        train_prev=d.train_prev,
        valid_prev=d.validation_prev,
    )

    with open("test_mc.md", "w") as f:
        f.write(cr.data().to_markdown())


if __name__ == "__main__":
    test_mc()
