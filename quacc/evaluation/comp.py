import multiprocessing
import time
from typing import List

import pandas as pd
import quapy as qp
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

from quacc.dataset import Dataset
from quacc.environ import env
from quacc.evaluation import baseline, method
from quacc.evaluation.report import DatasetReport, EvaluationReport

qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE

pd.set_option("display.float_format", "{:.4f}".format)


class CompEstimator:
    __dict = {
        "OUR_BIN_SLD": method.evaluate_bin_sld,
        "OUR_MUL_SLD": method.evaluate_mul_sld,
        "KFCV": baseline.kfcv,
        "ATC_MC": baseline.atc_mc,
        "ATC_NE": baseline.atc_ne,
        "DOC_FEAT": baseline.doc_feat,
        "RCA": baseline.rca_score,
        "RCA_STAR": baseline.rca_star_score,
    }

    def __class_getitem__(cls, e: str | List[str]):
        if isinstance(e, str):
            try:
                return cls.__dict[e]
            except KeyError:
                raise KeyError(f"Invalid estimator: estimator {e} does not exist")
        elif isinstance(e, list):
            try:
                return [cls.__dict[est] for est in e]
            except KeyError as ke:
                raise KeyError(
                    f"Invalid estimator: estimator {ke.args[0]} does not exist"
                )


CE = CompEstimator


def fit_and_estimate(_estimate, train, validation, test):
    model = LogisticRegression()

    model.fit(*train.Xy)
    protocol = APP(
        test, n_prevalences=env.PROTOCOL_N_PREVS, repeats=env.PROTOCOL_REPEATS
    )
    start = time.time()
    result = _estimate(model, validation, protocol)
    end = time.time()
    print(f"{_estimate.__name__}: {end-start:.2f}s")

    return {
        "name": _estimate.__name__,
        "result": result,
        "time": end - start,
    }


def evaluate_comparison(
    dataset: Dataset, estimators=["OUR_BIN_SLD", "OUR_MUL_SLD"]
) -> EvaluationReport:
    with multiprocessing.Pool(8) as pool:
        dr = DatasetReport(dataset.name)
        for d in dataset():
            print(f"train prev.: {d.train_prev}")
            start = time.time()
            tasks = [(estim, d.train, d.validation, d.test) for estim in CE[estimators]]
            results = [pool.apply_async(fit_and_estimate, t) for t in tasks]
            results = list(map(lambda r: r.get(), results))
            er = EvaluationReport.combine_reports(
                *list(map(lambda r: r["result"], results)), name=dataset.name
            )
            times = {r["name"]: r["time"] for r in results}
            end = time.time()
            times["tot"] = end - start
            er.times = times
            er.train_prevs = d.prevs
            dr.add(er)
            print()

    return dr
