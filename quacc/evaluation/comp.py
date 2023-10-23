import multiprocessing
import time
import traceback
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
        "our_bin_SLD": method.evaluate_bin_sld,
        "our_mul_SLD": method.evaluate_mul_sld,
        "our_bin_SLD_nbvs": method.evaluate_bin_sld_nbvs,
        "our_mul_SLD_nbvs": method.evaluate_mul_sld_nbvs,
        "our_bin_SLD_bcts": method.evaluate_bin_sld_bcts,
        "our_mul_SLD_bcts": method.evaluate_mul_sld_bcts,
        "our_bin_SLD_ts": method.evaluate_bin_sld_ts,
        "our_mul_SLD_ts": method.evaluate_mul_sld_ts,
        "our_bin_SLD_vs": method.evaluate_bin_sld_vs,
        "our_mul_SLD_vs": method.evaluate_mul_sld_vs,
        "our_bin_CC": method.evaluate_bin_cc,
        "our_mul_CC": method.evaluate_mul_cc,
        "ref": baseline.reference,
        "kfcv": baseline.kfcv,
        "atc_mc": baseline.atc_mc,
        "atc_ne": baseline.atc_ne,
        "doc_feat": baseline.doc_feat,
        "rca": baseline.rca_score,
        "rca_star": baseline.rca_star_score,
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
    try:
        result = _estimate(model, validation, protocol)
    except Exception as e:
        print(f"Method {_estimate.__name__} failed.")
        traceback(e)
        return {
            "name": _estimate.__name__,
            "result": None,
            "time": 0,
        }

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
    with multiprocessing.Pool(len(estimators)) as pool:
        dr = DatasetReport(dataset.name)
        for d in dataset():
            print(f"train prev.: {d.train_prev}")
            start = time.time()
            tasks = [(estim, d.train, d.validation, d.test) for estim in CE[estimators]]
            results = [pool.apply_async(fit_and_estimate, t) for t in tasks]

            results_got = []
            for _r in results:
                try:
                    r = _r.get()
                    if r["result"] is not None:
                        results_got.append(r)
                except Exception as e:
                    print(e)

            er = EvaluationReport.combine_reports(
                *[r["result"] for r in results_got],
                name=dataset.name,
                train_prev=d.train_prev,
                valid_prev=d.validation_prev,
            )
            times = {r["name"]: r["time"] for r in results_got}
            end = time.time()
            times["tot"] = end - start
            er.times = times
            dr.add(er)
            print()

    return dr
