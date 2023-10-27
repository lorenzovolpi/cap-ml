import logging as log
import multiprocessing
import time
from typing import List

import numpy as np
import pandas as pd
import quapy as qp
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

from quacc.dataset import Dataset
from quacc.environment import env
from quacc.evaluation import baseline, method
from quacc.evaluation.report import CompReport, DatasetReport, EvaluationReport

qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE

pd.set_option("display.float_format", "{:.4f}".format)


class CompEstimator:
    __dict = method._methods | baseline._baselines

    def __class_getitem__(cls, e: str | List[str]):
        if isinstance(e, str):
            try:
                return cls.__dict[e]
            except KeyError:
                raise KeyError(f"Invalid estimator: estimator {e} does not exist")
        elif isinstance(e, list):
            _subtr = [k for k in e if k not in cls.__dict]
            if len(_subtr) > 0:
                raise KeyError(
                    f"Invalid estimator: estimator {_subtr[0]} does not exist"
                )

            return [fun for k, fun in cls.__dict.items() if k in e]


CE = CompEstimator


def fit_and_estimate(_estimate, train, validation, test, _env=None):
    _env = env if _env is None else _env
    model = LogisticRegression()

    model.fit(*train.Xy)
    protocol = APP(
        test,
        n_prevalences=_env.PROTOCOL_N_PREVS,
        repeats=_env.PROTOCOL_REPEATS,
        return_type="labelled_collection",
    )
    start = time.time()
    try:
        result = _estimate(model, validation, protocol)
    except Exception as e:
        log.error(f"Method {_estimate.__name__} failed. Exception: {e}")
        return {
            "name": _estimate.__name__,
            "result": None,
            "time": 0,
        }

    end = time.time()
    log.info(f"{_estimate.__name__} finished [took {end-start:.4f}s]")

    return {
        "name": _estimate.__name__,
        "result": result,
        "time": end - start,
    }


def evaluate_comparison(
    dataset: Dataset, estimators=["OUR_BIN_SLD", "OUR_MUL_SLD"]
) -> EvaluationReport:
    # with multiprocessing.Pool(1) as pool:
    with multiprocessing.Pool(len(estimators)) as pool:
        dr = DatasetReport(dataset.name)
        log.info(f"dataset {dataset.name}")
        for d in dataset():
            log.info(f"train prev.: {np.around(d.train_prev, decimals=2)}")
            tstart = time.time()
            tasks = [(estim, d.train, d.validation, d.test) for estim in CE[estimators]]
            results = [
                pool.apply_async(fit_and_estimate, t, {"_env": env}) for t in tasks
            ]

            results_got = []
            for _r in results:
                try:
                    r = _r.get()
                    if r["result"] is not None:
                        results_got.append(r)
                except Exception as e:
                    log.error(
                        f"Dataset sample {d.train[1]:.2f} of dataset {dataset.name} failed. Exception: {e}"
                    )

            tend = time.time()
            times = {r["name"]: r["time"] for r in results_got}
            times["tot"] = tend - tstart
            log.info(
                f"Dataset sample {d.train[1]:.2f} of dataset {dataset.name} finished [took {times['tot']:.4f}s"
            )
            dr += CompReport(
                [r["result"] for r in results_got],
                name=dataset.name,
                train_prev=d.train_prev,
                valid_prev=d.validation_prev,
                times=times,
            )

    return dr
