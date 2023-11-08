import multiprocessing
import time
from traceback import print_exception as traceback
from typing import List

import numpy as np
import pandas as pd
import quapy as qp

from quacc.dataset import Dataset
from quacc.environment import env
from quacc.evaluation import baseline, method
from quacc.evaluation.report import CompReport, DatasetReport, EvaluationReport
from quacc.evaluation.worker import estimate_worker
from quacc.logger import Logger

pd.set_option("display.float_format", "{:.4f}".format)
qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE


class CompEstimatorName_:
    def __init__(self, ce):
        self.ce = ce

    def __getitem__(self, e: str | List[str]):
        if isinstance(e, str):
            return self.ce._CompEstimator__get(e)[0]
        elif isinstance(e, list):
            return list(self.ce._CompEstimator__get(e).keys())


class CompEstimatorFunc_:
    def __init__(self, ce):
        self.ce = ce

    def __getitem__(self, e: str | List[str]):
        if isinstance(e, str):
            return self.ce._CompEstimator__get(e)[1]
        elif isinstance(e, list):
            return list(self.ce._CompEstimator__get(e).values())


class CompEstimator:
    __dict = method._methods | baseline._baselines

    def __get(cls, e: str | List[str]):
        if isinstance(e, str):
            try:
                return (e, cls.__dict[e])
            except KeyError:
                raise KeyError(f"Invalid estimator: estimator {e} does not exist")
        elif isinstance(e, list):
            _subtr = np.setdiff1d(e, list(cls.__dict.keys()))
            if len(_subtr) > 0:
                raise KeyError(
                    f"Invalid estimator: estimator {_subtr[0]} does not exist"
                )

            e_fun = {k: fun for k, fun in cls.__dict.items() if k in e}
            if "ref" not in e:
                e_fun["ref"] = cls.__dict["ref"]

            return e_fun

    @property
    def name(self):
        return CompEstimatorName_(self)

    @property
    def func(self):
        return CompEstimatorFunc_(self)


CE = CompEstimator()


def evaluate_comparison(dataset: Dataset, estimators=None) -> EvaluationReport:
    log = Logger.logger()
    # with multiprocessing.Pool(1) as pool:
    with multiprocessing.Pool(len(estimators)) as pool:
        dr = DatasetReport(dataset.name)
        log.info(f"dataset {dataset.name}")
        for d in dataset():
            log.info(
                f"Dataset sample {d.train_prev[1]:.2f} of dataset {dataset.name} started"
            )
            tstart = time.time()
            tasks = [
                (estim, d.train, d.validation, d.test) for estim in CE.func[estimators]
            ]
            results = [
                pool.apply_async(estimate_worker, t, {"_env": env, "q": Logger.queue()})
                for t in tasks
            ]

            results_got = []
            for _r in results:
                try:
                    r = _r.get()
                    if r["result"] is not None:
                        results_got.append(r)
                except Exception as e:
                    log.warning(
                        f"Dataset sample {d.train_prev[1]:.2f} of dataset {dataset.name} failed. Exception: {e}"
                    )

            tend = time.time()
            times = {r["name"]: r["time"] for r in results_got}
            times["tot"] = tend - tstart
            log.info(
                f"Dataset sample {d.train_prev[1]:.2f} of dataset {dataset.name} finished [took {times['tot']:.4f}s]"
            )
            try:
                cr = CompReport(
                    [r["result"] for r in results_got],
                    name=dataset.name,
                    train_prev=d.train_prev,
                    valid_prev=d.validation_prev,
                    times=times,
                )
            except Exception as e:
                log.warning(
                    f"Dataset sample {d.train_prev[1]:.2f} of dataset {dataset.name} failed. Exception: {e}"
                )
                traceback(e)
                cr = None
            dr += cr
    return dr
