import multiprocessing
import time
import traceback
from typing import List

import pandas as pd
import quapy as qp

from quacc.dataset import Dataset
from quacc.environment import env
from quacc.evaluation import baseline, method
from quacc.evaluation.report import CompReport, DatasetReport, EvaluationReport
from quacc.evaluation.worker import estimate_worker
from quacc.logging import Logger

pd.set_option("display.float_format", "{:.4f}".format)
qp.environ["SAMPLE_SIZE"] = env.SAMPLE_SIZE
log = Logger.logger()


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


def evaluate_comparison(
    dataset: Dataset, estimators=["OUR_BIN_SLD", "OUR_MUL_SLD"]
) -> EvaluationReport:
    # with multiprocessing.Pool(1) as pool:
    with multiprocessing.Pool(len(estimators)) as pool:
        dr = DatasetReport(dataset.name)
        log.info(f"dataset {dataset.name}")
        for d in dataset():
            log.info(
                f"Dataset sample {d.train_prev[1]:.2f} of dataset {dataset.name} started"
            )
            tstart = time.time()
            tasks = [(estim, d.train, d.validation, d.test) for estim in CE[estimators]]
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
