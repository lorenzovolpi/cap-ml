import time
from traceback import print_exception as traceback

import quapy as qp
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

from quacc.logger import SubLogger


def estimate_worker(_estimate, train, validation, test, _env=None, q=None):
    qp.environ["SAMPLE_SIZE"] = _env.SAMPLE_SIZE
    SubLogger.setup(q)
    log = SubLogger.logger()

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
        log.warning(f"Method {_estimate.__name__} failed. Exception: {e}")
        traceback(e)
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
