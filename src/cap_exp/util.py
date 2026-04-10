import logging
import os
from contextlib import contextmanager
from glob import glob
from time import time

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import UPP

import cap
from cap import env as capenv
from cap.models.base import ClassifierAccuracyPrediction
from cap.models.cont_table import CAPContingencyTable


def fit_or_switch(method: ClassifierAccuracyPrediction, V, V_posteriors, acc_fn, is_fit):
    if hasattr(method, "switch"):
        method, t_train = method.switch(acc_fn), None
        if not is_fit:
            tinit = time()
            method.fit(V, V_posteriors)
            t_train = time() - tinit
        return method, t_train
    elif hasattr(method, "switch_and_fit"):
        tinit = time()
        method = method.switch_and_fit(acc_fn, V, V_posteriors)
        t_train = time() - tinit
        return method, t_train
    else:
        ValueError("invalid method")


def get_ct_predictions(method: ClassifierAccuracyPrediction, test_prot, test_prot_posteriors):
    tinit = time()
    if isinstance(method, CAPContingencyTable):
        estim_accs, estim_cts = method.batch_predict(test_prot, test_prot_posteriors, get_estim_cts=True)
    else:
        estim_accs, estim_cts = (
            method.batch_predict(test_prot, test_prot_posteriors),
            None,
        )
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, estim_cts, t_test_ave


def get_logger(id="quacc"):
    _name = f"{id}_log"
    _path = os.path.join(cap.env["OUT_DIR"], f"{id}.log")
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if len(logger.handlers) == 0:
        fh = logging.FileHandler(_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%b %d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return np.around(prev[1:], decimals=4).tolist()
    else:
        return float(np.around(prev, decimals=4)[-1])


def timestamp(t_train: float, t_test_ave: float) -> str:
    t_train = round(t_train, ndigits=3)
    t_test_ave = round(t_test_ave, ndigits=3)
    return f"{t_train=}s; {t_test_ave=}s"


def get_test_prot(U: LabelledCollection, repeats=1000, sample_size=None, return_type="labelled_collection"):
    return UPP(
        U,
        repeats=repeats,
        sample_size=sample_size,
        random_state=qp.environ["_R_SEED"],
        return_type=return_type,
    )


def split_validation(V: LabelledCollection, ratio=0.6, repeats=100, sample_size=None, random_state=None):
    random_state = qp.environ["_R_SEED"] if random_state is None else random_state
    v_train, v_val = V.split_stratified(ratio, random_state=random_state)
    val_prot = UPP(
        v_val,
        repeats=repeats,
        sample_size=sample_size,
        return_type="labelled_collection",
        random_state=random_state,
    )
    return v_train, val_prot


def decorate_dataset(dataset: str):
    return r"\textsf{" + dataset + r"}"


def one_hot(y: np.ndarray, n_classes: int | None = None):
    assert isinstance(y, np.ndarray) and y.ndim == 1, (
        f"Function one_hot expects a numpy.ndarray of dimension 1; found and array of dimension {y.ndim}"
    )

    if n_classes is None:
        n_classes = max(np.unique(y).shape[0], np.max(y) + 1)

    _eye = np.eye(n_classes)
    return _eye[y, :]


@contextmanager
def temp_np_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def pretrain_basedir():
    return os.path.join(capenv["CAP_DATA"], "pretrain")


def get_dataset_path(domain: str, dataset_name: str, model_name: str | None):
    basedir = os.path.join(capenv["CAP_DATA"], "datasets")
    if model_name is None or model_name == "*":
        return glob(os.path.join(basedir, f"{domain}_{dataset_name}_*.npz"))[0]
    return os.path.join(basedir, f"{domain}_{dataset_name}_{model_name}.npz")


def get_model_outdir(h_name: str, d_name: str):
    outdir = os.path.join(capenv["CAP_DATA"], "models", h_name, d_name)
    os.makedirs(outdir, exist_ok=True)
    return outdir
