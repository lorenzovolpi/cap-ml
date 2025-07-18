import itertools as IT
import os
from dataclasses import dataclass
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from sklearn.base import BaseEstimator
from sklearn.base import clone as skl_clone
from sklearn.linear_model import LogisticRegression

import exp.leap.config as cfg
import exp.leap.env as env
import quacc as qc
from exp.leap.config import DatasetBundle, is_excluded, kdey
from exp.leap.util import all_exist_pre_check, gen_method_df, get_extra_from_method, local_path
from exp.util import (
    fit_or_switch,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    timestamp,
)
from quacc.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset, sort_datasets_by_size
from quacc.error import vanilla_acc
from quacc.models.cont_table import OCE
from quacc.models.direct import DoC
from quacc.utils.commons import get_shift, parallel, true_acc

SUBPROJECT = "sample_size"

log = get_logger(id=f"{env.PROJECT}.{SUBPROJECT}")


def dataset_full_name(dataset_name, sample_size):
    return f"{dataset_name}_{sample_size}"


def gen_sample_sizes():
    return np.unique(np.around(np.hstack([np.logspace(1, 4, 20), np.logspace(1, 4, 4)]), decimals=0).astype("int"))
    # if cfg.PROBLEM == "binary":
    #     return np.unique(np.around(np.hstack([np.logspace(1, 4, 20), np.logspace(1, 4, 4)]), decimals=0).astype("int"))
    # elif cfg.PROBLEM == "multiclass":
    #     return np.unique(
    #         np.around(np.hstack([np.logspace(1, 4.4, 20)[:-1], np.logspace(1, 4, 4), [2.5 * 1e4]]), decimals=0).astype(
    #             "int"
    #         )
    #     )


def gen_datasets(only_names=False):
    N_DATASETS = 3
    if env.PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        for dn in _sorted_uci_names[:N_DATASETS]:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif env.PROBLEM == "multiclass":
        _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIMulticlassDataset)
        for dn in _sorted_uci_names[:N_DATASETS]:
            dval = None if only_names else fetch_UCIMulticlassDataset(dn)
            yield dn, dval


def gen_methods(h: BaseEstimator, D: DatasetBundle):
    _, acc_fn = next(gen_acc_measure())
    yield "O-LEAP(KDEy-MLP)", OCE(acc_fn, kdey()), D.V, D.V_posteriors
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors), D.V1, D.V1_posteriors


def gen_acc_measure():
    yield "vanilla_accuracy", vanilla_acc


def get_method_names():
    _, mock_acc_fn = next(gen_acc_measure())
    mock_h = LogisticRegression()
    mock_D = DatasetBundle.mock()
    return [m for m, _, _, _ in gen_methods(mock_h, mock_D)]


def gen_classifiers():
    yield "LR", LogisticRegression()


def get_dataset_names():
    return [d for d, _ in gen_datasets(only_names=True)]


def get_classifier_names():
    return [c for c, _ in gen_classifiers()]


def get_acc_names():
    return [acc_name for acc_name, _ in gen_acc_measure()]


@dataclass
class EXP_ss:
    code: int
    cls_name: str
    dataset_name: str
    sample_size: int
    acc_name: str
    method_name: str
    df: pd.DataFrame = None
    t_train: float = None
    t_test_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP_ss(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP_ss(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP_ss(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


def exp_protocol(args):
    cls_name, dataset_name, sample_size, h, D, true_accs, method_name, method, val, val_posteriors = args
    qp.environ["SAMPLE_SIZE"] = sample_size
    results = []

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    t_train = None
    for acc_name, acc_fn in gen_acc_measure():
        if is_excluded(cls_name, dataset_name, method_name, acc_name):
            continue
        path = local_path(
            dataset_full_name(dataset_name, sample_size), cls_name, method_name, acc_name, subproject=SUBPROJECT
        )
        if os.path.exists(path):
            results.append(EXP_ss.EXISTS(cls_name, dataset_name, sample_size, acc_name, method_name))
            continue

        try:
            method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
            t_train = t_train if _t_train is None else _t_train

            test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()
            estim_accs, estim_cts, t_test_ave = get_ct_predictions(method, D.test_prot, D.test_prot_posteriors)
            if estim_cts is None:
                estim_cts = [None] * len(estim_accs)
            else:
                estim_cts = [ct.tolist() for ct in estim_cts]
        except Exception as e:
            print_exception(e)
            results.append(EXP_ss.ERROR(e, cls_name, dataset_name, sample_size, acc_name, method_name))
            continue

        ae = qc.error.ae(np.array(true_accs[acc_name]), np.array(estim_accs)).tolist()

        df_len = len(estim_accs)
        method_df = gen_method_df(
            df_len,
            shifts=test_shift,
            true_accs=true_accs[acc_name],
            estim_accs=estim_accs,
            acc_err=ae,
            estim_cts=estim_cts,
            true_cts=D.test_prot_true_cts,
            classifier=cls_name,
            method=method_name,
            dataset=dataset_name,
            sample_size=sample_size,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )
        get_extra_from_method(method_df, method)

        results.append(
            EXP_ss.SUCCESS(
                cls_name,
                dataset_name,
                sample_size,
                acc_name,
                method_name,
                df=method_df,
                t_train=t_train,
                t_test_ave=t_test_ave,
            )
        )

    return results


def train_cls(args):
    (cls_name, orig_h), (dataset_name, (L, V, U)), sample_size = args
    #
    # check if all results for current combination already exist
    # if so, skip the combination
    if all_exist_pre_check(dataset_full_name(dataset_name, sample_size), cls_name, subproject=SUBPROJECT):
        return (cls_name, dataset_name, sample_size, None, None, None)
    else:
        # clone model from the original one
        h = skl_clone(orig_h)
        # fit model
        h.fit(*L.Xy)
        # create dataset bundle
        D = DatasetBundle(L.prevalence(), V, U).create_bundle(h, sample_size=sample_size)
        # compute true accs for h on dataset
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in D.test_prot()]
        # store h-dataset combination
        return (cls_name, dataset_name, sample_size, h, D, true_accs)


def experiments():
    cls_train_args = []
    for _model in gen_classifiers():
        for _dataset in gen_datasets():
            for sample_size in gen_sample_sizes():
                cls_train_args.append((_model, _dataset, sample_size))

    cls_dataset_gen = parallel(
        func=train_cls,
        args_list=cls_train_args,
        n_jobs=qc.env["N_JOBS"],
        return_as="generator_unordered",
    )
    cls_dataset = []
    for cls_name, dataset_name, sample_size, h, D, true_accs in cls_dataset_gen:
        if h is None and D is None:
            log.info(f"All results for {cls_name} over {dataset_name}_{sample_size} exist, skipping")
        else:
            log.info(f"Trained {cls_name} over {dataset_name}_{sample_size}")
            cls_dataset.append((cls_name, dataset_name, sample_size, h, D, true_accs))

    exp_prot_args_list = []
    for cls_name, dataset_name, sample_size, h, D, true_accs in cls_dataset:
        for method_name, method, val, val_posteriors in gen_methods(h, D):
            exp_prot_args_list.append(
                (cls_name, dataset_name, sample_size, h, D, true_accs, method_name, method, val, val_posteriors)
            )

    results_gen = parallel(
        func=exp_protocol,
        args_list=exp_prot_args_list,
        n_jobs=qc.env["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    exp_cnt, n_exp = 0, len(exp_prot_args_list) * len(get_acc_names())
    for res in results_gen:
        for r in res:
            exp_cnt += 1
            if r.ok:
                path = local_path(
                    dataset_full_name(r.dataset_name, r.sample_size),
                    r.cls_name,
                    r.method_name,
                    r.acc_name,
                    subproject=SUBPROJECT,
                )
                r.df.to_json(path)
                log.info(
                    f"({exp_cnt}/{n_exp}) [{r.cls_name}@{r.dataset_name}_{r.sample_size}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(
                    f"({exp_cnt}/{n_exp}) [{r.cls_name}@{r.dataset_name}_{r.sample_size}] {r.method_name} on {r.acc_name} exists, skipping"
                )
            elif r.error:
                log.warning(
                    f"({exp_cnt}/{n_exp}) [{r.cls_name}@{r.dataset_name}_{r.sample_size}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
