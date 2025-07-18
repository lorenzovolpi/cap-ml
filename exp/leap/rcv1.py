import itertools as IT
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from traceback import print_exception

import numpy as np
import pandas as pd
import quapy as qp
from sklearn.base import clone as skl_clone
from sklearn.linear_model import LogisticRegression

import exp.leap.config as cfg
import exp.leap.env as env
import quacc as qc
from exp.leap.config import (
    EXP,
    DatasetBundle,
    acc,
    gen_classifiers,
    is_excluded,
    kdey,
)
from exp.leap.util import all_exist_pre_check, gen_method_df, get_extra_from_method, local_path
from exp.util import (
    fit_or_switch,
    gen_model_dataset,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    timestamp,
)
from quacc.data.datasets import fetch_RCV1WholeDataset
from quacc.error import f1_macro, vanilla_acc
from quacc.models.cont_table import CBPE, LEAP, OCE, PHD, NaiveCAP
from quacc.models.direct import ATC, Q_COT, DispersionScore, DoC, NuclearNorm
from quacc.utils.commons import get_shift, parallel, true_acc

log = get_logger(id=f"{env.PROJECT}.rcv1")

qp.environ["SAMPLE_SIZE"] = 1000


def gen_datasets(only_names=False):
    if env.PROBLEM != "multiclass":
        return
        yield

    dval = None if only_names else fetch_RCV1WholeDataset()
    yield "RCV1", dval


def gen_methods(h, D: DatasetBundle):
    _, acc_fn = next(gen_acc_measure())
    _v = D.V, D.V_posteriors
    _v1 = D.V1, D.V1_posteriors
    yield "Naive", NaiveCAP(acc_fn), *_v
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf"), *_v
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors), *_v1
    yield "DS", DispersionScore(acc_fn), *_v
    yield "CBPE", CBPE(acc_fn), *_v
    yield "NN", NuclearNorm(acc_fn), *_v
    yield "Q-COT", Q_COT(acc_fn, kdey()), *_v
    yield "LEAP(ACC)", LEAP(acc_fn, acc(), reuse_h=h, log_true_solve=True), *_v
    yield "LEAP(KDEy-MLP)", LEAP(acc_fn, kdey(), log_true_solve=True), *_v
    yield "S-LEAP(KDEy-MLP)", PHD(acc_fn, kdey()), *_v
    yield "O-LEAP(KDEy-MLP)", OCE(acc_fn, kdey()), *_v


def gen_acc_measure():
    yield "vanilla_accuracy", vanilla_acc
    yield "macro_f1", f1_macro


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def get_method_names(with_oracle=True):
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()

    return [m for m, _, _, _ in gen_methods(mock_h, mock_D)]


def get_acc_names():
    return [acc_name for acc_name, _ in gen_acc_measure()]


def exp_protocol(args):
    cls_name, dataset_name, h, D, true_accs, method_name, method, val, val_posteriors = args
    results = []

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    t_train = None
    for acc_name, acc_fn in gen_acc_measure():
        if is_excluded(cls_name, dataset_name, method_name, acc_name):
            continue
        path = local_path(dataset_name, cls_name, method_name, acc_name)
        if os.path.exists(path):
            results.append(EXP.EXISTS(cls_name, dataset_name, acc_name, method_name))
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
            results.append(EXP.ERROR(e, cls_name, dataset_name, acc_name, method_name))
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
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )
        get_extra_from_method(method_df, method)

        results.append(
            EXP.SUCCESS(
                cls_name, dataset_name, acc_name, method_name, df=method_df, t_train=t_train, t_test_ave=t_test_ave
            )
        )

    return results


def train_cls(args):
    (cls_name, orig_h), (dataset_name, (L, V, U)) = args
    #
    # check if all results for current combination already exist
    # if so, skip the combination
    if all_exist_pre_check(dataset_name, cls_name):
        return (cls_name, dataset_name, None, None, None)
    else:
        # clone model from the original one
        h = skl_clone(orig_h)
        # fit model
        h.fit(*L.Xy)
        # create dataset bundle
        D = DatasetBundle(L.prevalence(), V, U).create_bundle(h)
        # compute true accs for h on dataset
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure():
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in D.test_prot()]
        # store h-dataset combination
        return (cls_name, dataset_name, h, D, true_accs)


def experiments():
    cls_train_args = list(gen_model_dataset(gen_classifiers, gen_datasets))
    cls_dataset_gen = parallel(
        func=train_cls,
        args_list=cls_train_args,
        n_jobs=qc.env["N_JOBS"],
        return_as="generator_unordered",
    )
    cls_dataset = []
    for cls_name, dataset_name, h, D, true_accs in cls_dataset_gen:
        if h is None and D is None:
            log.info(f"All results for {cls_name} over {dataset_name} exist, skipping")
        else:
            log.info(f"Trained {cls_name} over {dataset_name}")
            cls_dataset.append((cls_name, dataset_name, h, D, true_accs))

    exp_prot_args_list = []
    for cls_name, dataset_name, h, D, true_accs in cls_dataset:
        for method_name, method, val, val_posteriors in gen_methods(h, D):
            if all(
                [
                    os.path.exists(local_path(dataset_name, cls_name, method_name, acc_name))
                    for acc_name in get_acc_names()
                ]
            ):
                log.info(f"([{cls_name}@{dataset_name}] {method_name} on all acc measures exists, skipping")
                continue

            exp_prot_args_list.append(
                (cls_name, dataset_name, h, D, true_accs, method_name, method, val, val_posteriors)
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
                path = local_path(r.dataset_name, r.cls_name, r.method_name, r.acc_name)
                r.df.to_json(path)
                log.info(
                    f"({exp_cnt}/{n_exp}) [{r.cls_name}@{r.dataset_name}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(
                    f"({exp_cnt}/{n_exp}) [{r.cls_name}@{r.dataset_name}] {r.method_name} on {r.acc_name} exists, skipping"
                )
            elif r.error:
                log.warning(
                    f"({exp_cnt}/{n_exp}) [{r.cls_name}@{r.dataset_name}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


if __name__ == "__main__":
    parser = ArgumentParser()
    args = parser.parse_args()

    # if args.rename:
    #     rename_files()
    # elif args.ileap:
    #     import_leap_from_bcuda()
    # else:
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
