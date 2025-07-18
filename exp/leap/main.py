import os
from argparse import ArgumentParser
from traceback import print_exception

import numpy as np
import quapy as qp
from sklearn.base import clone as skl_clone

import exp.leap.env as env
import quacc as qc
from exp.leap.config import (
    EXP,
    DatasetBundle,
    gen_acc_measure,
    gen_classifiers,
    gen_datasets,
    gen_methods,
    get_acc_names,
    is_excluded,
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
from quacc.utils.commons import get_shift, parallel, true_acc

log = get_logger(id=env.PROJECT)

qp.environ["SAMPLE_SIZE"] = 100


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
    parser.add_argument("--rename", action="store_true", help="Rename files")
    parser.add_argument("--ileap", action="store_true", help="Import LEAP results from leap_bcuda directory")
    args = parser.parse_args()

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
