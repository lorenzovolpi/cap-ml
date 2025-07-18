import itertools as IT
import os
from traceback import print_exception

import numpy as np
import quapy as qp
from quapy.protocol import UPP

import quacc as qc
from exp.generators import (
    gen_acc_measure,
    gen_bin_datasets,
    gen_bin_lm_datasets,
    gen_classifiers,
    gen_lm_classifiers,
    gen_lm_model_dataset,
    gen_methods,
    gen_multi_datasets,
    gen_tweet_datasets,
    get_method_names,
)
from exp.report import TestReport
from exp.util import (
    fit_or_switch,
    gen_model_dataset,
    get_logger,
    get_plain_prev,
    get_predictions,
    prevs_from_prot,
    split_validation,
)
from quacc.utils.commons import save_dataset_stats, true_acc

PROBLEM = "binary"
MODEL_TYPE = "simple"
root_folder = os.path.join(qc.env["OUT_DIR"], "results")

log = get_logger()


def all_exist_pre_check(basedir, cls_name, dataset_name, model_type):
    method_names = get_method_names(PROBLEM, model_type)
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        path = TestReport.build_path(root_folder, basedir, cls_name, acc, dataset_name, method)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def experiments():
    basedir = PROBLEM
    NUM_TEST = 1000
    R_SEED = 42
    qp.environ["_R_SEED"] = R_SEED

    if PROBLEM == "binary":
        qp.environ["SAMPLE_SIZE"] = 100
        gen_datasets = gen_bin_datasets
    elif PROBLEM == "multiclass":
        qp.environ["SAMPLE_SIZE"] = 250
        gen_datasets = gen_multi_datasets
    elif PROBLEM == "tweet":
        qp.environ["SAMPLE_SIZE"] = 100
        gen_datasets = gen_tweet_datasets

    log.info("-" * 31 + "  start  " + "-" * 31)

    for (cls_name, h), (dataset_name, (L, V, U)) in gen_model_dataset(gen_classifiers, gen_datasets):
        if all_exist_pre_check(basedir, cls_name, dataset_name, MODEL_TYPE):
            log.info(f"{cls_name} on dataset={dataset_name}: all results already exist, skipping")
            continue

        # fit model
        log.info(f"{cls_name} training on dataset={dataset_name}")
        h.fit(*L.Xy)

        # test generation protocol
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=qp.environ["_R_SEED"])

        # compute some stats of the dataset
        save_dataset_stats(dataset_name, test_prot, L, V)

        # split validation set
        V1, V2_prot = split_validation(V)

        # precomumpute model posteriors for validation and test sets
        V_posteriors = h.predict_proba(V.X)
        V1_posteriors = h.predict_proba(V1.X)
        V2_prot_posteriors = []
        for sample in V2_prot():
            V2_prot_posteriors.append(h.predict_proba(sample.X))

        # get posteriors for test samples
        test_prot_posteriors, test_prot_y_hat = [], []
        for sample in test_prot():
            P = h.predict_proba(sample.X)
            test_prot_posteriors.append(P)
            test_prot_y_hat.append(np.argmax(P, axis=-1))

        # precompute the actual accuracy values
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure(multiclass=True):
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

        L_prev = get_plain_prev(L.prevalence())
        for method_name, method, val, val_posteriors in gen_methods(
            h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors, PROBLEM, MODEL_TYPE
        ):
            val_prev = get_plain_prev(V.prevalence())

            t_train = None
            for acc_name, acc_fn in gen_acc_measure(multiclass=True):
                report = TestReport(
                    root_folder, basedir, cls_name, acc_name, dataset_name, L_prev, val_prev, method_name
                )
                if report.exists():
                    log.info(f"{method_name}: {acc_name} exists, skipping")
                    continue

                try:
                    method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
                    t_train = t_train if _t_train is None else _t_train

                    test_prevs = prevs_from_prot(test_prot)
                    estim_accs, t_test_ave = get_predictions(method, test_prot, test_prot_posteriors)
                    report.add_result(test_prevs, true_accs[acc_name], estim_accs, t_train, t_test_ave)
                except Exception as e:
                    print_exception(e)
                    log.warning(f"{method_name}: {acc_name} gave error '{e}' - skipping")
                    continue

                report.save_json()

                log.info(f"{method_name}: {acc_name} done [t_train:{t_train:.3f}s; t_test_ave:{t_test_ave:.3f}s]")

    log.info("-" * 32 + "  end  " + "-" * 32)


def lmexperiments():
    basedir = PROBLEM
    NUM_TEST = 100
    R_SEED = 42
    qp.environ["_R_SEED"] = R_SEED

    if PROBLEM == "binary":
        gen_lm_datasets = gen_bin_lm_datasets
        qp.environ["SAMPLE_SIZE"] = 500

    log.info("-" * 31 + "  start  " + "-" * 31)

    for (cls_name, model), (dataset_name, (L, V, U)) in gen_lm_model_dataset(gen_lm_classifiers, gen_lm_datasets):
        if all_exist_pre_check(basedir, cls_name, dataset_name, MODEL_TYPE):
            log.info(f"{cls_name} on dataset={dataset_name}: all results already exist, skipping")
            continue

        log.info(f"{cls_name} training on dataset={dataset_name}")
        model.fit(L, dataset_name)

        # test generation protocol
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=R_SEED)

        # compute some stats of the dataset
        save_dataset_stats(dataset_name, test_prot, L, V)

        # split validation set
        V1, V2_prot = split_validation(V)

        # precomumpute model posteriors for validation and test sets
        V_posteriors = model.predict_proba(V.X, V.attention_mask, verbose=True)
        V1_posteriors = model.predict_proba(V1.X, V1.attention_mask, verbose=True)
        V2_prot_posteriors = []
        for sample in V2_prot():
            V2_prot_posteriors.append(model.predict_proba(sample.X, sample.attention_mask))

        test_prot_posteriors, test_prot_y_hat = [], []
        for sample in test_prot():
            P = model.predict_proba(sample.X, sample.attention_mask)
            test_prot_posteriors.append(P)
            test_prot_y_hat.append(np.argmax(P, axis=-1))

        # precompute the actual accuracy values
        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure(multiclass=True):
            true_accs[acc_name] = [acc_fn(y_hat, Ui.y) for y_hat, Ui in zip(test_prot_y_hat, test_prot())]

        L_prev = get_plain_prev(L.prevalence())
        for method_name, method, val, val_posteriors in gen_methods(
            model, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors, PROBLEM, MODEL_TYPE
        ):
            val_prev = get_plain_prev(val.prevalence())

            t_train = None
            for acc_name, acc_fn in gen_acc_measure(multiclass=True):
                report = TestReport(
                    root_folder, basedir, cls_name, acc_name, dataset_name, L_prev, val_prev, method_name
                )
                if report.exists():
                    log.info(f"{method_name}: {acc_name} exists, skipping")
                    continue

                try:
                    method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
                    t_train = t_train if _t_train is None else _t_train

                    test_prevs = prevs_from_prot(test_prot)
                    estim_accs, t_test_ave = get_predictions(method, test_prot, test_prot_posteriors)
                    report.add_result(test_prevs, true_accs[acc_name], estim_accs, t_train, t_test_ave)
                except Exception as e:
                    print_exception(e)
                    log.warning(f"{method_name}: {acc_name} gave_error '{e}' - skipping")
                    continue

                report.save_json(root_folder)
                log.info(f"{method_name}: {acc_name} done [t_train:{t_train:.3f}s; t_test_ave:{t_test_ave:.3f}s]")

    log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    try:
        if MODEL_TYPE == "simple":
            experiments()
        if MODEL_TYPE == "large":
            lmexperiments()
    except Exception as e:
        log.error(e)
        print_exception(e)
