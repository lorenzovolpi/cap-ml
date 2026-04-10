import os
from argparse import ArgumentParser
from dataclasses import dataclass
from logging import Logger
from time import time
from traceback import print_exception
from typing import Iterable

import numpy as np
import quapy as qp

import cap
from bin.reverse.config import gen_acc_measure, gen_methods, get_acc_names, get_method_names
from bin.reverse.env import PROJECT
from bin.reverse.util import all_results_exist, local_path
from cap import env as capenv
from cap.models.base import CAP, NeedsValidationProtocol
from cap.models.cont_table import CAPContingencyTable
from cap.utils.commons import get_shift, parallel
from cap_exp.pretrain.data import PretrainInfo, load_info_paths
from cap_exp.results import RDF
from cap_exp.util import get_logger, get_plain_prev, timestamp

EXPERIMENT = "main"
DOMAIN = "classic"

NUM_TEST = 1000

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


@dataclass()
class EXP:
    code: int
    p: PretrainInfo
    acc_name: str
    method_name: str
    df: RDF = None
    t_train: float = None
    t_test_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


def fit_or_switch(method: CAP, V, V_posteriors, acc_fn, is_fit):
    # TODO: add base class to better manage switch and switch_and_fit
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


def get_ct_predictions(method: CAP, test_prot, test_prot_posteriors):
    tinit = time()
    if isinstance(method, CAPContingencyTable):
        estim_accs, estim_cts = method.batch_predict(test_prot, test_prot_posteriors, get_estim_cts=True)
        estim_cts = [ct.tolist() for ct in estim_cts]
    else:
        estim_accs = method.batch_predict(test_prot, test_prot_posteriors)
        estim_cts = [None] * len(estim_accs)
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, estim_cts, t_test_ave


def exp_protocol(args: tuple[str, str, CAP]) -> EXP:
    # bundle_path, method_name, method, acc_name, acc_fn = args
    # clsf, D, method_name, method, val, val_posteriors = args
    info_path, method_name, method = args
    results = []

    p = PretrainInfo.load(info_path, fast=True)
    d_info, h_info = p.d_info, p.h_info

    all_exist = True
    for acc in get_acc_names():
        all_exist = all_exist and os.path.exists(
            local_path(p.domain, d_info.name, h_info.full_name, method_name, acc, experiment=EXPERIMENT)
        )
    if all_exist:
        results.append(EXP.EXISTS(p, "all", method_name))
        return results

    D = p.load_dataset_bundle()
    h = p.load_pretrained_classifier(D)

    D.with_test_data(NUM_TEST).with_posteriors(h)
    if isinstance(method, NeedsValidationProtocol):
        val, val_posteriors = D.V1, D.V1_posteriors
        method.set_validation_protocol(D.V2_prot, D.V2_prot_posteriors)
    else:
        val, val_posteriors = D.V, D.V_posteriors

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    df_len = D.test_prot.total()
    test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()
    tp_true_cts = D.test_prot_true_cts

    t_train, is_fit = None, False

    for acc_name, acc_fn in gen_acc_measure(d_info.n_classes > 2):
        path = local_path(p.domain, d_info.name, h_info.full_name, method_name, acc_name, experiment=EXPERIMENT)
        if os.path.exists(path):
            results.append(EXP.EXISTS(p, acc_name, method_name))
            continue

        try:
            method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, is_fit)
            t_train = t_train if _t_train is None else _t_train
            is_fit = True
            estim_accs, estim_cts, t_test_ave = get_ct_predictions(method, D.test_prot, D.test_prot_posteriors)
            true_accs = [acc_fn(ct) for ct in tp_true_cts]
            acc_err = cap.error.ae(np.array(true_accs), np.array(estim_accs)).tolist()
        except Exception as e:
            print_exception(e)
            results.append(EXP.ERROR(e, p, acc_name, method_name))
            continue

        # df_len = len(estim_accs)
        method_df = RDF.from_records(
            df_len,
            # uids=np.arange(df_len).tolist(),
            shifts=test_shift,
            true_cts=tp_true_cts,
            estim_accs=estim_accs,
            acc_err=acc_err,
            estim_cts=estim_cts,
            classifier=h_info.name,
            method=method_name,
            dataset=d_info.name,
            collection=d_info.collection,
            n_classes=d_info.n_classes,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )

        results.append(
            EXP.SUCCESS(
                p,
                acc_name,
                method_name,
                df=method_df,
                t_train=t_train,
                t_test_ave=t_test_ave,
            )
        )

    return results


def experiments(log: Logger, domain: str):
    experiment_args = []
    info_paths = load_info_paths(domain=domain)
    filtered_paths = []
    for path in info_paths:
        p = PretrainInfo.load(path, fast=True)
        d_info, h_info = p.d_info, p.h_info
        if not all_results_exist(
            p.domain, d_info.name, h_info.full_name, get_method_names(), get_acc_names(), EXPERIMENT
        ):
            filtered_paths.append(path)
        else:
            log.info(f"[{h_info.name}@{d_info.name}] all results exist, skipping")

    for info_path in filtered_paths:
        for method_name, method in gen_methods():
            experiment_args.append((info_path, method_name, method))

    results_gen: Iterable[list[EXP]] = parallel(
        func=exp_protocol,
        args_list=experiment_args,
        n_jobs=capenv["N_JOBS"],
        return_as="generator_unordered",
        max_nbytes=None,
    )

    for res in results_gen:
        for r in res:
            if r.ok:
                path = local_path(
                    r.p.domain,
                    r.p.d_info.name,
                    r.p.h_info.full_name,
                    r.method_name,
                    r.acc_name,
                    experiment=EXPERIMENT,
                )
                r.df.save_result(path)
                log.info(
                    f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name} on {r.acc_name} done [{timestamp(r.t_train, r.t_test_ave)}]"
                )
            elif r.old:
                log.info(f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name} on {r.acc_name} exists, skipping")
            elif r.error:
                log.warning(
                    f"[{r.p.h_info.name}@{r.p.d_info.name}] {r.method_name}: {r.acc_name} gave error '{r.err}' - skipping"
                )


def main():
    log = get_logger(id=f"{PROJECT}.{EXPERIMENT}")

    parser = ArgumentParser()
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments(log, pargs.domain)
    except Exception as e:
        log.error(e)
        print_exception(e)
    finally:
        log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    main()
