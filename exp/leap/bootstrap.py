import os
from dataclasses import dataclass
from traceback import print_exception
from typing import override

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.base import clone as skl_clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample

import exp.leap.env as env
import quacc as qc
from exp.leap.config import (
    EXP,
    DatasetBundle,
    gen_datasets,
    kdey,
)
from exp.leap.util import all_exist_pre_check, gen_method_df, get_extra_from_method, is_excluded, local_path
from exp.util import (
    fit_or_switch,
    gen_model_dataset,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    timestamp,
)
from quacc.error import vanilla_acc
from quacc.models.cont_table import OCE
from quacc.models.direct import DoC
from quacc.utils.commons import get_shift, parallel, true_acc

SUBPROJECT = "bootstrap"
NUM_REPEATS = 500
qp.environ["SAMPLE_SIZE"] = 500

log = get_logger(id=f"{env.PROJECT}.{SUBPROJECT}")


class RUPP(UPP):
    def __init__(
        self,
        data: LabelledCollection,
        sample_size=None,
        repeats=100,
        samples_per_repeat=1,
        random_state=0,
        return_type="sample_prev",
    ):
        super().__init__(
            data,
            sample_size=sample_size,
            repeats=repeats,
            random_state=random_state,
            return_type=return_type,
        )
        self.samples_per_repeat = samples_per_repeat

    def samples_parameters(self):
        indexes = []
        for prevs in qp.functional.uniform_simplex_sampling(n_classes=self.data.n_classes, size=self.repeats):
            index = self.data.sampling_index(self.sample_size, *prevs)
            indexes.append(index)
            for _ in range(self.samples_per_repeat):
                _resample_index = resample(index, replace=True, n_samples=index.shape[0])
                indexes.append(_resample_index)
        return indexes

    # def get_dist_prevs(self) -> np.ndarray:
    #     with ExitStack() as stack:
    #         if self.random_state == -1:
    #             raise ValueError(
    #                 "The random seed has never been initialized. Set it to None not to impose replicability."
    #             )
    #         if self.random_state is not None:
    #             stack.enter_context(qp.util.temp_seed(self.random_state))
    #
    #         base_dist_prevs = qp.functional.uniform_simplex_sampling(n_classes=self.data.n_classes, size=self.repeats)
    #         return np.repeat(base_dist_prevs, self.samples_per_repeat, axis=0)


@dataclass
class RepDatasetBundle(DatasetBundle):
    @override
    def get_test_prot(self, sample_size=None):
        return RUPP(
            self.U,
            repeats=env.NUM_TEST,
            samples_per_repeat=NUM_REPEATS,
            sample_size=sample_size,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )


def gen_acc_measure():
    yield "vanilla_accuracy", vanilla_acc


def gen_methods(h: BaseEstimator, D: RepDatasetBundle):
    _, acc_fn = next(gen_acc_measure())
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors), D.V1, D.V1_posteriors
    # yield "S-LEAP(KDEy-MLP)", PHD(acc_fn, kdey()), D.V, D.V_posteriors
    yield "O-LEAP(KDEy-MLP)", OCE(acc_fn, kdey()), D.V, D.V_posteriors


def gen_classifiers():
    yield "LR", LogisticRegression()


def get_acc_names():
    return [a for a, _ in gen_acc_measure()]


def get_classifier_names():
    return [c for c, _ in gen_classifiers()]


# def gen_some_datasets():
#     n_sets = 0
#     for name, dataset in gen_datasets():
#         n_sets += 1
#         yield name, dataset
#         if n_sets >= 1:
#             break


def get_method_names():
    _, mock_acc_fn = next(gen_acc_measure())
    mock_h = LogisticRegression()
    mock_D = RepDatasetBundle.mock()
    return [m for m, _, _, _ in gen_methods(mock_h, mock_D)]


# def compute_confidence_intervals(df):
#     for _id in df["sample_distrib_id"].unique():
#         _size = len(df.loc[df["sample_distrib_id"] == _id, :])
#         _estim_accs = df.loc[df["sample_distrib_id"] == _id, "estim_accs"].to_numpy()
#         _true_accs = df.loc[df["sample_distrib_id"] == _id, "true_accs"].to_numpy()
#         ci_low, ci_high = np.percentile(_estim_accs, [2.5, 97.5])
#         ci_delta = ci_high - ci_low
#         coverage = np.logical_and(_true_accs >= ci_low, _true_accs <= ci_high)
#         df.loc[df["sample_distrib_id"] == _id, "ci_delta"] = [ci_delta] * _size
#         df.loc[df["sample_distrib_id"] == _id, "coverage"] = coverage


def compute_confidence_intervals(df):
    for _id in df["sample_distrib_id"].unique():
        _size = len(df.loc[df["sample_distrib_id"] == _id, :])
        _estim_accs = df.loc[df["sample_distrib_id"] == _id, "estim_accs"].to_numpy()[1:]
        _true_acc = df.loc[df["sample_distrib_id"] == _id, "true_accs"].to_numpy()[0]
        ci_low, ci_high = np.percentile(_estim_accs, [2.5, 97.5])
        ci_delta = ci_high - ci_low
        coverage = _true_acc >= ci_low and _true_acc <= ci_high
        df.loc[df["sample_distrib_id"] == _id, "ci_delta"] = [ci_delta] * _size
        df.loc[df["sample_distrib_id"] == _id, "coverage"] = [coverage] * _size

    return df.groupby(by=["sample_distrib_id"]).first().reset_index()


def exp_protocol(args):
    cls_name, dataset_name, h, D, true_accs, method_name, method, val, val_posteriors = args
    results = []

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    t_train = None
    for acc_name, acc_fn in gen_acc_measure():
        if is_excluded(cls_name, dataset_name, method_name, acc_name):
            continue
        path = local_path(dataset_name, cls_name, method_name, acc_name, subproject=SUBPROJECT)
        if os.path.exists(path):
            results.append(EXP.EXISTS(cls_name, dataset_name, acc_name, method_name))
            continue

        try:
            method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
            t_train = t_train if _t_train is None else _t_train

            # test_prevs = D.test_prot.get_dist_prevs().tolist()
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
            # test_prevs=test_prevs,
            sample_distrib_id=np.repeat(np.arange(env.NUM_TEST), NUM_REPEATS + 1).tolist(),
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
        method_df = compute_confidence_intervals(method_df)
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
    if all_exist_pre_check(dataset_name, cls_name, subproject=SUBPROJECT, method_names=get_method_names()):
        return (cls_name, dataset_name, None, None, None)
    else:
        # clone model from the original one
        h = skl_clone(orig_h)
        # fit model
        h.fit(*L.Xy)
        # create dataset bundle
        D = RepDatasetBundle(L.prevalence(), V, U).create_bundle(h, sample_size=qp.environ["SAMPLE_SIZE"])
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
                path = local_path(r.dataset_name, r.cls_name, r.method_name, r.acc_name, subproject=SUBPROJECT)
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
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
