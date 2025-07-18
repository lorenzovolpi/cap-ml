import itertools as IT
import os
from time import time
from traceback import print_exception
from typing import override

import numpy as np
import quapy as qp
import scipy.sparse.linalg
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.functional import prevalence_from_labels
from sklearn.base import clone as skl_clone
from sklearn.linear_model import LogisticRegression

import exp.leap.env as env
import quacc as qc
from exp.leap.config import EXP, DatasetBundle, acc, is_excluded, kdey
from exp.leap.util import all_exist_pre_check, gen_method_df, get_extra_from_method, local_path
from exp.util import fit_or_switch, gen_model_dataset, get_logger, get_plain_prev, timestamp
from quacc.data.datasets import fetch_UCIBinaryDataset, sort_datasets_by_size
from quacc.error import vanilla_acc
from quacc.models._leap_opt import _optim_Adam, _optim_cvxpy, _optim_lsq_linear, _optim_minimize
from quacc.models.cont_table import (
    CAPContingencyTable,
    ContTableTransferCAP,
    NsquaredEquationsCAP,
    OverConstrainedEquationsCAP,
)
from quacc.utils.commons import get_shift, parallel, true_acc

SUBPROJECT = "qerr"
log = get_logger(id=f"{env.PROJECT}.{SUBPROJECT}")

qp.environ["SAMPLE_SIZE"] = 100


class CAPCTQ_qerr: ...


class LEAP_qerr(NsquaredEquationsCAP, CAPCTQ_qerr):
    @override
    def predict_ct(self, test, posteriors):
        n = self.cont_table.shape[1]

        h_label_preds = np.argmax(posteriors, axis=-1)

        cc_prev_estim = prevalence_from_labels(h_label_preds, self.classes_)
        q_prev_estim = self.q.quantify(test)

        A = self.A
        b = self.partial_b

        # b is partially filled; we finish the vector by plugin in the classify and count
        # prevalence estimates (n-1 values only), and the quantification estimates (n-1 values only)

        b[-2 * (n - 1) : -(n - 1)] = cc_prev_estim[1:]
        b[-(n - 1) :] = q_prev_estim[1:]

        # try the fast solution (may not be valid)
        if self.sparse_matrix:
            x = scipy.sparse.linalg.spsolve(A, b)
        else:
            x = np.linalg.solve(A, b)

        _true_solve = True
        n_classes = n**2
        if any(x < 0) or not np.isclose(x.sum(), 1) or self.always_optimize:
            self._sout("L", end="")
            _true_solve = False

            # try the iterative solution
            def loss(x):
                return np.linalg.norm(A @ x - b, ord=2)

            if self.optim_method == "SLSQP":
                x = _optim_minimize(loss, n_classes=n_classes, method="SLSQP")
            elif self.optim_method == "cvxpy":
                x = _optim_cvxpy(A, b)
            elif self.otpim_method == "lsq_linear":
                x = _optim_lsq_linear(A, b)
            elif self.optim_method == "Adam":
                x = _optim_Adam(A, b)

        else:
            self._sout(".", end="")

        cont_table_test = x.reshape(n, n)

        if self.log_true_solve:
            self._true_solve_log.append([_true_solve])

        return cont_table_test, q_prev_estim


class SLEAP_qerr(ContTableTransferCAP, CAPCTQ_qerr):
    @override
    def predict_ct(self, test, posteriors):
        prev_hat = self.q.quantify(test)
        adjustment = prev_hat / self.train_prev
        return self.cont_table * adjustment[:, np.newaxis], prev_hat


class OLEAP_qerr(OverConstrainedEquationsCAP, CAPCTQ_qerr):
    @override
    def predict_ct(self, test, posteriors):
        n = self.cont_table.shape[1]

        h_label_preds = np.argmax(posteriors, axis=-1)

        cc_prev_estim = prevalence_from_labels(h_label_preds, self.classes_)
        q_prev_estim = self.q.quantify(test)

        A = self.A
        b = self.partial_b

        # b is partially filled; we finish the vector by plugin in the classify and count
        # prevalence estimates (n-1 values only), and the quantification estimates (n-1 values only)
        b[-2 * n : -n] = cc_prev_estim
        b[-n:] = q_prev_estim

        def loss(x):
            return np.linalg.norm(A @ x - b, ord=2)

        n_classes = n**2
        if self.optim_method == "SLSQP":
            x = _optim_minimize(loss, n_classes=n_classes, method=self.optim_method)
        elif self.optim_method == "cvxpy":
            x = _optim_cvxpy(A, b)
        elif self.otpim_method == "lsq_linear":
            x = _optim_lsq_linear(A, b)
        elif self.optim_method == "Adam":
            x = _optim_Adam(A, b)

        cont_table_test = x.reshape(n, n)
        return cont_table_test, q_prev_estim


def gen_classifiers():
    yield "LR", LogisticRegression()


def gen_datasets(only_names=False):
    if env.PROBLEM == "binary":
        _uci_sel = ["pageblocks.5", "yeast", "haberman", "iris.2"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d in _uci_sel]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        for dn in _sorted_uci_names:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif env.PROBLEM == "multiclass":
        return
        yield


def gen_acc_measure():
    yield "vanilla_accuracy", vanilla_acc


def gen_methods(h, D):
    _, acc_fn = next(gen_acc_measure())
    _v = D.V, D.V_posteriors
    _v1 = D.V1, D.V1_posteriors
    yield "LEAP(ACC)", LEAP_qerr(acc_fn, acc(), reuse_h=h, log_true_solve=True), *_v
    yield "LEAP(KDEy-MLP)", LEAP_qerr(acc_fn, kdey(), log_true_solve=True), *_v
    yield "S-LEAP(KDEy-MLP)", SLEAP_qerr(acc_fn, kdey()), *_v
    yield "O-LEAP(KDEy-MLP)", OLEAP_qerr(acc_fn, kdey()), *_v


def get_acc_names():
    return [a for a, _ in gen_acc_measure()]


def get_method_names():
    mock_h = LogisticRegression()
    mock_D = DatasetBundle.mock()
    return [m for m, _, _, _ in gen_methods(mock_h, mock_D)]


def get_dataset_names():
    return [d for d, _ in gen_datasets(only_names=True)]


def get_classifier_names():
    return [c for c, _ in gen_classifiers()]


def get_predictions(method, test_prot, test_prot_posteriors):
    tinit = time()
    if isinstance(method, CAPCTQ_qerr):
        estim_accs, estim_cts, q_errs = [], [], []
        for Ui, P in IT.zip_longest(test_prot(), test_prot_posteriors):
            estim_ct, q_hat = method.predict_ct(Ui.X, P)
            q_err = qp.error.ae(q_hat, Ui.prevalence())
            estim_accs.append(method.acc_fn(estim_ct))
            estim_cts.append(estim_ct)
            q_errs.append(q_err)
    elif isinstance(method, CAPContingencyTable):
        estim_accs, estim_cts = method.batch_predict(test_prot, test_prot_posteriors, get_estim_cts=True)
        q_errs = None
    else:
        estim_accs = method.batch_predict(test_prot, test_prot_posteriors), None
        estim_cts, q_errs = None, None
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, estim_cts, q_errs, t_test_ave


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

            test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()
            estim_accs, estim_cts, q_errs, t_test_ave = get_predictions(method, D.test_prot, D.test_prot_posteriors)
            if estim_cts is None:
                estim_cts = [None] * len(estim_accs)
            else:
                estim_cts = [ct.tolist() for ct in estim_cts]
            if q_errs is None:
                q_errs = [None] * len(estim_accs)
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
            q_errs=q_errs,
            classifier=cls_name,
            method=method_name,
            dataset=dataset_name,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )

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
    if all_exist_pre_check(dataset_name, cls_name, subproject=SUBPROJECT):
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
                    os.path.exists(local_path(dataset_name, cls_name, method_name, acc_name, subproject=SUBPROJECT))
                    for acc_name in get_acc_names()
                ]
            ):
                log.info(f"([{cls_name}@{dataset_name}] {method_name} on all acc measures exists, skipping")
                continue

            exp_prot_args_list.append(
                (cls_name, dataset_name, h, D, true_accs, method_name, method, val, val_posteriors)
            )

    # results_gen = parallel(
    #     func=exp_protocol,
    #     args_list=exp_prot_args_list,
    #     n_jobs=qc.env["N_JOBS"],
    #     return_as="generator_unordered",
    #     max_nbytes=None,
    # )
    results_gen = (exp_protocol(_args) for _args in exp_prot_args_list)

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
