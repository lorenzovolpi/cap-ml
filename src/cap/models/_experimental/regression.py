import itertools as IT
from copy import deepcopy
from time import time

import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import ACC, BaseQuantifier
from quapy.protocol import UPP, AbstractProtocol
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import cap
from cap.models.cont_table import LEAP, QuAcc
from cap.models.direct import ATC, CAPDirect, DoC
from cap.models.model_selection import GridSearchCAP as GSCAP
from cap.models.utils import get_posteriors_from_h, max_conf, max_inverse_softmax, neg_entropy
from cap.utils.commons import parallel as cap_parallel


class ReQua(CAPDirect):
    def __init__(
        self,
        acc_fn,
        reg_model,
        quacc_classes: QuAcc | list[QuAcc],
        param_grid: dict,
        protocol: AbstractProtocol,
        prot_posteriors,
        clip_vals=(0, 1),
        add_conf=False,
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(acc_fn)
        self.param_grid = param_grid
        self.reg_model = reg_model
        self._build_models(quacc_classes)
        self.protocol = protocol
        self.prot_posteriors = prot_posteriors
        self.clip_vals = clip_vals
        self.add_conf = add_conf
        self.n_jobs = cap.commons.get_njobs(n_jobs)
        self.parallel = self.n_jobs != 0
        self.verbose = verbose
        self.joblib_verbose = 10 if verbose else 0

    def _sout(self, msg):
        if self.verbose:
            print(msg)

    def _build_models(self, quacc_classes: QuAcc | list[QuAcc]):
        if isinstance(quacc_classes, list):
            _base_models = [deepcopy(qcc) for qcc in quacc_classes]
        else:
            _base_models = [deepcopy(quacc_classes)]

        grid_product = list(IT.product(*list(self.param_grid.values())))
        quacc_params = [dict(zip(list(self.param_grid.keys()), vs)) for vs in grid_product]

        self.models = []
        for _base_m, grid in IT.product(_base_models, quacc_params):
            m = deepcopy(_base_m)
            m.set_params(**grid)
            self.models.append(m)

    def fit_regression(self, feats_list, accs):
        reg_feats = np.hstack(feats_list)
        self.reg_model.fit(reg_feats, accs)

    def predict_regression(self, feats_list: list[np.ndarray]):
        test_feats = np.hstack(feats_list)
        if test_feats.ndim == 1:
            test_feats = test_feats.reshape(1, -1)
        pred_acc = self.reg_model.predict(test_feats)
        return pred_acc

    def _fit_quacc_models(self, val, posteriors):
        def _fit_model(args):
            m, train, _posteriors = args
            try:
                m = m.fit(train, _posteriors)
                return m, True
            except Exception as _:
                return m, False

        # training models
        models_fit_args = [(m, val, posteriors) for m in self.models]
        if self.parallel:
            outs = cap_parallel(
                _fit_model,
                models_fit_args,
                n_jobs=self.n_jobs,
                seed=qp.environ.get("_R_SEED", None),
                verbose=self.joblib_verbose,
                return_as="array",
            )
            self.models, self.fit_mask = zip(*outs)
        else:
            outs = [_fit_model(arg) for arg in models_fit_args]
            self.models, self.fit_mask = zip(*outs)

    def _get_quacc_feats(self, X, posteriors):
        def predict_cts(m, _X):
            return m.predict_ct(_X, posteriors).flatten()

        cts = np.hstack([predict_cts(m, X) for m, _valid in zip(self.models, self.fit_mask) if _valid])
        return cts

    def _get_batch_quacc_feats(self, prot, posteriors):
        def _predict_model_cts(args):
            m, _prot, _posteriors = args
            cts = np.vstack(
                [m.predict_ct(sigma_i.X, P).flatten() for sigma_i, P in IT.zip_longest(_prot(), _posteriors)]
            )

            return cts

        # predicting v2 sample cont. tables for each model
        models_cts_args = [(m, prot, posteriors) for m, _valid in zip(self.models, self.fit_mask) if _valid]
        if self.parallel:
            v2_ctss = cap_parallel(
                _predict_model_cts,
                models_cts_args,
                n_jobs=self.n_jobs,
                seed=qp.environ.get("_R_SEED", None),
                return_as="list",
                verbose=self.joblib_verbose,
                batch_size=round(len(self.models) / (self.n_jobs * 2)),
            )
        else:
            v2_ctss = [_predict_model_cts(arg) for arg in models_cts_args]

        v2_ctss = np.hstack(v2_ctss)

        return v2_ctss

    def _get_linear_feats(self, X, posteriors):
        conf_fns = [
            max_conf,
            neg_entropy,
            max_inverse_softmax,
        ]
        lin_feats = np.hstack([fn(posteriors, keepdims=True) for fn in conf_fns]).mean(axis=0)
        return lin_feats

    def _get_batch_linear_feats(self, prot, posteriors):
        lin_feats = np.vstack(
            [self._get_linear_feats(sigma_i.X, P) for sigma_i, P in IT.zip_longest(prot(), posteriors)]
        )
        return lin_feats

    def true_acc(self, sample: LabelledCollection, posteriors):
        if posteriors is None:
            posteriors = get_posteriors_from_h(self.h, sample.X)
        y_pred = np.argmax(posteriors, axis=-1)
        y_true = sample.y
        conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sample.classes_)
        return self.acc(conf_table)

    def fit(self, val: LabelledCollection, val_posteriors):
        t_fit_init = time()
        self._fit_quacc_models(val, val_posteriors)
        self._sout(f"trained quacc models: {np.sum(self.fit_mask)}/{len(self.models)}")
        self._sout(f"training quacc models took {time() - t_fit_init:.3f}s")

        # compute features to train the regressor

        features = []

        t_ct_init = time()
        quacc_feats = self._get_batch_quacc_feats(self.protocol, self.prot_posteriors)
        features.append(quacc_feats)
        self._sout(f"generating quacc features took {time() - t_ct_init:.3f}s")

        if self.add_conf:
            t_lin_init = time()
            lin_feats = self._get_batch_linear_feats(self.protocol, self.prot_posteriors)
            features.append(lin_feats)
            self._sout(f"generating linear features took {time() - t_lin_init:.3f}s")

        # compute true accs as targets for the regressor

        v2_accs = np.asarray(
            [self.true_acc(v2_i, P) for v2_i, P in IT.zip_longest(self.protocol(), self.prot_posteriors)]
        )

        # train regression model

        t_init = time()
        self.fit_regression(features, v2_accs)
        self._sout(f"training reg_model took {time() - t_init:.3f}s")

        return self

    def predict(self, X, posteriors, oracle_prev=None) -> float:
        features = []
        features.append(self._get_quacc_feats(X, posteriors))
        if self.add_conf:
            features.append(self._get_linear_feats(X, posteriors))

        acc_pred = self.predict_regression(features)

        if self.clip_vals is not None:
            acc_pred = np.clip(acc_pred, *self.clip_vals)
        return acc_pred[0]

    def batch_predict(self, prot: AbstractProtocol, posteriors, oracle_prevs=None) -> list[float]:
        t_bpred_init = time()

        features = []

        quacc_feats = self._get_batch_quacc_feats(prot, posteriors)
        features.append(quacc_feats)

        if self.add_conf:
            lin_feats = self._get_batch_linear_feats(prot, posteriors)
            features.append(lin_feats)

        acc_pred = self.predict_regression(features)
        self._sout(f"batch prediction took {time() - t_bpred_init:.3f}s")

        if self.clip_vals is not None:
            acc_pred = np.clip(acc_pred, *self.clip_vals)
        return acc_pred.tolist()


# TODO: adapt to fit with posteriors change
class reDAN(CAPDirect):
    def __init__(
        self,
        h,
        acc_fn,
        reg_model,
        q_class: BaseQuantifier,
        sample_size,
        q_params: dict | None = None,
        n_val_samples=500,
        n2e_opt_h0=True,
        add_conf=False,
        val_prop=0.5,
        clip_vals=(0, 1),
        n_jobs=None,
        verbose=False,
    ):
        super().__init__(h, acc_fn)
        self.reg_model = reg_model
        self.q_class = q_class
        self.q_params = q_params
        self.sample_size = sample_size
        self.n_val_samples = n_val_samples
        self.n2e_opt_h0 = n2e_opt_h0
        self.add_conf = add_conf
        self.val_prop = val_prop
        self.clip_vals = clip_vals
        self.n_jobs = cap.commons.get_njobs(n_jobs)
        self.verbose = verbose
        self.joblib_verbose = 10 if verbose else 0

    def _fit_models(self, val: LabelledCollection):
        n2e_acc = LEAP(self.h, self.acc, ACC(LogisticRegression()), reuse_h=True).fit(val)
        doc = DoC(self.h, self.acc, self.sample_size).fit(val)
        atc = ATC(self.h, self.acc, scoring_fn="maxconf").fit(val)

        v11, v12 = val.split_stratified(self.val_prop, random_state=qp.environ["_R_SEED"])
        v_prot = UPP(
            v12,
            self.sample_size,
            repeats=100,
            random_state=qp.environ["_R_SEED"],
            return_type="labelled_collection",
        )
        _n2e_q = LEAP(self.h, self.acc, self.q_class, reuse_h=self.n2e_opt_h0)
        _params = None if self.q_params is None else {("q_class__" + k, v) for k, v in self.q_params.items()}
        n2e_opt = _n2e_q.fit(val) if _params is None else GSCAP(_n2e_q, _params, v_prot, self.acc).fit(v11)
        self.models = [n2e_acc, n2e_opt, doc, atc]

    def _get_models_feats(self, X):
        feats = np.hstack([m.predict(X) for m in self.models])
        if self.add_conf:
            P = get_posteriors_from_h(self.h, X)
            conf_fns = [max_conf, neg_entropy, max_inverse_softmax]
            conf_feats = np.hstack([fn(P, keepdims=True) for fn in conf_fns]).mean(axis=0)
            feats = np.hstack([feats, conf_feats])

        return feats

    def fit_regression(self, feats, accs):
        self.reg_model.fit(feats, accs)

    def predict_regression(self, feats):
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        pred_acc = self.reg_model.predict(feats)
        return pred_acc

    def fit(self, val: LabelledCollection):
        v2, v1 = val.split_stratified(train_prop=self.val_prop)

        v2_prot = UPP(
            v2,
            sample_size=self.sample_size,
            repeats=self.n_val_samples,
            return_type="labelled_collection",
        )

        self._fit_models(v1)
        feats = np.vstack([self._get_models_feats(Ui.X) for Ui in v2_prot()])

        v2_accs = np.asarray([self.true_acc(v2_i) for v2_i in v2_prot()])

        self.reg_model.fit(feats, v2_accs)

        return self

    def predict(self, X, oracle_prev=None) -> float:
        feats = self._get_models_feats(X)
        acc_pred = self.predict_regression(feats)

        if self.clip_vals is not None:
            acc_pred = np.clip(acc_pred, *self.clip_vals)
        return acc_pred[0]
