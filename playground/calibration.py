import numpy as np
import pandas as pd
import quapy as qp
from pandas.core.internals.blocks import shift
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression

# from abstention.calibration import TempScaling
from quacc.calibration import TS
from quacc.calibration.bcts import BCTS
from quacc.calibration.error import calibration_error
from quacc.data.datasets import fetch_UCIBinaryDataset
from quacc.error import vanilla_acc
from quacc.models.cont_table import QuAcc1xN2

dataset_name = "haberman"

qp.environ["_R_SEED"] = 0


def lblM_from_P(P, n_classes):
    return np.eye(n_classes)[np.argmax(P, axis=-1)]


def label_shift_calibration(V_P_recalib, test_P, calib_fn, shift_estimator=None):
    if shift_estimator is None:
        return calib_fn(test_P)
    elif shift_estimator == "em":
        V_prior_recalib = np.sum(V_P_recalib, axis=0) / V_P_recalib.shape[0]
        _, test_posteriors_recalib = EMQ.EM(V_prior_recalib, test_P)
        return test_posteriors_recalib


def calib_testing():
    ts = TS(verbose=False)
    bcts = BCTS(verbose=False)

    L, V, U = fetch_UCIBinaryDataset(dataset_name)
    h = LogisticRegression().fit(*L.Xy)

    V_P = h.predict_proba(V.X)
    ts_fn = ts(V_P, lblM_from_P(V_P, V.n_classes), posterior_supplied=True)
    bcts_fn = bcts(V_P, lblM_from_P(V_P, V.n_classes), posterior_supplied=True)
    V_P_ts = ts_fn(V_P)
    V_P_bcts = bcts_fn(V_P)

    test_prot = UPP(U, sample_size=100, repeats=10, return_type="labelled_collection")
    ce_base, ce_ts, ce_bcts = [], [], []
    for U_i in test_prot():
        P_i = h.predict_proba(U_i.X)
        ts_Pi = label_shift_calibration(V_P_ts, P_i, ts_fn)
        bcts_Pi = label_shift_calibration(V_P_bcts, P_i, bcts_fn, shift_estimator="em")
        ce_base.append(calibration_error(P_i, U_i.y, norm="l2"))
        ce_ts.append(calibration_error(ts_Pi, U_i.y, norm="l2"))
        ce_bcts.append(calibration_error(bcts_Pi, U_i.y, norm="l2"))
    ce_base, ce_ts, ce_bcts = np.array(ce_base), np.array(ce_ts), np.array(ce_bcts)

    df = pd.DataFrame(np.hstack([ce_base[:, None], ce_ts[:, None], ce_bcts[:, None]]), columns=["uncal", "ts", "bcts"])
    print(df)


def recalib_quacc():
    dfs = []
    for dataset_name in UCI_BINARY_DATASETS:
        L, V, U = fetch_UCIBinaryDataset(dataset_name)
        h = LogisticRegression().fit(*L.Xy)

        V_P = h.predict_proba(V.X)

        q = QuAcc1xN2(vanilla_acc, KDEyML(), add_maxinfsoft=False).fit(V, V_P)

        test_prot = UPP(U, sample_size=100, return_type="labelled_collection")
        test_P = [h.predict_proba(U_i.X) for U_i in test_prot()]
        true_acc = np.array([vanilla_acc(np.argmax(P_i, axis=-1), U_i.y) for P_i, U_i in zip(test_P, test_prot())])

        def add_df(P, method):
            _preds = np.array([q.predict(U_i.X, P_i) for U_i, P_i in zip(test_prot(), P)])
            _acc_err = np.abs(_preds - true_acc)[:, None]
            _df = pd.DataFrame(_acc_err, columns=["acc_err"])
            _df["dataset"] = dataset_name
            _df["method"] = method
            dfs.append(_df)

        add_df(test_P, "base")

        test_P_r = [post for _, post in [EMQ.EM(V.prevalence(), P_i) for P_i in test_P]]
        add_df(test_P_r, "em")

        ts = TS()(V_P, lblM_from_P(V_P, V.n_classes), posterior_supplied=True)
        test_P_r = [ts(P_i) for P_i in test_P]
        add_df(test_P_r, "ts")

        print(dataset_name)

    df = pd.concat(dfs, axis=0)
    # print(df.groupby(["dataset"]).mean())
    print(df.pivot_table(index="dataset", columns="method", values="acc_err"))


if __name__ == "__main__":
    calib_testing()
    # recalib_quacc()
