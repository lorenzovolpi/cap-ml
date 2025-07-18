from collections import defaultdict

import numpy as np
import pandas as pd
import quapy as qp
from quapy.method.aggregative import SLD
from quapy.protocol import UPP
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP
from tqdm import tqdm

from quacc.data.datasets import HF_DATASETS
from quacc.error import vanilla_acc
from quacc.experiments.generators import gen_bin_lm_datasets
from quacc.experiments.util import split_validation
from quacc.models._large_models import DistilBert
from quacc.models.cont_table import QuAcc1xN2, QuAcc1xNp1, QuAccNxN
from quacc.models.direct import DoC
from quacc.models.model_selection import GridSearchCAP
from quacc.models.regression import ReQua

NUM_SAMPLES = 100
SAMPLE_SIZE = 500
RANDOM_STATE = 42

qp.environ["_R_SEED"] = RANDOM_STATE
qp.environ["SAMPLE_SIZE"] = SAMPLE_SIZE


def mlp():
    MLP((100, 15), activation="logistic", solver="adam")


def sld():
    _sld = SLD(LogisticRegression(), val_split=5)
    _sld.SUPPRESS_WARNINGS = True
    return _sld


def main():
    BASE = True
    OPT_N2 = True
    OPT_NN = True
    REQUA = True

    model = DistilBert()

    results = defaultdict(lambda: [])
    for dataset_name, (L, V, U) in gen_bin_lm_datasets(model.tokenizer, model.data_collator):
        print("-" * 10, dataset_name, "-" * 10)

        model.fit(L, dataset_name)

        test_prot = UPP(U, sample_size=SAMPLE_SIZE, repeats=NUM_SAMPLES, return_type="labelled_collection")

        V1, V2_prot = split_validation(V)

        V_posteriors = model.predict_proba(V.X, V.attention_mask, verbose=True)
        print("V_posteriors")
        V1_posteriors = model.predict_proba(V1.X, V1.attention_mask, verbose=True)
        print("V1_posteriors")
        V2_prot_posteriors = []
        for sample in tqdm(V2_prot(), total=V2_prot.total()):
            V2_prot_posteriors.append(model.predict_proba(sample.X, sample.attention_mask))
        print("V2_prot_posteriors")

        sld_requa_params = {
            "add_X": [False],
            "add_posteriors": [True, False],
            "add_y_hat": [True, False],
            "add_maxconf": [True, False],
            "add_negentropy": [True, False],
            "add_maxinfsoft": [True, False],
        }

        sld_opt_params = sld_requa_params | {
            "q_class__classifier__C": np.logspace(-3, 3, 7),
            "q_class__classifier__class_weight": [None, "balanced"],
            "q_class__recalib": [None, "bcts"],
        }

        if BASE:
            quacc_n2 = QuAcc1xN2(
                vanilla_acc,
                sld(),
                add_X=False,
                add_y_hat=True,
                add_maxinfsoft=True,
            ).fit(V, V_posteriors)
            print("quacc_n2 fit")
            quacc_nn = QuAccNxN(
                vanilla_acc,
                sld(),
                add_X=False,
                add_y_hat=True,
                add_maxinfsoft=True,
            ).fit(V, V_posteriors)
            print("quacc_nn fit")
        if OPT_NN:
            quacc_nn_opt = GridSearchCAP(
                QuAccNxN(vanilla_acc, sld()), sld_opt_params, V2_prot, V2_prot_posteriors, vanilla_acc, refit=False
            ).fit(V1, V1_posteriors)
            print("quacc_nn_opt fit")
        if OPT_N2:
            quacc_n2_opt = GridSearchCAP(
                QuAcc1xN2(vanilla_acc, sld()), sld_opt_params, V2_prot, V2_prot_posteriors, vanilla_acc, refit=False
            ).fit(V1, V1_posteriors)
            print("quacc_n2_opt fit")
        if REQUA:
            requa = ReQua(
                vanilla_acc,
                KRR(),
                [QuAcc1xN2(vanilla_acc, sld()), QuAccNxN(vanilla_acc, sld()), QuAcc1xNp1(vanilla_acc, sld())],
                sld_requa_params,
                V2_prot,
                V2_prot_posteriors,
                n_jobs=0,
            ).fit(V1, V1_posteriors)
            print("requa fit")

        doc = DoC(vanilla_acc, V2_prot, V2_prot_posteriors).fit(V1, V1_posteriors)
        print("doc fit")

        test_y_hat, test_y = [], []
        if BASE:
            quacc_n2_accs = []
            quacc_nn_accs = []
        if OPT_NN:
            quacc_nn_opt_accs = []
        if OPT_N2:
            quacc_n2_opt_accs = []
        if REQUA:
            requa_accs = []
        doc_accs = []
        true_accs = []
        for i, U_i in enumerate(tqdm(test_prot(), total=test_prot.total())):
            P = model.predict_proba(U_i.X, U_i.attention_mask)
            y_hat = np.argmax(P, axis=-1)
            test_y_hat.append(y_hat)
            test_y.append(U_i.y)
            if BASE:
                quacc_nn_accs.append(quacc_nn.predict(U_i.X, P))
                # print(f"quacc_nn prediction #{i}")
                quacc_n2_accs.append(quacc_n2.predict(U_i.X, P))
                # print(f"quacc_n2 prediction #{i}")
            if OPT_NN:
                quacc_nn_opt_accs.append(quacc_nn_opt.predict(U_i.X, P))
                # print(f"quacc_nn_opt prediction #{i}")
            if OPT_N2:
                quacc_n2_opt_accs.append(quacc_n2_opt.predict(U_i.X, P))
                # print(f"quacc_n2_opt prediction #{i}")
            if REQUA:
                requa_accs.append(requa.predict(U_i.X, P))
                # print(f"requa prediction #{i}")
            doc_accs.append(doc.predict(U_i.X, P))
            # print(f"doc prediction #{i}")
            true_accs.append(vanilla_acc(y_hat, U_i.y))

        if BASE:
            quacc_n2_accs = np.asarray(quacc_n2_accs)
            quacc_nn_accs = np.asarray(quacc_nn_accs)
            quacc_n2_mean = np.mean(np.abs(quacc_n2_accs - true_accs))
            quacc_nn_mean = np.mean(np.abs(quacc_nn_accs - true_accs))
            results["quacc_nn"].append(quacc_nn_mean)
            results["quacc_n2"].append(quacc_n2_mean)
        if OPT_NN:
            quacc_nn_opt_accs = np.asarray(quacc_nn_opt_accs)
            quacc_nn_opt_mean = np.mean(np.abs(quacc_nn_opt_accs - true_accs))
            results["quacc_nn_opt"].append(quacc_nn_opt_mean)
        if OPT_N2:
            quacc_n2_opt_accs = np.asarray(quacc_n2_opt_accs)
            quacc_n2_opt_mean = np.mean(np.abs(quacc_n2_opt_accs - true_accs))
            results["quacc_n2_opt"].append(quacc_n2_opt_mean)
        if REQUA:
            requa_accs = np.asarray(requa_accs)
            requa_mean = np.mean(np.abs(requa_accs - true_accs))
            results["requa"].append(requa_mean)

        doc_accs = np.asarray(doc_accs)
        doc_mean = np.mean(np.abs(doc_accs - true_accs))
        results["doc"].append(doc_mean)

    df = pd.DataFrame(np.vstack(list(results.values())), columns=HF_DATASETS, index=list(results.keys()))
    return df


if __name__ == "__main__":
    res = main()
    print(res)
