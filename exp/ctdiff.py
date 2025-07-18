import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import quapy as qp
import seaborn as sns
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import EMQ, KDEyML
from quapy.protocol import UPP
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

from exp.util import get_logger, split_validation
from quacc.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from quacc.error import vanilla_acc
from quacc.models.cont_table import LEAP, PHD, CAPContingencyTable, LabelledCollection
from quacc.utils.commons import contingency_table

TRUE_CTS_NAME = "true_cts"
PROBLEM = "binary"
MODEL_TYPE = "simple"

log = get_logger(id="ctdiff")

qp.environ["_R_SEED"] = 0

if PROBLEM == "binary":
    qp.environ["SAMPLE_SIZE"] = 100
elif PROBLEM == "multiclass":
    qp.environ["SAMPLE_SIZE"] = 100


def sld():
    return EMQ(LR(), val_split=5)


def kdey():
    return KDEyML(LR())


class PredictedSet:
    def __init__(self, set, posteriors):
        self.A = set
        self.post = posteriors


def gen_classifiers():
    if MODEL_TYPE == "simple":
        yield "LR", LR()
        yield "KNN_10", KNN(n_neighbors=10)
        yield "SVM(rbf)", SVC(probability=True)
        yield "RFC", RFC()
        yield "MLP", MLP(hidden_layer_sizes=(100, 15), max_iter=300, random_state=0)


def gen_datasets() -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    if PROBLEM == "binary":
        if MODEL_TYPE == "simple":
            _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
            _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
            for dn in _uci_names:
                yield dn, fetch_UCIBinaryDataset(dn)
    elif PROBLEM == "multiclass":
        if MODEL_TYPE == "simple":
            _uci_skip = ["isolet", "wine-quality", "letter"]
            _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
            for dataset_name in _uci_names:
                yield dataset_name, fetch_UCIMulticlassDataset(dataset_name)


def gen_methods(h, V_ps, V1_ps, V2_prot_ps):
    acc_fn = vanilla_acc
    yield "LEAP(SLD)", LEAP(acc_fn, sld(), reuse_h=h), V_ps
    yield "PHD(SLD)", PHD(acc_fn, sld(), reuse_h=h), V_ps
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h), V_ps
    yield "PHD(KDEy)", PHD(acc_fn, kdey(), reuse_h=h), V_ps


def get_method_names():
    mock_h = LR()
    for method_name, _, _ in gen_methods(mock_h, None, None, None):
        yield method_name
    yield TRUE_CTS_NAME


def gen_classifier_dataset():
    for classifier in gen_classifiers():
        for dataset in gen_datasets():
            yield classifier, dataset


def get_cts(method: CAPContingencyTable, test_prot, test_prot_posteriors):
    cts_list = []
    for sample, sample_post in zip(test_prot(), test_prot_posteriors):
        cts_list.append(method.predict_ct(sample.X, sample_post))
    return np.asarray(cts_list)


def get_cts_diff_mean(cts1, cts2):
    cts_diff = np.abs(cts1 - cts2)
    return np.mean(cts_diff, axis=0)


def get_cts_mean(cts):
    return np.mean(cts, axis=0)


def get_cts_bias_median(cts1, cts2):
    cts_bias = cts1 - cts2
    return np.median(cts_bias, axis=0)


def save_heatmap(cls_name, dataset, method1, method2, ctss):
    def map_heatmap(*args, **kwargs):
        data = kwargs.pop("data")
        plot_name = data["plot"].to_numpy()[0]
        vmin, vmax = data["vmin"].to_numpy()[0], data["vmax"].to_numpy()[0]
        cmap = data["cmap"].to_numpy()[0]
        data = data.drop(["plot", "vmin", "vmax", "col", "row", "cmap"], axis=1).to_numpy()
        cbar = plot_name.startswith(method2)
        plot = sns.heatmap(data, cbar=cbar, vmin=vmin, vmax=vmax, **kwargs, cmap=cmap)
        plot.set_title(plot_name)

    _diff = np.vstack([ctss["compare"][(method1, method2)], ctss["ae"][method1], ctss["ae"][method2]])
    df_diff = pd.DataFrame(_diff)
    df_diff["col"] = np.repeat(np.arange(3), _diff.shape[1])
    df_diff["row"] = "diff cts"
    df_diff["plot"] = np.repeat([f"{method1} - {method2}", f"{method1} - true", f"{method2} - true"], _diff.shape[1])
    df_diff["vmin"] = np.min(_diff)
    df_diff["vmax"] = np.max(_diff)
    df_diff["cmap"] = "rocket_r"

    _true = np.vstack([ctss["true"][TRUE_CTS_NAME], ctss["true"][method1], ctss["true"][method2]])
    df_true = pd.DataFrame(_true)
    df_true["col"] = np.repeat(np.arange(3), _true.shape[1])
    df_true["row"] = "true cts"
    df_true["plot"] = np.repeat(["true", method1, method2], _true.shape[1])
    df_true["vmin"] = np.min(_true)
    df_true["vmax"] = np.max(_true)
    df_true["cmap"] = "mako_r"

    _bias = np.vstack([ctss["comp_bias"][(method1, method2)], ctss["bias"][method1], ctss["bias"][method2]])
    df_bias = pd.DataFrame(_bias)
    df_bias["col"] = np.repeat(np.arange(3), _bias.shape[1])
    df_bias["row"] = "bias cts"
    df_bias["plot"] = np.repeat([f"{method1} - {method2} bias", f"{method1} bias", f"{method2} bias"], _bias.shape[1])
    df_bias["vmin"] = np.min(_bias)
    df_bias["vmax"] = np.max(_bias)
    df_bias["cmap"] = "flare"

    df = pd.concat([df_diff, df_true, df_bias])

    plot = sns.FacetGrid(df, col="col", row="row")
    plot.map_dataframe(map_heatmap, annot=_diff.shape[1] <= 4)

    parent_dir = os.path.join("plots", "ctdiff", PROBLEM, cls_name)
    os.makedirs(parent_dir, exist_ok=True)
    fig_path = os.path.join(parent_dir, f"{dataset}_{method1}+{method2}.png")
    plot.figure.savefig(fig_path)
    plot.figure.clear()


def get_parent_dir(cls_name, dataset_name):
    return os.path.join("output", "ctdiff", PROBLEM, cls_name, dataset_name)


def save_json(cls_name, dataset_name, method_name, estim_cts):
    parent_dir = get_parent_dir(cls_name, dataset_name)
    os.makedirs(parent_dir, exist_ok=True)
    path = os.path.join(parent_dir, f"{method_name}.json")
    with open(path, "w") as f:
        json.dump({"estim": estim_cts.tolist()}, f)


def load_json(cls_name, dataset_name, method_name):
    parent_dir = get_parent_dir(cls_name, dataset_name)
    path = os.path.join(parent_dir, f"{method_name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return np.asarray(json.load(f)["estim"])


def all_exist_pre_check(cls_name, dataset_name):
    all_exist = True
    for method_name in get_method_names():
        path = os.path.join(get_parent_dir(cls_name, dataset_name), f"{method_name}.json")
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def ctdiff():
    NUM_TEST = 1000

    _comparing_methods = {
        ("LEAP(SLD)", "PHD(SLD)"),
        ("LEAP(KDEy)", "PHD(KDEy)"),
    }

    log.info("-" * 31 + "  start  " + "-" * 31)

    cts = defaultdict(lambda: {})
    for (cls_name, h), (dataset_name, (L, V, U)) in gen_classifier_dataset():
        if all_exist_pre_check(cls_name, dataset_name):
            log.info(f"{cls_name} on dataset={dataset_name}: all results already exist, skipping")
            for method_name in get_method_names():
                cts[(cls_name, dataset_name)][method_name] = load_json(cls_name, dataset_name, method_name)
            continue

        log.info(f"{cls_name} training on dataset={dataset_name}")
        h.fit(*L.Xy)

        # test generation protocol
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)

        # split validation set
        V1, V2_prot = split_validation(V)

        # precomumpute model posteriors for validation and test sets
        V_ps = PredictedSet(V, h.predict_proba(V.X))
        V1_ps = PredictedSet(V1, h.predict_proba(V1.X))
        V2_prot_ps = PredictedSet(V2_prot, [h.predict_proba(sample.X) for sample in V2_prot()])

        test_prot_posteriors, test_prot_y_hat, true_prot_cts = [], [], []
        for sample in test_prot():
            P = h.predict_proba(sample.X)
            y_hat = np.argmax(P, axis=-1)
            true_prot_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))
            test_prot_posteriors.append(P)
            test_prot_y_hat.append(y_hat)

        true_prot_cts = load_json(cls_name, dataset_name, TRUE_CTS_NAME)
        if true_prot_cts is None:
            true_prot_cts = np.asarray(
                [
                    contingency_table(sample.y, y_hat, sample.n_classes)
                    for sample, y_hat in zip(test_prot(), test_prot_y_hat)
                ]
            )
            save_json(cls_name, dataset_name, TRUE_CTS_NAME, true_prot_cts)
            log.info(f"{TRUE_CTS_NAME} done")
        else:
            log.info(f"{TRUE_CTS_NAME} exists, skipping")
        cts[(cls_name, dataset_name)][TRUE_CTS_NAME] = true_prot_cts

        for method_name, method, val_ps in gen_methods(h, V_ps, V1_ps, V2_prot_ps):
            loaded_cts = load_json(cls_name, dataset_name, method_name)
            if loaded_cts is not None:
                cts[(cls_name, dataset_name)][method_name] = loaded_cts
                log.info(f"{method_name} exists, skipping")
                continue

            val, val_posteriors = val_ps.A, val_ps.post
            method.fit(val, val_posteriors)
            estim_cts = get_cts(method, test_prot, test_prot_posteriors)
            cts[(cls_name, dataset_name)][method_name] = estim_cts
            save_json(cls_name, dataset_name, method_name, estim_cts)
            log.info(f"{method_name} done")

    results = {}
    for (cls_name, dataset_name), data in cts.items():
        compare_cts, comp_bias_cts, bias_cts, ae_cts = {}, {}, {}, {}
        true_cts = {TRUE_CTS_NAME: get_cts_mean(data[TRUE_CTS_NAME])}
        for method1, method2 in _comparing_methods:
            if method1 == method2 or (method1, method2) in compare_cts or (method2, method1) in compare_cts:
                continue
            compare_cts[(method1, method2)] = get_cts_diff_mean(data[method1], data[method2])
            comp_bias_cts[(method1, method2)] = get_cts_bias_median(data[method1], data[method2])
            ae_cts[method1] = get_cts_diff_mean(data[method1], data[TRUE_CTS_NAME])
            ae_cts[method2] = get_cts_diff_mean(data[method2], data[TRUE_CTS_NAME])
            bias_cts[method1] = get_cts_bias_median(data[method1], data[TRUE_CTS_NAME])
            bias_cts[method2] = get_cts_bias_median(data[method2], data[TRUE_CTS_NAME])
            true_cts[method1] = get_cts_mean(data[method1])
            true_cts[method2] = get_cts_mean(data[method2])

        results[(cls_name, dataset_name)] = {
            "compare": compare_cts,
            "comp_bias": comp_bias_cts,
            "ae": ae_cts,
            "bias": bias_cts,
            "true": true_cts,
        }
        log.info(f"{cls_name} on dataset={dataset_name}: diffs generated")

    for (cls_name, dataset_name), data in results.items():
        for (method1, method2), ctss in compare_cts.items():
            save_heatmap(cls_name, dataset_name, method1, method2, data)
        log.info(f"{cls_name} on dataset={dataset_name}: plots generated")

    log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    ctdiff()
