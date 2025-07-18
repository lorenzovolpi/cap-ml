import itertools as IT
import os
import pathlib

from critdd import Diagram
from quapy.data.datasets import UCI_BINARY_DATASETS

import quacc as qc
from exp.report import Report

PROBLEM = "binary"
root_folder = os.path.join(qc.env["OUT_DIR"], "results")


def get_acc_names():
    yield "vanilla_accuracy"


def get_dataset_names():
    if PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        return [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]


def get_classifier_names():
    return ["LR", "SVM(rbf)", "KNN_10", "MLP"]


def get_method_names():
    return [
        "DoC",
        "LEAP(SLD)",
        "OCE(SLD)",
        "PHD(SLD)",
        # "ReQua(SLD-KRR)",
        # "ReQua(SLD-Ridge)",
    ]


def get_path(acc_name, cls_name, ext="pdf"):
    basedir = os.path.join("output", "critdd", PROBLEM)
    os.makedirs(basedir, exist_ok=True)
    return os.path.join(basedir, f"critdd_{acc_name}_{cls_name}.{ext}")


def clean_aux_files(acc_name, cls_name):
    log_path = get_path(acc_name, cls_name, ext="log")
    aux_path = get_path(acc_name, cls_name, ext="aux")
    pathlib.Path.unlink(log_path)
    pathlib.Path.unlink(aux_path)


def plotting():
    for acc_name, cls_name in IT.product(get_acc_names(), get_classifier_names()):
        datasets = get_dataset_names()
        methods = get_method_names()

        df = (
            Report.load_results(root_folder, PROBLEM, cls_name, acc_name, datasets, methods)
            .table_data()
            .pivot(index="dataset", columns="method", values="acc_err")
        )
        diagram = Diagram(df.to_numpy(), treatment_names=df.columns)

        path = get_path(acc_name, cls_name)
        diagram.to_file(
            path,
            alpha=0.05,
            adjustment="holm",
            reverse_x=True,
            axis_options={"title": "critdd"},
        )
        clean_aux_files(acc_name, cls_name)


if __name__ == "__main__":
    plotting()
