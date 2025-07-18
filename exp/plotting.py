import itertools as IT
import os

import numpy as np
import pandas as pd

import quacc as qc
from quacc.exp.generators import gen_acc_measure, gen_bin_datasets
from quacc.exp.report import Report

PROBLEM = "binary"
basedir = PROBLEM
plots_basedir = PROBLEM
root_folder = os.path.join(qc.env["OUT_DIR"], "results")

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets


def rename_methods(df: pd.DataFrame, methods: dict):
    for om, nm in methods.items():
        df.loc[df["method"] == om, "method"] = nm


def plot_grid_of_diagonals(methods, dataset_names, classifiers, filename=None, n_cols=5, **kwargs):
    for cls_name, (acc_name, _) in IT.product(classifiers, gen_acc_measure()):
        rep = Report.load_results(
            root_folder, basedir, cls_name, acc_name, datasets=dataset_names, methods=list(methods.keys())
        )
        df = rep.table_data(mean=False)
        rename_methods(df, methods)
        qc.plot.seaborn.plot_diagonal_grid(
            df,
            cls_name,
            acc_name,
            dataset_names,
            basedir=plots_basedir,
            n_cols=n_cols,
            x_label="True Accuracy",
            y_label="Estimated Accuracy",
            filename=f"{PROBLEM}_{filename}" if filename else PROBLEM,
            **kwargs,
        )
        print(f"{cls_name}-{acc_name} plots generated")


if __name__ == "__main__":
    methods = {
        "ATC-MC": "ATC",
        "DoC": "DoC",
        # "Naive",
        "N2E(ACC-h0)": "P$h$D",
        "N2E(KDEy-h0)": "P$h$D*",
    }
    selected_datasets = ["sonar", "haberman", "cmc.2", "german", "iris.2"]
    plot_grid_of_diagonals(
        methods,
        selected_datasets,
        ["LR"],
        filename="5x1",
        n_cols=5,
        legend_bbox_to_anchor=(0.96, 0.3),
        legend_wspace=0.08,
        xtick_vert=True,
        aspect=0.8,
        xticks=np.linspace(0, 1, 6, endpoint=True),
        yticks=np.linspace(0, 1, 6, endpoint=True),
    )

    all_datasets = [name for name, _ in gen_datasets(only_names=True)]
    classifiers = ["LR", "KNN_10", "SVM(rbf)", "MLP"]
    plot_grid_of_diagonals(
        methods, all_datasets, classifiers, filename="all", n_cols=5, legend_bbox_to_anchor=(0.84, 0.06)
    )
