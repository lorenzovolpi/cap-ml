import itertools as IT
import math
import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, MultipleLocator

import exp.leap.env as env
from exp.leap.config import get_acc_names, get_classifier_names, get_dataset_names
from exp.leap.util import load_results, rename_datasets, rename_methods
from quacc.plot.utils import get_binned_values, save_figure

method_map = {
    # "Naive": 'Na\\"ive',
    "ATC-MC": "ATC",
    "CBPE": "L-CBPE",
    "LEAP(ACC)": "LEAP$_{\\mathrm{ACC}}$",
    "LEAP(KDEy-MLP)": "LEAP$_{\\mathrm{KDEy}}$",
    "S-LEAP(KDEy-MLP)": "S-LEAP$_{\\mathrm{KDEy}}$",
    "O-LEAP(KDEy-MLP)": "O-LEAP$_{\\mathrm{KDEy}}$",
    "O-LEAP(CC-MLP)": "O-LEAP$_{\\mathrm{CC}}$",
    "O-LEAP(oracle)": "O-LEAP$_{\\Phi}$",
    "LEAP(KDEy-MLP)-SLSQP": "LEAP$_{\\mathrm{KDEy}}^{\\mathrm{SLSQP}}$",
    "O-LEAP(KDEy-MLP)-SLSQP": "O-LEAP$_{\\mathrm{KDEy}}^{\\mathrm{SLSQP}}$",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}

acc_map = {
    "vanilla_accuracy": "Vanilla Acc.",
}


def get_selection_datasets():
    if env.PROBLEM == "binary":
        return ["pageblocks.5", "yeast", "haberman", "iris.2"]
    elif env.PROBLEM == "multiclass":
        return ["poker_hand", "shuttle", "page_block", "phishing"]


def get_palette(methods=None):
    base_palette = sns.color_palette("bright", 10)
    _dict = {
        "DoC": 0,
        "LEAP(ACC)": 6,
        "LEAP(KDEy-MLP)": 1,
        "S-LEAP(KDEy-MLP)": 2,
        "O-LEAP(KDEy-MLP)": 3,
        "O-LEAP(CC-MLP)": 4,
        "O-LEAP(oracle)": 8,
    }
    if methods is None:
        return {method_map.get(k, k): base_palette[v] for k, v in _dict.items()}
    else:
        return {method_map.get(k, k): base_palette[v] for k, v in _dict.items() if method_map.get(k, k) in methods}


def plots():
    all_classifiers = get_classifier_names()
    accs = ["vanilla_accuracy"]
    main_methods = ["DoC", "LEAP(KDEy-MLP)", "S-LEAP(KDEy-MLP)", "O-LEAP(KDEy-MLP)"]
    oracle_methods = ["O-LEAP(CC-MLP)", "O-LEAP(KDEy-MLP)", "O-LEAP(oracle)"]

    configs = [
        # {
        #     "name": "all",
        #     "datasets": get_dataset_names(),
        #     "methods": main_methods,
        #     "classifiers": all_classifiers,
        # },
        {
            "name": "4x1",
            "datasets": get_selection_datasets(),
            "methods": main_methods,
            "classifiers": ["LR"],
        },
        # {
        #     "name": "all_oracle",
        #     "datasets": get_dataset_names(),
        #     "methods": oracle_methods,
        #     "classifiers": all_classifiers,
        # },
        {
            "name": "4x1_oracle",
            "datasets": get_selection_datasets(),
            "methods": oracle_methods,
            "classifiers": ["LR"],
        },
    ]

    parent_dir = os.path.join(env.root_dir, "plots")
    os.makedirs(parent_dir, exist_ok=True)

    for _cfg, acc in IT.product(configs, accs):
        name, classifiers, datasets, methods = _cfg["name"], _cfg["classifiers"], _cfg["datasets"], _cfg["methods"]
        for cls_name in classifiers:
            res = load_results(acc=acc, classifier=cls_name, filter_methods=methods)
            df = res.loc[(res["dataset"].isin(datasets)) & (res["method"].isin(methods)), :]
            assert len(df["method"].unique()) == len(methods), (
                f"Error while generating {name} for {cls_name} [{acc}]: some methods missing!"
            )
            _methods, _df = rename_methods(method_map, methods, df=df)
            _datasets, _df = rename_datasets(dataset_map, datasets, df=_df)

            sns.set_context("paper", font_scale=1.1)
            plot = sns.FacetGrid(
                df,
                col="dataset",
                col_order=_datasets,
                col_wrap=len(_datasets),
                hue="method",
                hue_order=_methods,
                xlim=(0, 1),
                ylim=(0, 1),
                aspect=0.8,
                palette=get_palette(),
            )
            plot.map_dataframe(sns.scatterplot, x="true_accs", y="estim_accs", alpha=0.3, s=20)
            # plot.map_dataframe(sns.scatterplot, x="true_accs", y="estim_accs", alpha=0.2, s=20, edgecolor=None)
            # name += "_noedge"
            for ax in plot.axes.flat:
                ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)
                ax.tick_params(axis="x", labelrotation=90)
                ax.set_xticks(np.linspace(0, 1, 6, endpoint=True))
                ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))

            plot.figure.subplots_adjust(hspace=0.1, wspace=0.08)

            plot.set_titles("{col_name}")
            plot.add_legend(title="", markerscale=1.8)
            sns.move_legend(plot, "center right")
            for lh in plot.legend.legend_handles:
                lh.set_alpha(1)

            plot.set_xlabels("True Accuracy")
            plot.set_ylabels("Estimated Accuracy")

            save_figure(plot=plot, basedir=parent_dir, filename=f"grid_{cls_name}_{env.PROBLEM}_{name}")
            print(f"Plotted {name} for {cls_name} [{acc}]")


def plot_sample_size():
    from exp.leap.sample_size import get_acc_names as ss_acc_names
    from exp.leap.sample_size import get_classifier_names as ss_classifier_names
    from exp.leap.sample_size import get_dataset_names as ss_dataset_names

    methods = ["DoC", "O-LEAP(KDEy-MLP)"]
    classifiers = ss_classifier_names()
    datasets = ss_dataset_names()
    accs = ss_acc_names()

    res_dir = os.path.join(env.root_dir, "sample_size")

    for acc, cls_name in IT.product(accs, classifiers):
        res = load_results(base_dir=res_dir, acc=acc, classifier=cls_name, filter_methods=methods)
        df = res.loc[res["method"].isin(methods), :]
        assert len(df["method"].unique()) == len(methods), (
            f"Error while plotting {cls_name} [{acc}]: some methods missing!"
        )
        _methods, _df = rename_methods(method_map, methods, df=df)
        _datasets, _df = rename_datasets(dataset_map, datasets, df=_df)

        sns.set_context("paper", font_scale=1.6)
        plot = sns.relplot(
            _df,
            x="sample_size",
            y="acc_err",
            col="dataset",
            col_order=_datasets,
            col_wrap=len(datasets),
            hue="method",
            hue_order=_methods,
            kind="line",
            # sns.lineplot args
            estimator="mean",
            errorbar="se",
            err_style="bars",
            err_kws=dict(capsize=2.0, capthick=1.0),
            linewidth=1,
        )

        # set plot title
        plot.set_titles("{col_name}")

        # config legend
        plot.legend.set_title(None)
        sns.move_legend(
            plot,
            "center right",
            ncol=1,
        )

        # set axes labels
        plot.set_xlabels("Sample size")
        plot.set_ylabels("AE")

        # set log scale for x axis
        plt.xscale("log")

        # plot directory
        plot_dir = os.path.join(env.root_dir, "plots", "sample_size")
        os.makedirs(plot_dir, exist_ok=True)

        save_figure(plot, plot_dir, f"sample_size_{cls_name}_{env.PROBLEM}")
        print(f"Plotting {cls_name} [{acc} - {env.PROBLEM}]")


def plot_times():
    times_method_map = method_map | {
        "LEAP(KDEy-MLP)": "LEAP$^{\\;\\!\\mathrm{sparse}}$",
        "S-LEAP(KDEy-MLP)": "S-LEAP$^{\\;\\!\\mathrm{sparse}}$",
        "O-LEAP(KDEy-MLP)": "O-LEAP$^{\\;\\!\\mathrm{sparse}}$",
        "LEAP(KDEy-MLP)-SLSQP": "LEAP$^{\\;\\!\\mathrm{dense}}$",
        "O-LEAP(KDEy-MLP)-SLSQP": "O-LEAP$^{\\;\\!\\mathrm{dense}}$",
    }

    def get_palette():
        base_palette = sns.color_palette("Paired")
        _dict = {
            "ATC-MC": 0,
            "DoC": 1,
            "LEAP(KDEy-MLP)": 3,
            "S-LEAP(KDEy-MLP)": 7,
            "O-LEAP(KDEy-MLP)": 5,
            "LEAP(KDEy-MLP)-SLSQP": 2,
            "O-LEAP(KDEy-MLP)-SLSQP": 4,
        }
        return {times_method_map.get(k, k): base_palette[v] for k, v in _dict.items()}

    def _get_n_classes(train_prev):
        if isinstance(train_prev, list):
            return len(train_prev) + 1
        elif isinstance(train_prev, float):
            return 2
        else:
            return 0

    classifiers = ["LR"]
    accs = ["vanilla_accuracy"]
    datasets = get_dataset_names()
    methods = [
        "ATC-MC",
        "DoC",
        # "DS",
        # "CBPE",
        # "NN",
        # "Q-COT",
        "LEAP(KDEy-MLP)",
        "S-LEAP(KDEy-MLP)",
        "O-LEAP(KDEy-MLP)",
        "LEAP(KDEy-MLP)-SLSQP",
        "O-LEAP(KDEy-MLP)-SLSQP",
    ]

    parent_dir = os.path.join(env.root_dir, "plots", "times")
    os.makedirs(parent_dir, exist_ok=True)

    dfs = []
    for acc, cls_name in IT.product(accs, classifiers):
        df = load_results(acc=acc, classifier=cls_name, filter_methods=methods)
        if df.empty:
            continue
        df = df.loc[
            (df["dataset"].isin(datasets)) & (df["method"].isin(methods)),
            ["t_train", "t_test_ave", "method", "dataset", "train_prev"],
        ]
        df["n_classes"] = df["train_prev"].map(_get_n_classes)
        df = df.drop(columns=["train_prev"])
        df = df.groupby(by=["n_classes", "method", "dataset"]).mean().reset_index()
        dfs.append(df)

    res = pd.concat(dfs)
    _methods, _df = rename_methods(times_method_map, methods, df=res)

    # pivot(df, parent_dir)
    exts = ["png", "pdf"]
    paths = [os.path.join(parent_dir, f"time_plot_{env.PROBLEM}.{ext}") for ext in exts]

    plot = sns.lineplot(
        data=_df,
        x="n_classes",
        y="t_test_ave",
        hue="method",
        hue_order=_methods,
        errorbar="sd",
        err_style="bars",
        err_kws=dict(capsize=2.0, capthick=1.0),
        palette=get_palette(),
    )
    plot.legend(title="")
    plot.set_xlabel("Number of classes ($n$)")
    plot.set_ylabel("Avg. time log (s)")
    plot.set(yscale="log")
    plot.xaxis.set_major_locator(MultipleLocator(2, offset=df["n_classes"].min()))
    plot.yaxis.set_major_locator(LogLocator(10))
    plot.yaxis.set_minor_locator(MultipleLocator(1e-2))
    plot.minorticks_on()
    plot.grid(which="minor")

    for p in paths:
        plot.figure.savefig(p)
    plot.figure.clear()
    plt.close(plot.figure)


def plot_ctdiffs():
    N_COLS = 4

    def get_cts(df, method, cts_name):
        return np.array(df.loc[df["method"] == method, cts_name].to_list())

    def draw_heatmap(data, plot_names, **kwargs):
        col = data["col"].to_numpy()[0]
        row = data["row"].to_numpy()[0]
        plot_name = plot_names[col + N_COLS * row]
        data = data.drop(["col", "row"], axis=1).to_numpy()
        plot = sns.heatmap(data, **kwargs)
        plot.set_title(plot_name)

    classifiers = get_classifier_names()
    accs = get_acc_names()
    datasets = get_dataset_names()
    methods = ["Naive", "LEAP(KDEy-MLP)", "S-LEAP(KDEy-MLP)", "O-LEAP(KDEy-MLP)"]

    parent_dir = os.path.join(env.root_dir, "plots", "ctdiffs")
    os.makedirs(parent_dir, exist_ok=True)

    for cls_name, acc, dataset in IT.product(classifiers, accs, datasets):
        df = load_results(acc=acc, classifier=cls_name, dataset=dataset, filter_methods=methods)
        dataset, df = rename_datasets(dataset_map, dataset, df=df)
        methods, df = rename_methods(method_map, methods, df=df)

        cnt = 0

        plot_names, cbars = [], []
        vmin, vmax = 1, 0
        annot = False

        mdfs = []
        for m in methods:
            true_cts = get_cts(df, m, "true_cts")
            estim_cts = get_cts(df, m, "estim_cts")
            _ae = np.abs(estim_cts - true_cts).mean(axis=0)
            sqae = np.sqrt(_ae)

            mdf = pd.DataFrame(data=sqae)
            mdf["col"] = cnt % N_COLS
            mdf["row"] = 0
            mdfs.append(mdf)
            plot_names.append(m)
            cbars.append(cnt == len(methods) - 1)
            ae_min, ae_max = np.min(sqae), np.max(sqae)
            vmin = ae_min if ae_min < vmin else vmin
            vmax = ae_max if ae_max > vmax else vmax
            annot = sqae.shape[1] <= 4
            cnt += 1

        hmdf = pd.concat(mdfs, axis=0)
        plot = sns.FacetGrid(hmdf, col="col", row="row")
        cbar_ax = plot.fig.add_axes([0.92, 0.15, 0.02, 0.7])
        plot = plot.map_dataframe(
            draw_heatmap,
            plot_names=plot_names,
            vmin=vmin,
            vmax=vmax,
            cmap="rocket_r",
            annot=annot,
            cbar_ax=cbar_ax,
        )
        plot.fig.subplots_adjust(right=0.9)
        for ax in plot.axes.flatten():
            formatter = tkr.FuncFormatter(lambda x, p: "$\\omega_{" + f"{int(math.floor(x) + 1)}" + "}$")
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

        # save figure
        save_figure(plot=plot, basedir=parent_dir, filename=f"heatmap_{cls_name}_{dataset}_{env.PROBLEM}")


def plot_by_shift():
    n_bins = 20
    classifiers = ["LR"]
    accs = ["vanilla_accuracy"]
    datasets = get_dataset_names()
    methods = [
        "Naive",
        "ATC-MC",
        "DoC",
        "DS",
        "CBPE",
        "NN",
        "Q-COT",
        "LEAP(ACC)",
        "LEAP(KDEy-MLP)",
        "S-LEAP(KDEy-MLP)",
        "O-LEAP(KDEy-MLP)",
    ]

    for acc, cls_name in IT.product(accs, classifiers):
        df = load_results(acc=acc, classifier=cls_name, filter_methods=methods)
        df.loc[:, "shifts_bin"] = get_binned_values(df, "shifts", n_bins)
        _methods, df = rename_methods(method_map, methods, df=df)

        base_dir = os.path.join(env.root_dir, "plots", "shift")
        os.makedirs(base_dir, exist_ok=True)

        # plot cumulative data for all datasets
        plot = sns.lineplot(
            data=df,
            x="shifts_bin",
            y="acc_err",
            hue="method",
            hue_order=_methods,
            estimator="mean",
            errorbar="se",
            err_style="bars",
            err_kws=dict(capsize=2.0, capthick=1.0),
            linewidth=1,
            palette=sns.color_palette("Paired")[:10] + sns.color_palette("Paired")[11:],
        )
        plot.set_xlabel("Amount of PPS")
        plot.set_ylabel("AE")

        sns.move_legend(plot, "center right", bbox_to_anchor=(1.2, 0.5), title=None, frameon=False)
        save_figure(plot=plot, basedir=base_dir, filename=f"shift_{cls_name}_{env.PROBLEM}")
        print(f"Plotted {cls_name} - {acc} - {env.PROBLEM}")


def plot_pseudo_label_shift():
    def get_pseudo_label_shift(df: pd.DataFrame, datasets: list[str]):
        for d in datasets:
            _df = df.loc[df["dataset"] == d, :]
            true_cts = np.stack(_df["true_cts"].to_numpy(), axis=0)
            _posteriors = true_cts.sum(axis=1)
            _priors = true_cts.sum(axis=2)
            pl_shift = np.abs(_priors - _posteriors).sum(axis=1)
            df.loc[df["dataset"] == d, "pl_shift"] = pl_shift

    n_bins = 20
    classifiers = ["LR"]
    accs = ["vanilla_accuracy"]
    datasets = get_dataset_names()
    methods = [
        # "DoC",
        "LEAP(ACC)",
        "LEAP(KDEy-MLP)",
        "O-LEAP(KDEy-MLP)",
    ]

    for acc, cls_name in IT.product(accs, classifiers):
        df = load_results(acc=acc, classifier=cls_name, filter_methods=methods)
        get_pseudo_label_shift(df, datasets)
        df.loc[:, "pl_shift_bin"] = get_binned_values(df, "pl_shift", n_bins)
        _methods, df = rename_methods(method_map, methods, df=df)

        base_dir = os.path.join(env.root_dir, "plots", "pl_shift")
        os.makedirs(base_dir, exist_ok=True)

        # plot cumulative data for all datasets
        plot = sns.lineplot(
            data=df,
            x="pl_shift_bin",
            y="acc_err",
            hue="method",
            hue_order=_methods,
            estimator="mean",
            errorbar="se",
            err_style="bars",
            err_kws=dict(capsize=2.0, capthick=1.0),
            linewidth=1,
            # palette=sns.color_palette("Paired")[:10] + sns.color_palette("Paired")[11:],
        )
        plot.set_xlabel("Amount of pseudo-label shift")
        plot.set_ylabel("AE")

        sns.move_legend(plot, "center right", bbox_to_anchor=(1.2, 0.5), title=None, frameon=False)
        save_figure(plot=plot, basedir=base_dir, filename=f"pl_shift_{cls_name}_{env.PROBLEM}")
        print(f"Plotted {cls_name} - {acc} - {env.PROBLEM}")


def plot_qerr():
    from exp.leap.qerr import get_acc_names as qerr_accs
    from exp.leap.qerr import get_classifier_names as qerr_clssifiers
    from exp.leap.qerr import get_dataset_names as qerr_datasets
    from exp.leap.qerr import get_method_names as qerr_methods

    n_bins = 20
    classifiers = qerr_clssifiers()
    accs = qerr_accs()
    datasets = qerr_datasets()
    methods = qerr_methods()

    base_dir = os.path.join(env.root_dir, "qerr")

    for acc, cls_name in IT.product(accs, classifiers):
        df = load_results(base_dir=base_dir, acc=acc, classifier=cls_name, filter_methods=methods)
        df.loc[:, "q_errs_bin"] = get_binned_values(df, "q_errs", n_bins)
        _methods, df = rename_methods(method_map, methods, df=df)
        _datasets, df = rename_datasets(dataset_map, datasets, df=df)

        base_dir = os.path.join(env.root_dir, "plots", "qerr")
        os.makedirs(base_dir, exist_ok=True)

        plot_dir = os.path.join(env.root_dir, "plots", "qerr")
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"qerr_{cls_name}_{env.PROBLEM}"

        sns.set_context("paper", font_scale=1.1)
        plot = sns.FacetGrid(
            df,
            col="dataset",
            col_order=_datasets,
            col_wrap=len(_datasets),
            hue="method",
            hue_order=_methods,
            xlim=(0, 1),
            ylim=(0, 1),
            aspect=0.8,
            palette=get_palette(),
        )
        plot.map_dataframe(sns.scatterplot, x="q_errs", y="acc_err", alpha=0.3, s=20)
        # plot.map_dataframe(sns.scatterplot, x="q_errs", y="acc_err", alpha=0.2, s=20, edgecolor=None)
        # filename += "_noedge"
        for ax in plot.axes.flat:
            ax.tick_params(axis="x", labelrotation=90)
            ax.set_xticks(np.linspace(0, 1, 6, endpoint=True))
            ax.set_yticks(np.linspace(0, 1, 6, endpoint=True))

        plot.figure.subplots_adjust(hspace=0.1, wspace=0.08)

        plot.set_titles("{col_name}")
        plot.add_legend(title="", markerscale=1.8)
        sns.move_legend(plot, "center right")
        for lh in plot.legend.legend_handles:
            lh.set_alpha(1)

        plot.set_xlabels("Quantification Error")
        plot.set_ylabels("AE")

        save_figure(plot=plot, basedir=plot_dir, filename=filename)
        print(f"Plotted {cls_name} - {acc} - {env.PROBLEM}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--problem", action="store", default="binary", help="Select the problem you want to generate plots for"
    )
    parser.add_argument("--main", action="store_true", help="Plots for main experiments")
    parser.add_argument("--ss", action="store_true", help="Plots for sample_size experiments")
    parser.add_argument("--times", action="store_true", help="Plots for times obtained in main experiments")
    parser.add_argument(
        "--ctdiff", action="store_true", help="Plots for contingency table diffs obtained in main experiments"
    )
    parser.add_argument("--shift", action="store_true", help="Plots accuracy error by amount of PPS")
    parser.add_argument("--pseudo", action="store_true", help="Plots accuracy error by amount of pseudo-label shift")
    parser.add_argument("--qerr", action="store_true", help="Plots accuracy error by amount of quantification error")
    args = parser.parse_args()

    if args.problem not in env._valid_problems:
        raise ValueError(f"Invalid problem {args.problem}: valid problems are {env._valid_problems}")
    env.PROBLEM = args.problem

    if args.main:
        plots()
    elif args.ss:
        plot_sample_size()
    elif args.times:
        plot_times()
    elif args.ctdiff:
        plot_ctdiffs()
    elif args.shift:
        plot_by_shift()
    elif args.pseudo:
        plot_pseudo_label_shift()
    elif args.qerr:
        plot_qerr()
    else:
        parser.print_help()
