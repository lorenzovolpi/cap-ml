import dataclasses
import glob
import itertools as IT
import os
import pdb
import pickle
import subprocess
from abc import ABC
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from quapy.functional import projection_simplex_sort

import exp.leap.config as cfg
import exp.leap.env as env
from exp.leap.config import (
    get_acc_names,
    get_baseline_names,
    get_classifier_names,
    get_dataset_names,
    get_method_names,
)
from exp.leap.util import decorate_datasets, load_results, rename_datasets, rename_methods
from quacc.table import Format, Table
from quacc.utils.commons import NaNError

method_map = {
    "Naive": 'Na\\"ive',
    "ATC-MC": "ATC",
    "CBPE": "L-CBPE",
    "LEAP(ACC)": "\\leapacc",
    "LEAP(KDEy-MLP)": "\\leapplus",
    "S-LEAP(KDEy-MLP)": "\\leapppskde",
    "O-LEAP(KDEy-MLP)": "\\oleapkde",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}

acc_map = {
    "vanilla_accuracy": "Acc.",
    "macro_f1": "$F_1$",
}

table_paths = {
    "leap": os.path.join(env.root_dir, "tables"),
    "bootstrap": os.path.join(env.root_dir, "tables", "bootstrap"),
    "rcv1": os.path.join(env.root_dir, "tables", "rcv1"),
}


class TexBuilder(ABC):
    def __init__(self):
        self.lines = []

    def __iadd__(self, line: str | list[str]):
        if isinstance(line, str):
            self.lines.append(line)
        elif isinstance(line, list):
            if all([isinstance(s, str) for s in line]):
                self.lines.extend(line)

        return self

    def build(self):
        return "\n".join(self.lines)


def build_methods(all_methods: dict[str, list[str]], accs: list[str], ignored_methods: dict[str, list[str]]):
    methods = {acc: all_methods.copy() for acc in accs}
    for acc, _igm in ignored_methods.items():
        if acc not in accs:
            continue
        for _m in _igm:
            if _m not in methods[acc]:
                continue
            methods[acc].remove(_m)

    return methods


def save_tables(tbls=None, project="leap"):
    pdf_path = os.path.join(table_paths[project], f"{env.PROBLEM}.pdf")

    if tbls is None:
        pickle_path = os.path.join(table_paths[project], f"{env.PROBLEM}.pickle")
        with open(pickle_path, "rb") as f:
            tbls = pickle.load(f)

    new_commands = [
        "\\newcommand{\\leapacc}{LEAP$_\\mathrm{ACC}$}",
        "\\newcommand{\\leapplus}{LEAP$_\\mathrm{KDEy}$}",
        "\\newcommand{\\leapppskde}{S-LEAP$_\\mathrm{KDEy}$}",
        "\\newcommand{\\oleapkde}{O-LEAP$_\\mathrm{KDEy}$}",
    ]

    if project == "leap":
        Table.LatexPDF(pdf_path, tables=tbls, landscape=False, new_commands=new_commands)
    elif project == "bootstrap":
        tabulars = _bootstrap_tex_gen(tbls)
        _bootstrap_pdf_gen(pdf_path, tabulars=tabulars, new_commands=new_commands)
    elif project == "rcv1":
        tabular = _rcv1_tex_gen(tbls)
        pdf_path = os.path.join(table_paths[project], "rcv1.pdf")
        _rcv1_pdf_gen(pdf_path, tabular=tabular, new_commands=new_commands)


def _bootstrap_tex_gen(tbls: dict[str, list[Table]]):
    tex_dir = os.path.join(table_paths["bootstrap"], "tables")
    os.makedirs(tex_dir, exist_ok=True)

    for tbl in tbls:
        methods = tbl.get_methods()
        benchmarks = tbl.get_benchmarks()

        cline = "\\cline{2-" + str(len(methods) + 1) + "}"

    tabulars = []
    for name, tbl_ls in tbls.items():
        _tbl_methods = [_tbl.get_methods() for _tbl in tbl_ls]
        _tbl_benchmarks = [_tbl.get_benchmarks() for _tbl in tbl_ls]
        assert all([_tm == _tbl_methods[0] for _tm in _tbl_methods])
        assert all([_tb == _tbl_benchmarks[0] for _tb in _tbl_benchmarks])
        methods = _tbl_methods[0]
        benchmarks = _tbl_benchmarks[0]

        cline = "\\cline{2-" + str(len(tbl_ls) * len(methods) + 1) + "}"

        tex = TexBuilder()
        tex += "\\begin{tabular}{|c|" + ((("c" * len(tbl_ls)) + "|") * len(methods)) + "}"
        tex += cline
        tex += (
            "\\multicolumn{1}{c|}{} & "
            + " & ".join(["\\multicolumn{" + str(len(tbl_ls)) + "}{c|}{" + m + "}" for m in methods])
            + " \\\\"
            + cline
        )
        tbl_ls_columns = [_tbl.name for _tbl in tbl_ls]
        tex += (
            "\\multicolumn{1}{c|}{} & " + " & ".join(IT.chain(*IT.repeat(tbl_ls_columns, len(methods)))) + "\\\\\\hline"
        )

        for b in benchmarks:
            measure_grouped_cells = [[_tbl.get(b, m).print() for m in _tbl.get_methods()] for _tbl in tbl_ls]
            ln = b + " & "
            ln += " & ".join(IT.chain(*zip(*measure_grouped_cells)))
            ln += " \\\\"
            tex += ln

        tex += "\\hline"
        tex += "\\end{tabular}"
        tex_str = tex.build()
        tabulars.append((name, tex_str))

        tex_path = os.path.join(tex_dir, f"{name}.tex")
        with open(tex_path, "w") as f:
            f.write(tex_str)

        print(f"Table '{name}.tex' done.")

        return tabulars


def _bootstrap_pdf_gen(pdf_path, tabulars, new_commands=None, debug=False):
    new_commands = [] if new_commands is None else new_commands

    tex_path = pdf_path.replace(".pdf", ".tex")

    doc = TexBuilder()
    doc += "\\documentclass[10pt,a4paper]{article}"
    doc += "\\usepackage[utf8]{inputenc}"
    doc += "\\usepackage{amsmath}"
    doc += "\\usepackage{amsfonts}"
    doc += "\\usepackage{amssymb}"
    doc += "\\usepackage{graphicx}"
    doc += "\\usepackage[dvipsnames]{xcolor}"
    doc += "\\usepackage{colortbl}"
    doc += "\\usepackage{booktabs}"
    doc += new_commands
    doc += ""
    doc += "\\begin{document}"

    for name, tbr in tabulars:
        caption = name.replace("_", "\\_")
        label = f"tab:{name}"

        doc += ""
        doc += "\\begin{table}[h]"
        doc += "\\center"
        doc += "\\resizebox{\\textwidth}{!}{%"
        doc += tbr
        doc += "}%"
        doc += f"\\caption{{{caption}}}"
        doc += f"\\label{{{label}}}"
        doc += "\\end{table}"

        doc += ["", "", "\\newpage", ""]
        doc += "\\end{document}"

    document = doc.build()
    with open(tex_path, "w") as f:
        f.write(document)

    dir = Path(pdf_path).parent
    pwd = os.getcwd()
    os.chdir(dir)
    tex_path_name = Path(tex_path).name
    command = ["pdflatex", tex_path_name]
    if debug:
        subprocess.run(command)
    else:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    basename = tex_path_name.replace(".tex", "")
    if debug:
        os.system(f"rm {basename}.aux {basename}.log")
    else:
        os.system(f"rm {basename}.aux {basename}.log {basename}.tex")
    os.chdir(pwd)

    print(f"Document '{Path(pdf_path).name}' done.")


def _rcv1_tex_gen(tbls: dict[str, Table]):
    tex_dir = os.path.join(table_paths["rcv1"], "tables")
    os.makedirs(tex_dir, exist_ok=True)

    one_acc = len(tbls) == 1
    _tbl_methods = [_tbl.get_methods() for _tbl in tbls.values()]
    _tbl_benchmarks = [_tbl.get_benchmarks() for _tbl in tbls.values()]
    assert all([_tm == _tbl_methods[0] for _tm in _tbl_methods])
    assert all([_tb == _tbl_benchmarks[0] for _tb in _tbl_benchmarks])
    methods = _tbl_methods[0]
    benchmarks = _tbl_benchmarks[0]

    cline = "\\cline{2-" + str(len(methods) + 1) + "}" if one_acc else "\\cline{3-" + str(len(methods) + 2) + "}"

    tex = TexBuilder()
    tex += "\\begin{tabular}{" + ("|c|" if one_acc else "|c|c|") + ("c" * len(methods)) + "|}"
    tex += cline
    tex += "\\multicolumn{" + ("1" if one_acc else "2") + "}{c|}{} & "
    tex += " & ".join([m for m in methods]) + " \\\\\\hline"

    for k, (name, _tbl) in enumerate(tbls.items()):
        for i, b in enumerate(benchmarks):
            _cells = [_tbl.get(b, m).print() for m in _tbl.get_methods()]
            if one_acc:
                ln = ""
            else:
                ln = (
                    "\\multirow{" + str(len(benchmarks)) + "}{*}{\\side{" + acc_map.get(name, name) + "}}\n& "
                    if i == 0
                    else "& "
                )
            ln += b + " & "
            ln += " & ".join(_cells)
            ln += " \\\\"
            tex += ln

        tex += "\\hline" if k + 1 == len(tbls) else "\\midrule"

    tex += "\\end{tabular}"
    tabular = tex.build()

    tbl_name = f"rcv1_{env.PROBLEM}"
    tex_path = os.path.join(tex_dir, f"{tbl_name}.tex")
    with open(tex_path, "w") as f:
        f.write(tabular)

    print(f"Table '{tbl_name}.tex' done.")

    return tabular


def _rcv1_pdf_gen(pdf_path, tabular, new_commands=None, debug=False):
    new_commands = [] if new_commands is None else new_commands

    tex_path = pdf_path.replace(".pdf", ".tex")
    name = Path(tex_path).stem

    doc = TexBuilder()
    doc += "\\documentclass[10pt,a4paper]{article}"
    doc += "\\usepackage[utf8]{inputenc}"
    doc += "\\usepackage{amsmath}"
    doc += "\\usepackage{amsfonts}"
    doc += "\\usepackage{amssymb}"
    doc += "\\usepackage{graphicx}"
    doc += "\\usepackage{rotating}"
    doc += "\\usepackage[dvipsnames]{xcolor}"
    doc += "\\usepackage{colortbl}"
    doc += "\\usepackage{booktabs}"
    doc += "\\usepackage{multirow}"
    doc += "\\newcommand{\\side}[1]{\\begin{sideways}{#1}\\end{sideways}}"
    doc += new_commands
    doc += ""
    doc += "\\begin{document}"

    caption = name.replace("_", "\\_")
    label = f"tab:{name}"

    doc += ""
    doc += "\\begin{table}[h]"
    doc += "\\center"
    doc += "\\resizebox{\\textwidth}{!}{%"
    doc += tabular
    doc += "}%"
    doc += f"\\caption{{{caption}}}"
    doc += f"\\label{{{label}}}"
    doc += "\\end{table}"

    doc += ["", ""]
    doc += "\\end{document}"

    document = doc.build()
    with open(tex_path, "w") as f:
        f.write(document)

    dir = Path(pdf_path).parent
    pwd = os.getcwd()
    os.chdir(dir)
    tex_path_name = Path(tex_path).name
    command = ["pdflatex", tex_path_name]
    if debug:
        subprocess.run(command)
    else:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    basename = tex_path_name.replace(".tex", "")
    if debug:
        os.system(f"rm {basename}.aux {basename}.log")
    else:
        os.system(f"rm {basename}.aux {basename}.log {basename}.tex")
    os.chdir(pwd)

    print(f"Document '{Path(pdf_path).name}' done.")


def tables():
    classifiers = get_classifier_names()
    datasets = get_dataset_names()
    baselines = []
    all_methods = [
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
    ignored_methods = {
        "f1": ["Q-COT", "ATC-MC"],
        "macro_f1": ["Q-COT", "ATC-MC"],
    }
    accs = get_acc_names()
    methods = build_methods(all_methods, accs, ignored_methods)

    def gen_table(df: pd.DataFrame, name, datasets, methods, baselines):
        tbl = Table(name=name, benchmarks=datasets, methods=methods, baselines=baselines)
        tbl.format = Format(
            mean_prec=3,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=True,
            mean_macro=False,
            color=True,
            # color_mode="baselines",
            color_mode="local",
            simple_stat=True,
            # best_color="OliveGreen",
            best_color="green",
            mid_color="SeaGreen",
        )
        for dataset, method in IT.product(datasets, methods):
            values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), "acc_err"].to_numpy()
            if np.any(np.isnan(values)):
                raise NaNError(dataset=dataset, method=method)
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    tbls = []
    for classifier, acc in IT.product(classifiers, accs):
        df = load_results(acc=acc, classifier=classifier, filter_methods=methods[acc])
        name = f"{env.PROBLEM}_{classifier}_{acc}"
        _datasets, _df = rename_datasets(dataset_map, datasets, df=df)
        _datasets, _df = decorate_datasets(_datasets, df=_df)
        _methods, _df, _baselines = rename_methods(method_map, methods[acc], df=_df, baselines=baselines)
        try:
            tbls.append(gen_table(_df, name, _datasets, _methods, _baselines))
        except NaNError as e:
            print(name, e.dataset, e.method)
            continue

    # dump_tables(tbls, project="leap")
    save_tables(tbls=tbls, project="leap")


def leap_true_solve():
    methods = ["LEAP(KDEy)", "LEAP(KDEy-a)", "LEAP(MDy)"]
    res = load_results(filter_methods=methods)
    md_path = os.path.join(table_paths["leap"], f"{env.PROBLEM}_true_solve.md")

    pd.pivot_table(
        res.loc[res["method"].isin(methods)],
        columns=["classifier", "method"],
        index=["dataset"],
        values="true_solve",
    ).to_markdown(md_path)


def confidence_intervals():
    from exp.leap.bootstrap import get_acc_names as bootstrap_acc_names
    from exp.leap.bootstrap import get_classifier_names as bootstrap_classifier_names
    from exp.leap.bootstrap import get_method_names as bootstrap_method_names

    def add_to_table(tbl: Table, df: pd.DataFrame, dataset, methods, column):
        for method in methods:
            values = df.loc[df["method"] == method, column]
            for v in values.to_numpy():
                tbl.add(dataset, method, v)

    classifiers = bootstrap_classifier_names()
    datasets = get_dataset_names()
    methods = bootstrap_method_names()
    acc_names = bootstrap_acc_names()

    base_dir = os.path.join(env.root_dir, "bootstrap")

    tbls = {}
    for acc, cls_name in IT.product(acc_names, classifiers):
        _datasets = decorate_datasets(rename_datasets(dataset_map, datasets))
        _methods = rename_methods(method_map, methods)
        name = (
            f"bootstrap_{env.PROBLEM}"
            if len(acc_names) == 1 and len(classifiers) == 1
            else f"bootstrap_{env.PROBLEM}_{cls_name}_{acc}"
        )
        tbl_format = Format(
            mean_prec=3,
            show_std=False,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=False,
            mean_macro=False,
            color=True,
            color_mode="local",
            simple_stat=True,
            maxtone=35,
        )
        ae_tbl = Table(name="AE", benchmarks=_datasets, methods=_methods)
        ae_tbl.format = dataclasses.replace(tbl_format)
        delta_tbl = Table(name="$\\mathcal{A}$", benchmarks=_datasets, methods=_methods)
        delta_tbl.format = dataclasses.replace(tbl_format)
        cov_tbl = Table(name="$\\mathcal{C}$", benchmarks=_datasets, methods=_methods)
        cov_tbl.format = dataclasses.replace(tbl_format)
        cov_tbl.format.lower_is_better = False

        for dataset in datasets:
            res = load_results(base_dir=base_dir, acc=acc, dataset=dataset)
            res: pd.DataFrame = res.loc[
                (res["acc_name"] == acc) & (res["classifier"] == cls_name),
                ["dataset", "method", "sample_distrib_id", "acc_err", "ci_delta", "coverage"],
            ]
            res["coverage"] = res["coverage"].astype(int)

            _dataset, _df = rename_datasets(dataset_map, dataset, df=res)
            _dataset, _df = decorate_datasets(_dataset, df=_df)
            _d_methods, _df = rename_methods(method_map, methods, df=_df)
            add_to_table(ae_tbl, _df.loc[:, ["method", "acc_err"]], _dataset, _d_methods, "acc_err")
            add_to_table(delta_tbl, _df.loc[:, ["method", "ci_delta"]], _dataset, _d_methods, "ci_delta")
            add_to_table(cov_tbl, _df.loc[:, ["method", "coverage"]], _dataset, _d_methods, "coverage")

        tbls[name] = [ae_tbl, delta_tbl, cov_tbl]

    # dump_tables(tbls, project="bootstrap")
    save_tables(tbls, project="bootstrap")


def rcv1_tables():
    from exp.leap.rcv1 import get_acc_names as rcv1_accs
    from exp.leap.rcv1 import get_dataset_names as rcv1_datasets
    from exp.leap.rcv1 import get_method_names as rcv1_methods

    def add_to_table(tbl: Table, df: pd.DataFrame, cls_name, methods, column):
        for method in methods:
            values = df.loc[df["method"] == method, column]
            for v in values.to_numpy():
                tbl.add(cls_name, method, v)

    classifiers = get_classifier_names()
    dataset = rcv1_datasets()[0]
    methods = rcv1_methods()
    acc_names = rcv1_accs()

    tbls = {}
    for acc in acc_names:
        _methods = rename_methods(method_map, methods)
        tbl_format = Format(
            mean_prec=3,
            show_std=False,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=False,
            mean_macro=False,
            color=True,
            color_mode="local",
            simple_stat=True,
            maxtone=35,
        )
        tbl = Table(name="rcv1", benchmarks=classifiers, methods=_methods)
        tbl.format = dataclasses.replace(tbl_format)

        for cls_name in classifiers:
            res = load_results(acc=acc, dataset=dataset, classifier=cls_name, filter_methods=methods)
            res: pd.DataFrame = res.loc[:, ["method", "acc_err"]]

            _d_methods, df = rename_methods(method_map, methods, df=res)
            add_to_table(tbl, df, cls_name, _d_methods, column="acc_err")

        tbls[acc] = tbl

    # dump_tables(tbls, project="bootstrap")
    save_tables(tbls, project="rcv1")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-p", "--problem", action="store", default="binary", help="Select the problem you want to generate plots for"
    )
    parser.add_argument("--main", action="store_true", help="Compute main leap tables")
    parser.add_argument("--ts", action="store_true", help="Compute leap true_solve table")
    parser.add_argument("--ci", action="store_true", help="Compute confidence intervals tables")
    parser.add_argument("--rcv1", action="store_true", help="Compute rcv1 table")
    parser.add_argument("--pdf", action="store_true", help="Save pdf table from pickle file")
    args = parser.parse_args()

    if args.problem not in env._valid_problems:
        raise ValueError(f"Invalid problem {args.problem}: valid problems are {env._valid_problems}")
    env.PROBLEM = args.problem

    if args.main:
        tables()
    elif args.ts:
        leap_true_solve()
    elif args.ci:
        confidence_intervals()
    elif args.rcv1:
        env.PROBLEM = "multiclass"
        rcv1_tables()
    elif args.pdf:
        save_tables(project="bootstrap")
    else:
        parser.print_help()
