import itertools as IT
import os
from hmac import new

import pandas as pd

from exp.trd.config import PROBLEM, get_acc_names, get_dataset_names, root_dir
from exp.trd.model_selection import model_selection
from exp.trd.util import decorate_datasets, load_results, rename_datasets, rename_methods
from quacc.table import Format, Table

method_map = {
    "Naive": 'Na\\"ive',
    "ATC-MC": "ATC",
    "LEAP(KDEy)": "\\leapplus",
    "S-LEAP(KDEy)": "\\leapppskde",
    "O-LEAPCE(KDEy)-SLSQP": "\\oleapkde",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def tables():
    # res = load_results()
    # res = model_selection(res, oracle=False, only_default=True)

    def add_to_table(tbl: Table, df: pd.DataFrame, dataset, methods):
        tbl.format = Format(
            lower_is_better=False,
            mean_prec=3,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=True,
            mean_macro=False,
            color=True,
            color_mode="local",
            simple_stat=True,
        )
        for method in methods:
            values = df.loc[df["method"] == method, "true_accs"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    tbls = []
    accs = get_acc_names()
    datasets = get_dataset_names()
    for acc in accs:
        name = f"{PROBLEM}_{acc}"
        tbl = Table(name=name)
        for dataset in datasets:
            _df = load_results(acc_name=acc, dataset=dataset)
            _df = model_selection(_df, oracle=False, only_default=True)
            print(f"{dataset} results loaded")
            # _df = res.loc[res["acc_name"] == acc, :]
            methods = _df["method"].unique().tolist()
            _df, _dataset = rename_datasets(dataset_map, _df, dataset)
            _df, _dataset = decorate_datasets(_df, _dataset)
            _df, _methods = rename_methods(method_map, _df, methods)
            tbl = add_to_table(tbl, _df, _dataset, _methods)

        tbls.append(tbl)
        print(f"table {name} genned")

    pdf_path = os.path.join(root_dir, "tables", f"{PROBLEM}.pdf")
    new_commands = [
        "\\newcommand{\\leapacc}{LEAP$_\\mathrm{ACC}$}",
        "\\newcommand{\\leapplus}{LEAP$_\\mathrm{KDEy}$}",
        "\\newcommand{\\leapppskde}{S-LEAP$_\\mathrm{KDEy}$}",
        "\\newcommand{\\oleapkde}{O-LEAP$_\\mathrm{KDEy}$}",
    ]
    Table.LatexPDF(pdf_path, tables=tbls, landscape=False, new_commands=new_commands)


if __name__ == "__main__":
    tables()
