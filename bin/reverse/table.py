import os
import pickle
from argparse import ArgumentParser
from typing import Literal

import pandas as pd
from pandatex import Format, Table
from results import Results
from util import decorate_dataset

from bin.reverse import env, main
from bin.reverse.config import acc_from_ct, get_acc_names, get_existing_dataset_names

method_map = {
    "O-LEAP": "\\leap",
    "RQBS": "\\rqbs",
    "DoC": "\\doc",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
    "stanfordnlp__imdb": "imdb",
    "fancyzhx__yelp_polarity": "yelp-polarity",
    "stanfordnlp__sst2": "sst2",
    "fancyzhx__ag_news": "ag-news",
    "fancyzhx__dbpedia_14": "dbpedia-14",
    "community-datasets__yahoo_answers_topics": "yahoo-answers-topics",
    "yelp_reviews": "yelp-reviews",
    "ag_news-lt": "ag-news-lt",
    "dbpedia_14-lt": "dbpedia-14-lt",
    "ethz__food101": "food101",
}


def ms_selection():
    return {
        "oracle-hi": True,
        "oracle-low": True,
        "default": [],
        "method": [
            # ("IMS", "LR"),
            # ("IMS", "kNN"),
            # ("IMS", "SVM"),
            # ("IMS", "MLP"),
            ("IMS", None),
            ("TMS_LEAP", None),
            ("TMS_RQBS", None),
            # ("TMS_PrediQuant", None),
            # ("TMS_DoC", None),
        ],
    }


def gen_tables(pargs):
    experiment = main.EXPERIMENT
    domain = pargs.domain
    # rank_label = "ranking_vals"

    main_base_dir = os.path.join(env.root_dir, experiment)
    # ensamble_base_dir = os.path.join(env.root_dir, "ensamble")

    def add_to_table(tbl: Table, df: pd.DataFrame, dataset, methods):
        for method in methods:
            values = df.loc[df["method"] == method, "true_accs"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    tbls = []
    accs = get_acc_names()
    datasets = get_existing_dataset_names("main", domain)
    for acc in accs:
        name = f"{experiment}_{domain}_{acc}"
        tbl = Table(name=name, oracles=["oracle-hi", "oracle-low"])
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
        for dataset in datasets:
            res = (
                Results.load(base_dir=main_base_dir, acc_name=acc, dataset=dataset, domain=domain)
                # .split_by_shift(prevs=0.5)
                # .model_selection(selection=ms_selection(), rank_label=rank_label)
                .map_column_values("method", method_map)
                .map_column_values("dataset", dataset_map)
                .apply_to_column("dataset", decorate_dataset)
                .apply_to_column("true_cts", lambda ct: acc_from_ct(acc, ct), new_col="true_accs")
            )

            _methods = [method_map.get(m, m) for m in res.unique_column_values("method")]
            _dataset = decorate_dataset(dataset_map.get(dataset, dataset))
            tbl = add_to_table(tbl, res.df, _dataset, _methods)
            print(f"[{acc}] {dataset} done")

        tbls.append(tbl)
        print(f"table {name} genned")

    table_dir = os.path.join(env.root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    pickle_path = os.path.join(table_dir, f"{experiment}_{domain}.pickle")
    with open(pickle_path, "wb") as f:
        pickle.dump(tbls, f)


def gen_pdf(pargs):
    experiment = main.EXPERIMENT
    domain = pargs.domain
    table_dir = os.path.join(env.root_dir, "tables")
    os.makedirs(table_dir, exist_ok=True)
    pickle_path = os.path.join(table_dir, f"{experiment}_{domain}.pickle")
    assert os.path.exists(pickle_path), "pickle file does not exist"

    with open(pickle_path, "rb") as f:
        tbls = pickle.load(f)

    pdf_path = os.path.join(table_dir, f"{experiment}_{domain}.pdf")
    new_commands = [
        "\\newcommand{\\leap}{O-LEAP}",
        "\\newcommand{\\rqbs}{RQBS}",
        "\\newcommand{\\doc}{DoC}",
    ]
    # column_alignment = [3], "c"
    # additional_headers = [("oracle", 2), ("IMS", 1), ("TMS", 2)]
    Table.LatexPDF(
        pdf_path,
        tables=tbls,
        landscape=False,
        new_commands=new_commands,
        # column_alignment=column_alignment,
        # additional_headers=additional_headers,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-t", "--tables", action="store_true", help="Generate tables")
    parser.add_argument("-p", "--pdf", action="store_true", help="Generate PDF from tables")
    parser.add_argument("-a", "--all", action="store_true", help="Generate both tables and PDF")
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    if pargs.tables:
        gen_tables(pargs)
    if pargs.pdf:
        gen_pdf(pargs)
