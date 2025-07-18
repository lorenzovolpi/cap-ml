import json
import os
from glob import glob
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash import Input, Output, State, callback, ctx, dash_table, dcc, html, register_page
from dash.dash_table.Format import Format, Scheme

import quacc as qc
from qcdash.navbar import APP_NAME
from quacc.experiments.report import Report

register_page(__name__, name=f"{APP_NAME} - table", top_nav=True, path="/table")

root_folder = os.path.join(qc.env["OUT_DIR"], "results")
valid_errors = sorted(list(qc.error.ACCURACY_ERROR_SINGLE_NAMES))


def mbr_label(mbr):
    return "Methods x Datasets" if mbr else "Datasets x Methods"


def get_df(rep: Report, error, method_by_row=True, zscore=False):
    if error == "ae":
        df = rep.table_data()
    elif error == "se":
        df = rep.table_data(error=qc.error.se)

    if method_by_row:
        df = df.pivot_table(
            values="z_score" if zscore else "acc_err", index=["method"], columns=["dataset"], fill_value=np.nan
        )
    else:
        df = df.pivot_table(
            values="z_score" if zscore else "acc_err", index=["dataset"], columns=["method"], fill_value=np.nan
        )
    return df


def get_Table(df: pd.DataFrame, method_by_row=True):
    if df is None:
        return None

    # insert avg column
    if method_by_row:
        df.insert(0, "avg", df.mean(axis=1))
    else:
        df.loc["avg", :] = df.mean(axis=0)

    def gen_idxmin(_df):
        if method_by_row:
            df_idxmin = _df.idxmin(axis=0, numeric_only=True)
            df_idxmin_cols = df_idxmin.index.to_numpy()
            rows_sorter = np.argsort(_df.index.to_numpy())
            df_idxmin_rows = rows_sorter[
                np.searchsorted(_df.index.to_numpy(), df_idxmin.to_numpy(), sorter=rows_sorter)
            ]
        else:
            df_idxmin = _df.idxmin(axis=1, numeric_only=True)
            df_idxmin_cols = df_idxmin.to_numpy()
            df_idxmin_rows = np.arange(df_idxmin_cols.shape[0])

        for r, c in zip(df_idxmin_rows, df_idxmin_cols):
            yield r, c

    def gen_beats_baselines(_df):
        if method_by_row:
            df_idx = df.index.to_numpy()
            baselines = ["DoC", "ATC-MC", "ATC-NE"]
            dfnp = _df.to_numpy()
            base_idx = np.where(np.in1d(df_idx, baselines))[0]
            if len(base_idx) == 0:
                method_row_idx, method_col_idx, base_row_idx, base_col_idx = [], [], [], []
            else:
                methods_idx = np.setdiff1d(np.arange(len(df_idx)), base_idx)
                where_beats = np.all([dfnp <= dfnp[i] for i in base_idx], axis=0)

                where_methods_beats = np.copy(where_beats)
                where_methods_beats[base_idx] = False

                where_base_beats = np.copy(where_beats)
                where_base_beats[methods_idx] = False

                base_row_idx, base_col_idx = np.where(where_base_beats)
                method_row_idx, method_col_idx = np.where(where_methods_beats)
                base_col_idx = _df.columns[base_col_idx].to_numpy()
                method_col_idx = _df.columns[method_col_idx].to_numpy()
        else:
            df_col = df.columns.to_numpy()
            baselines = ["DoC", "ATC-MC", "ATC-NE"]
            dfnp = _df.to_numpy()
            base_col = np.where(np.in1d(df_col, baselines))[0]
            if len(base_col) == 0:
                method_row_idx, method_col_idx, base_row_idx, base_col_idx = [], [], [], []
            else:
                methods_col = np.setdiff1d(np.arange(len(df_col)), base_col)
                where_beats = np.all([dfnp.T <= dfnp.T[i] for i in base_col], axis=0)

                where_methods_beats = np.copy(where_beats)
                where_methods_beats[base_col] = False

                where_base_beats = np.copy(where_beats)
                where_base_beats[methods_col] = False

                base_col_idx, base_row_idx = np.where(where_base_beats)
                method_col_idx, method_row_idx = np.where(where_methods_beats)
                base_col_idx = _df.columns[base_col_idx].to_numpy()
                method_col_idx = _df.columns[method_col_idx].to_numpy()

        gen_base = ((_r, _c) for _r, _c in zip(base_row_idx, base_col_idx))
        gen_methods = ((_r, _c) for _r, _c in zip(method_row_idx, method_col_idx))
        return gen_base, gen_methods

    columns = {
        c: dict(
            id=c,
            name=c,
            type="numeric",
            format=Format(precision=9, scheme=Scheme.fixed, nully="nan"),
        )
        for c in df.columns
    }
    # columns["dataset"] |= dict(type="text", format=Format())
    columns = list(columns.values())

    _style_table = {
        "margin": "2vh 0px",
        "padding": "15px 0px",
        "overflowX": "auto",
        "border-radius": "6px",
        "maxWidth": "64vw",
    }
    _style_idx_table = {
        "margin": "2vh 0px",
        "padding": "15px 0px",
        "maxWidth": "20vw",
    }

    _style_cell = {"padding": "0 12px", "text_align": "center", "font_family": "sans"}
    _style_idx_cell = {"padding": "0 12px", "text_align": "right", "font_family": "sans"}
    # _style_cell_cond = [{"if": {"column_id": "dataset"}, "text_align": "right"}]
    _style_data_cond = []
    for _r, _c in gen_idxmin(df):
        _style_data_cond.append(
            {
                "if": {"column_id": _c, "row_index": _r},
                "font_weight": "bold",
            }
        )
    gen_baseline, gen_methods = gen_beats_baselines(df)
    for _r, _c in gen_baseline:
        _style_data_cond.append(
            {
                "if": {"column_id": _c, "row_index": _r},
                "background": "yellow",
            }
        )
    for _r, _c in gen_methods:
        _style_data_cond.append(
            {
                "if": {"column_id": _c, "row_index": _r},
                "background": "lightgreen",
            }
        )

    idx_table = dash_table.DataTable(
        df.index.to_frame().to_dict("records"),
        style_table=_style_idx_table,
        style_cell=_style_idx_cell,
        cell_selectable=False,
    )
    table = dash_table.DataTable(
        df.to_dict("records"),
        columns=columns,
        style_table=_style_table,
        style_cell=_style_cell,
        # style_cell_conditional=_style_cell_cond,
        style_data_conditional=_style_data_cond,
        cell_selectable=False,
    )
    return html.Div(
        [idx_table, table],
        style={
            "display": "flex",
            "flex-direction": "row",
            "align-items": "top",
            "justify-content": "center",
            # "height": "100vh",
        },
    )
    return table


def get_report(config, classifier, acc, dataset, methods):
    return Report.load_results(root_folder, config, classifier, acc, dataset, methods)


sidebar_style = {
    "top": 0,
    "left": 0,
    "bottom": 0,
    # "padding": "1vw",
    "padding-top": "2vw",
    "margin": "0px",
    "flex": 1,
    "overflow": "auto",
    "height": "94vh",
    "maxHeight": "94vh",
}

content_style = {
    "flex": 5,
}


def parse_href(href: str):
    parse_result = urlparse(href)
    params = parse_qsl(parse_result.query)
    return dict(params)


def get_sidebar(**kwargs):
    config = kwargs.get("config", None)
    classifier = kwargs.get("classifier", None)
    acc = kwargs.get("acc", None)
    error = kwargs.get("error", None)
    mbr = json.loads(kwargs.get("mbr", "True").lower())
    zscore = json.loads(kwargs.get("zscore", "False").lower())
    datasets = kwargs.get("datasets", [])
    methods = kwargs.get("methods", [])
    return [
        dbc.Row(
            [
                dbc.Label("config", width=3),
                dbc.Col(dbc.Select(id="tbl_config", value=config), width=9),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Label("classifier", width=3),
                dbc.Col(dbc.Select(id="tbl_classifier", value=classifier), width=9),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Label("accuracy", width=3),
                dbc.Col(dbc.Select(id="tbl_acc", value=acc), width=9),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Label("error", width=3),
                dbc.Col(dbc.Select(id="tbl_error", options=valid_errors, value=error), width=9),
            ],
            className="mb-3",
        ),
        dbc.Switch(id="tbl_mbr", label=mbr_label(mbr), value=mbr, class_name="mb-3"),
        dbc.Switch(id="tbl_zscore", label="z-score", value=zscore, class_name="mb-3"),
        dbc.Accordion(
            [
                dbc.AccordionItem([dbc.Checklist(id="tbl_datasets", value=datasets, switch=True)], title="Datasets"),
                dbc.AccordionItem([dbc.Checklist(id="tbl_methods", value=methods, switch=True)], title="Methods"),
            ],
            always_open=True,
            class_name="mb-1",
        ),
        # dbc.Row(
        #     [dbc.Label("Datasets"), dcc.Dropdown(id="tbl_datasets", value=datasets, multi=True, clearable=False)],
        #     className="mb-3",
        # ),
        # dbc.Row([dbc.Label("Methods"), dcc.Dropdown(id="tbl_methods", value=methods, multi=True, clearable=False)]),
    ]


def layout(**kwargs):
    layout = html.Div(
        [
            # dcc.Interval(id="reload", interval=10 * 60 * 1000),
            dcc.Location(id="tbl_url", refresh=False),
            dcc.Store(id="tbl_root", storage_type="session", data=root_folder),
            dcc.Store(id="tbl_tree", storage_type="session", data={}),
            dbc.Row(
                children=[
                    html.Div(get_sidebar(**kwargs), id="tbl_app_sidebar", style=sidebar_style),
                    html.Div(id="tbl_app_content", style=content_style),
                ],
                id="tbl_page_layout",
            ),
        ],
        style={
            "display": "flex",
            "flex-flow": "column nowrap",
            "maxHeight": "94vh",
            "overflow": "auto",
            "padding": "0px 1vw",
        },
    )

    return layout


def build_tree():
    tree = {}
    for path, dirs, files in os.walk(root_folder):
        if len(files) == 0:
            tree[path] = dirs
        else:
            tree[path] = [Path(f).stem for f in files]

    return tree


def apply_param(href, triggered_id, id, curr):
    match triggered_id:
        case "tbl_url":
            params = parse_href(href)
            if id == "mbr":
                val = json.loads(params.get(id, "True").lower())
            elif id == "zscore":
                val = json.loads(params.get(id, "False").lower())
            else:
                val = params.get(id, None)

            return val
        case _:
            return curr


def get_valid_fields(tree, req, *args):
    if req == "config":
        assert len(args) == 0
    elif req == "classifier":
        assert len(args) == 1 and None not in args
    elif req == "acc":
        assert len(args) == 2 and None not in args
    elif req == "datasets":
        assert len(args) == 3 and None not in args
    elif req == "methods":
        assert len(args) == 4 and None not in args

    if req == "methods":
        config, classifier, acc, datasets = args
        idxs = [os.path.join(root_folder, config, classifier, acc, d) for d in datasets]
        res = []
        for path in idxs:
            res += tree.get(path, [])
        res = np.unique(res).tolist()
    else:
        idx = os.path.join(root_folder, *args)
        res = np.unique(tree.get(idx, [])).tolist()
    return res


@callback(
    Output("tbl_config", "value"),
    Output("tbl_config", "options"),
    Output("tbl_tree", "data"),
    Input("tbl_url", "href"),
)
def tbl_update_config(href):
    tree = build_tree()
    req_config = parse_href(href).get("config", None)
    valid_configs = get_valid_fields(tree, "config")
    assert len(valid_configs) > 0, "no valid configs"
    new_config = req_config if req_config in valid_configs else valid_configs[0]
    return new_config, valid_configs, tree


@callback(
    Output("tbl_classifier", "value"),
    Output("tbl_classifier", "options"),
    Input("tbl_url", "href"),
    Input("tbl_config", "value"),
    State("tbl_tree", "data"),
    State("tbl_classifier", "value"),
)
def tbl_update_classifier(href, config, tree, classifier):
    req_classifier = apply_param(href, ctx.triggered_id, "classifier", classifier)
    valid_classifiers = get_valid_fields(tree, "classifier", config)
    assert len(valid_classifiers) > 0, "no valid classifiers"
    new_classifier = req_classifier if req_classifier in valid_classifiers else valid_classifiers[0]
    return new_classifier, valid_classifiers


@callback(
    Output("tbl_acc", "value"),
    Output("tbl_acc", "options"),
    Input("tbl_url", "href"),
    Input("tbl_config", "value"),
    Input("tbl_classifier", "value"),
    State("tbl_tree", "data"),
    State("tbl_acc", "value"),
)
def tbl_update_acc(href, config, classifier, tree, acc):
    req_acc = apply_param(href, ctx.triggered_id, "acc", acc)
    valid_accs = get_valid_fields(tree, "acc", config, classifier)
    assert len(valid_accs) > 0, "no valid accs"
    default_acc = "vanilla_accuracy" if "vanilla_accuracy" in valid_accs else valid_accs[0]
    new_acc = req_acc if req_acc in valid_accs else default_acc
    return new_acc, valid_accs


@callback(
    Output("tbl_datasets", "value"),
    Output("tbl_datasets", "options"),
    Input("tbl_url", "href"),
    Input("tbl_config", "value"),
    Input("tbl_classifier", "value"),
    Input("tbl_acc", "value"),
    State("tbl_tree", "data"),
    State("tbl_datasets", "value"),
)
def tbl_update_dataset(href, config, classifier, acc, tree, datasets):
    valid_datasets = get_valid_fields(tree, "datasets", config, classifier, acc)
    assert len(valid_datasets) > 0, "no valid datasets"

    req_datasets = apply_param(href, ctx.triggered_id, "datasets", datasets)
    if isinstance(req_datasets, str):
        try:
            req_datasets = json.loads(req_datasets)
        except JSONDecodeError:
            req_datasets = valid_datasets
    else:
        req_datasets = valid_datasets

    if req_datasets is None or len(req_datasets) == 0:
        return valid_datasets, valid_datasets

    new_dataset = np.unique(np.array(req_datasets)[np.in1d(req_datasets, valid_datasets)]).tolist()
    if len(new_dataset) == 0:
        new_dataset = valid_datasets

    return new_dataset, valid_datasets


@callback(
    Output("tbl_methods", "value"),
    Output("tbl_methods", "options"),
    Input("tbl_url", "href"),
    Input("tbl_config", "value"),
    Input("tbl_classifier", "value"),
    Input("tbl_acc", "value"),
    Input("tbl_datasets", "value"),
    State("tbl_tree", "data"),
    State("tbl_methods", "value"),
)
def tbl_update_methods(href, config, classifier, acc, dataset, tree, methods):
    req_methods = apply_param(href, ctx.triggered_id, "methods", methods)
    if isinstance(req_methods, str):
        try:
            req_methods = json.loads(req_methods)
        except JSONDecodeError:
            req_methods = []
    valid_methods = get_valid_fields(tree, "methods", config, classifier, acc, dataset)

    if req_methods is None or len(req_methods) == 0:
        return [], valid_methods

    new_methods = np.unique(np.array(req_methods)[np.in1d(req_methods, valid_methods)]).tolist()
    return new_methods, valid_methods


@callback(
    Output("tbl_error", "value"),
    Output("tbl_error", "options"),
    Input("tbl_url", "href"),
    State("tbl_error", "value"),
)
def tbl_update_error(href, error):
    req_error = apply_param(href, ctx.triggered_id, "error", error)
    assert len(valid_errors) > 0, "no valid errors"
    new_error = req_error if req_error in valid_errors else valid_errors[0]
    return new_error, valid_errors


@callback(
    Output("tbl_mbr", "value"),
    Input("tbl_url", "href"),
    State("tbl_mbr", "value"),
)
def tbl_update_mbr(href, mbr):
    req_mbr = apply_param(href, ctx.triggered_id, "mbr", mbr)
    assert isinstance(mbr, bool), "invalid mbr value"
    return req_mbr


@callback(
    Output("tbl_zscore", "value"),
    Input("tbl_url", "href"),
    State("tbl_zscore", "value"),
)
def tbl_update_zscore(href, zscore):
    req_zscore = apply_param(href, ctx.triggered_id, "zscore", zscore)
    assert isinstance(zscore, bool), "invalid zscore value"
    return req_zscore


@callback(
    Output("tbl_app_content", "children"),
    Output("tbl_url", "search"),
    Output("tbl_mbr", "label"),
    Input("tbl_config", "value"),
    Input("tbl_classifier", "value"),
    Input("tbl_acc", "value"),
    Input("tbl_error", "value"),
    Input("tbl_mbr", "value"),
    Input("tbl_zscore", "value"),
    Input("tbl_datasets", "value"),
    Input("tbl_methods", "value"),
    State("tbl_tree", "data"),
)
def tbl_update_content(config, classifier, acc, error, mbr, zscore, datasets, methods, tree):
    def get_search():
        return urlencode(
            dict(
                config=config,
                classifier=classifier,
                acc=acc,
                error=error,
                mbr=mbr,
                zscore=zscore,
                datasets=json.dumps(datasets),
                methods=json.dumps(methods),
            ),
            quote_via=quote,
        )

    search_str = f"?{get_search()}"

    if methods is None or len(methods) == 0:
        return [], search_str, mbr_label(mbr)

    report = get_report(config, classifier, acc, datasets, methods)

    if report is None:
        return [], search_str, mbr_label(mbr)

    df = get_df(report, error, method_by_row=mbr, zscore=zscore)
    table = get_Table(df, method_by_row=mbr)
    app_content = [] if table is None else [table]

    return app_content, search_str, mbr_label(mbr)
