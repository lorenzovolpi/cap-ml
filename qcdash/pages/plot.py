import json
import os
from json import JSONDecodeError
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlparse

import dash_bootstrap_components as dbc
import numpy as np
from dash import Input, Output, State, callback, ctx, dcc, html, register_page

import quacc as qc
from qcdash.navbar import APP_NAME
from quacc.experiments.report import Report
from quacc.plot.plotly import plot_diagonal, plot_shift

register_page(__name__, name=f"{APP_NAME} - plot", top_nav=True, path="/plot")

valid_plot_modes = ["shift[ae]", "shift[se]", "diagonal"]
root_folder = os.path.join(qc.env["OUT_DIR"], "results")


def get_fig(rep: Report, cls_name, acc_name, dataset_name, methods, mode):
    if mode == "diagonal":
        df = rep.diagonal_plot_data()
        return plot_diagonal(df, cls_name, acc_name, dataset_name)
    elif mode == "shift[ae]":
        df = rep.shift_plot_data()
        return plot_shift(df, cls_name, acc_name, dataset_name)
    elif mode == "shift[se]":
        df = rep.shift_plot_data(error=qc.error.se)
        return plot_shift(df, cls_name, acc_name, dataset_name)
    else:
        return None


def get_Graph(fig):
    if fig is None:
        return None

    return dcc.Graph(
        id="graph1",
        figure=fig,
        style={
            "margin": 0,
            "height": "94vh",
        },
    )


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
    datasets = kwargs.get("datasets", None)
    mode = kwargs.get("mode", None)
    methods = kwargs.get("methods", [])
    return [
        html.H4("Parameters:", style={"margin-bottom": "1vw"}),
        dbc.Row(
            [
                dbc.Label("config", width=3),
                dbc.Col(dbc.Select(id="config", value=config), width=9),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Label("classifier", width=3),
                dbc.Col(dbc.Select(id="classifier", value=classifier), width=9),
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Label("accuracy", width=3),
                dbc.Col(dbc.Select(id="acc", value=acc), width=9),
            ],
            className="mb-3",
        ),
        # dbc.Row(
        #     [
        #         dbc.Label("dataset", width=3),
        #         dbc.Col(dbc.Select(id="dataset", value=dataset), width=9),
        #     ],
        #     className="mb-3",
        # ),
        dbc.Row(
            [
                dbc.Label("plot", width=3),
                dbc.Col(dbc.Select(id="mode", value=mode), width=9),
            ],
            className="mb-3",
        ),
        dbc.Accordion(
            [
                dbc.AccordionItem([dbc.Checklist(id="datasets", value=datasets, switch=True)], title="Datasets"),
                dbc.AccordionItem([dbc.Checklist(id="methods", value=methods, switch=True)], title="Methods"),
            ],
            class_name="mb-1",
        ),
        # dbc.Col([dbc.Label("Methods"), dcc.Dropdown(id="methods", value=methods, multi=True)]),
    ]


def layout(**kwargs):
    layout = html.Div(
        [
            # dcc.Interval(id="reload", interval=10 * 60 * 1000),
            dcc.Location(id="url", refresh=False),
            dcc.Store(id="root", storage_type="session", data=root_folder),
            dcc.Store(id="tree", storage_type="session", data={}),
            dbc.Row(
                [
                    html.Div(get_sidebar(**kwargs), id="app_sidebar", style=sidebar_style),
                    html.Div(id="app_content", style=content_style),
                ],
                id="page_layout",
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
        case "url":
            params = parse_href(href)
            return params.get(id, None)
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
    Output("config", "value"),
    Output("config", "options"),
    Output("tree", "data"),
    Input("url", "href"),
)
def update_config(href):
    tree = build_tree()
    req_config = parse_href(href).get("config", None)
    valid_configs = get_valid_fields(tree, "config")
    assert len(valid_configs) > 0, "no valid configs"
    new_config = req_config if req_config in valid_configs else valid_configs[0]
    return new_config, valid_configs, tree


@callback(
    Output("classifier", "value"),
    Output("classifier", "options"),
    Input("url", "href"),
    Input("config", "value"),
    State("tree", "data"),
    State("classifier", "value"),
)
def update_classifier(href, config, tree, classifier):
    req_classifier = apply_param(href, ctx.triggered_id, "classifier", classifier)
    valid_classifiers = get_valid_fields(tree, "classifier", config)
    assert len(valid_classifiers) > 0, "no valid classifiers"
    new_classifier = req_classifier if req_classifier in valid_classifiers else valid_classifiers[0]
    return new_classifier, valid_classifiers


@callback(
    Output("acc", "value"),
    Output("acc", "options"),
    Input("url", "href"),
    Input("config", "value"),
    Input("classifier", "value"),
    State("tree", "data"),
    State("acc", "value"),
)
def update_acc(href, config, classifier, tree, acc):
    req_acc = apply_param(href, ctx.triggered_id, "acc", acc)
    valid_accs = get_valid_fields(tree, "acc", config, classifier)
    assert len(valid_accs) > 0, "no valid accs"
    default_acc = "vanilla_accuracy" if "vanilla_accuracy" in valid_accs else valid_accs[0]
    new_acc = req_acc if req_acc in valid_accs else default_acc
    return new_acc, valid_accs


@callback(
    Output("datasets", "value"),
    Output("datasets", "options"),
    Input("url", "href"),
    Input("config", "value"),
    Input("classifier", "value"),
    Input("acc", "value"),
    State("tree", "data"),
    State("datasets", "value"),
)
def update_dataset(href, config, classifier, acc, tree, datasets):
    req_datasets = apply_param(href, ctx.triggered_id, "datasets", datasets)
    if isinstance(req_datasets, str):
        try:
            req_datasets = json.loads(req_datasets)
        except JSONDecodeError:
            req_datasets = []
    valid_datasets = get_valid_fields(tree, "datasets", config, classifier, acc)
    assert len(valid_datasets) > 0, "no valid datasets"

    if req_datasets is None or len(req_datasets) == 0:
        return [], valid_datasets

    new_dataset = np.unique(np.array(req_datasets)[np.in1d(req_datasets, valid_datasets)]).tolist()
    return new_dataset, valid_datasets


@callback(
    Output("methods", "value"),
    Output("methods", "options"),
    Input("url", "href"),
    Input("config", "value"),
    Input("classifier", "value"),
    Input("acc", "value"),
    Input("datasets", "value"),
    State("tree", "data"),
    State("methods", "value"),
)
def update_methods(href, config, classifier, acc, datasets, tree, methods):
    req_methods = apply_param(href, ctx.triggered_id, "methods", methods)
    if isinstance(req_methods, str):
        try:
            req_methods = json.loads(req_methods)
        except JSONDecodeError:
            req_methods = []
    valid_methods = get_valid_fields(tree, "methods", config, classifier, acc, datasets)

    if req_methods is None or len(req_methods) == 0:
        return [], valid_methods

    new_methods = np.unique(np.array(req_methods)[np.in1d(req_methods, valid_methods)]).tolist()
    return new_methods, valid_methods


@callback(
    Output("mode", "value"),
    Output("mode", "options"),
    Input("url", "href"),
    State("mode", "value"),
)
def update_mode(href, mode):
    req_mode = apply_param(href, ctx.triggered_id, "mode", mode)
    valid_modes = valid_plot_modes
    new_mode = req_mode if req_mode in valid_modes else valid_modes[0]
    return new_mode, valid_modes


@callback(
    Output("app_content", "children"),
    Output("url", "search"),
    Input("config", "value"),
    Input("classifier", "value"),
    Input("acc", "value"),
    Input("datasets", "value"),
    Input("methods", "value"),
    Input("mode", "value"),
    State("tree", "data"),
)
def update_content(config, classifier, acc, datasets, methods, mode, tree):
    search = urlencode(
        dict(
            config=config,
            classifier=classifier,
            acc=acc,
            datasets=json.dumps(datasets),
            methods=json.dumps(methods),
            mode=mode,
        ),
        quote_via=quote,
    )
    search_str = f"?{search}"

    if datasets is None or len(datasets) == 0:
        return [], search_str

    if methods is None or len(methods) == 0:
        return [], search_str

    report = get_report(config, classifier, acc, datasets, methods)

    if report is None:
        return [], search_str

    fig = get_fig(report, config, classifier, acc, datasets, mode)
    g = get_Graph(fig)
    app_content = [] if g is None else [g]

    return app_content, search_str
