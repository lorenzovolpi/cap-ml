import itertools as IT

import numpy as np
import pandas as pd

from exp.trd.config import get_acc_names, get_CAP_method_names, get_classifier_class_names, get_classifier_names


def CAP_model_selection(res: pd.DataFrame, method: str, classifier_class=None):
    # methods = get_CAP_method_names()
    accs = get_acc_names()

    dfs = []
    # apply model selection for each method and each acc measure
    for acc_name in accs:
        # filter by method and acc and make a copy
        if classifier_class is None:
            mdf = res.loc[(res["method"] == method) & (res["acc_name"] == acc_name), :].copy()
        else:
            mdf = res.loc[
                (res["method"] == method)
                & (res["acc_name"] == acc_name)
                & (res["classifier_class"] == classifier_class),
                :,
            ].copy()

        # index data by dataset, sample_id (uids), and classifier
        mdf = mdf.set_index(["dataset", "uids", "classifier"])
        # group data by sample_id and dataset and take the index of the minimum in the "estim_accs" column
        best_idx = mdf.groupby(["uids", "dataset"])["estim_accs"].idxmax()
        # use the index to filter the data, resetting the index
        mdf = mdf.loc[best_idx, :].reset_index(drop=False)
        if classifier_class is not None:
            mdf["method"] = [f"{method}-{classifier_class}"] * len(mdf)
        dfs.append(mdf)

    return pd.concat(dfs, axis=0)


def oracle_model_selection(res: pd.DataFrame):
    accs = get_acc_names()

    dfs = []
    for acc in accs:
        odf = res.loc[res["acc_name"] == acc, :].groupby(["dataset", "uids", "classifier"]).first()
        best_idx = odf.groupby(["uids", "dataset"])["true_accs"].idxmax()
        odf = odf.loc[best_idx, :].reset_index(drop=False)
        odf["method"] = ["oracle"] * len(odf)
        dfs.append(odf)

    return pd.concat(dfs, axis=0)


def no_model_selection(res: pd.DataFrame, only_default=False):
    accs = get_acc_names()
    classifiers = get_classifier_names()

    dfs = []
    for acc, classifier in IT.product(accs, classifiers):
        ndf = (
            res.loc[(res["acc_name"] == acc) & (res["classifier"] == classifier), :]
            .groupby(["dataset", "uids"])
            .first()
            .reset_index(drop=False)
        )
        if only_default and not np.all(ndf["default_c"].to_numpy()):
            continue
        ndf["method"] = ndf["classifier"]
        dfs.append(ndf)

    return pd.concat(dfs, axis=0)


def model_selection(res, oracle=False, only_default=False):
    classifier_classes = get_classifier_class_names()
    spread_methods = ["Naive"]
    methods = get_CAP_method_names()

    dfs = (
        [no_model_selection(res, only_default=only_default)]
        + [
            CAP_model_selection(res, method=m, classifier_class=cls_class)
            for m, cls_class in IT.product(spread_methods, classifier_classes)
        ]
        + [CAP_model_selection(res, method=m) for m in methods]
    )

    if oracle:
        dfs.append(oracle_model_selection(res))

    return pd.concat(dfs, axis=0)
