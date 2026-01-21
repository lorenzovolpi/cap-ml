from typing import Callable

import pandas as pd
from quapy.data.datasets import UCI_MULTICLASS_DATASETS
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier as MLP

import exp.direct_ours.env as env
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import vanilla_acc
from cap.models.cont_table import O_LEAP
from cap.models.direct import RQBS, PrediQuant
from exp.direct_ours.data import Clsf, DatasetBundle
from exp.direct_ours.util import kdey


def gen_acc_measures():
    yield "vanilla_accuracy", vanilla_acc


# fmt: off
def gen_methods(acc_fn: Callable, cl: Clsf, D: DatasetBundle):
    yield "PrediQuant", PrediQuant(acc_fn, kdey(cl.h), D.V2_prot, D.V2_prot_posteriors, alpha=0.3), D.V1, D.V1_posteriors
    yield "PrediQuantMM", PrediQuant(acc_fn, kdey(cl.h), D.V2_prot, D.V2_prot_posteriors, alpha=0.3), D.V1, D.V1_posteriors,
    yield "RQBS", RQBS(acc_fn, kdey()), D.V, D.V_posteriors
    yield "O-LEAP", O_LEAP(acc_fn, kdey()), D.V, D.V_posteriors

# fmt: on


def gen_datasets():
    if env.PROBLEM == "binary":
        uci_binary = [
            "breast-cancer",
            "german",
            "haberman",
            "ionosphere",
            "mammographic",
            "semeion",
            "sonar",
            "spambase",
            "spectf",
            "tictactoe",
            "transfusion",
            "wdbc",
        ]
        for dn in uci_binary:
            yield dn, fetch_UCIBinaryDataset(dn)
    elif env.PROBLEM == "multiclass":
        for dn in UCI_MULTICLASS_DATASETS:
            yield dn, fetch_UCIMulticlassDataset(dn)


def gen_classifiers():
    yield "LR", LogisticRegression()
    yield "MLP", MLP()
