import os
from collections import defaultdict

import numpy as np
from quapy.data.base import LabelledCollection
from quapy.data.datasets import TWITTER_SENTIMENT_DATASETS_TEST, UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method._kdey import KDEyML
from quapy.method.aggregative import ACC, CC, EMQ, PACC
from quapy.protocol import UPP
from sklearn.kernel_ridge import KernelRidge as KRR
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

from quacc.data.datasets import (
    HF_DATASETS,
    RCV1_BINARY_DATASETS,
    RCV1_MULTICLASS_DATASETS,
    fetch_HFDataset,
    fetch_RCV1BinaryDataset,
    fetch_RCV1MulticlassDataset,
    fetch_twitterDataset,
    fetch_UCIBinaryDataset,
    fetch_UCIMulticlassDataset,
)
from quacc.error import f1, f1_macro, vanilla_acc
from quacc.models._large_models import DistilBert
from quacc.models.cont_table import (
    LEAP,
    OCE,
    PHD,
    CAPContingencyTable,
    NaiveCAP,
    QuAcc1xN2,
    QuAcc1xNp1,
    QuAccNxN,
)
from quacc.models.direct import ATC, CAPDirect, DoC, PrediQuant
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.models.regression import ReQua
from quacc.utils.commons import get_results_path

_CC = defaultdict(lambda: False)
_CC |= dict(
    # LEAP=True,
    # OCE=True,
    # PHD=True,
)
_ACC = defaultdict(lambda: False)
_ACC |= dict(
    LEAP=True,
    # OCE=True,
    # PHD=True,
)
_SLD = defaultdict(lambda: False)
_SLD |= dict(
    # reDAN=True,
    # PQ=True,
    ReQua=False,
    # ReQua_conf=True,
    LEAP=True,
    # LEAP_OPT=True,
    OCE=True,
    PHD=True,
    QuAcc=True,
)
_KDEy = defaultdict(lambda: False)
_KDEy |= dict(
    # reDAN=True,
    # PQ=True,
    ReQua=False,
    # ReQua_conf=True,
    LEAP=True,
    # LEAP_OPT=True,
    OCE=True,
    PHD=True,
    QuAcc=True,
)


VANILLA = True
F1 = False


def sld():
    return EMQ(LR(), val_split=5)


def pacc():
    return PACC(LR())


def kdey():
    return KDEyML(LR())


def gen_classifiers():
    # param_grid = {"C": np.logspace(-4, -4, 9), "class_weight": ["balanced", None]}

    yield "LR", LR()
    # yield "LR-opt", GridSearchCV(LR(), param_grid, cv=5, n_jobs=qc.env["N_JOBS"])

    yield "KNN", KNN(n_neighbors=10)
    yield "SVM(rbf)", SVC(probability=True)
    yield "MLP", MLP(hidden_layer_sizes=(100, 15), max_iter=300, random_state=0)
    # yield "RFC", RFC()

    # yield 'NB', GaussianNB()
    # yield 'SVM(linear)', LinearSVC()


def gen_lm_classifiers():
    yield "DistilBert", DistilBert()


def gen_multi_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    # yields the UCI multiclass datasets
    _uci_skip = ["isolet", "wine-quality", "letter"]
    _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
    for dataset_name in _uci_names:
        yield dataset_name, None if only_names else fetch_UCIMulticlassDataset(dataset_name)

    # yields the 20 newsgroups dataset
    # yield "20news", None if only_names else fetch_20newsgroupsdataset()

    # yields the T1B@LeQua2022 (training) dataset
    # yield "T1B-LeQua2022", None if only_names else fetch_T1BLequa2022Dataset()

    # yields the RCV1 multiclass datasets
    for dataset_name in RCV1_MULTICLASS_DATASETS:
        yield dataset_name, None if only_names else fetch_RCV1MulticlassDataset(dataset_name)


def gen_tweet_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    for dataset_name in TWITTER_SENTIMENT_DATASETS_TEST:
        if only_names:
            yield dataset_name, None
        else:
            yield dataset_name, fetch_twitterDataset(dataset_name)


def gen_bin_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    # rcv1
    for dn in RCV1_BINARY_DATASETS:
        dval = None if only_names else fetch_RCV1BinaryDataset(dn)
        yield dn, dval
    # imdb
    # yield "imdb", None if only_names else fetch_IMDBDataset()
    # UCI
    _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
    _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
    for dn in _uci_names:
        dval = None if only_names else fetch_UCIBinaryDataset(dn)
        yield dn, dval


def gen_bin_lm_datasets(tokenizer, data_collator, only_names=False):
    for dataset_name in HF_DATASETS:
        yield dataset_name, None if only_names else fetch_HFDataset(dataset_name, tokenizer, data_collator)


def gen_lm_model_dataset(_gen_model, _gen_dataset):
    for model_name, model in _gen_model():
        for ds in _gen_dataset(model.tokenizer, model.data_collator):
            yield (model_name, model), ds


### baselines ###


def gen_CAP_baselines(acc_fn, config, mode_type) -> [str, CAPDirect]:
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")
    # yield 'ATC-NE', ATC(acc_fn, scoring_fn='neg_entropy')


def gen_CAP_baselines_vp(acc_fn, config, model_type, V2_prot, V2_prot_posteriors):
    yield "DoC", DoC(acc_fn, V2_prot, V2_prot_posteriors)


### models ###


# fmt: off
def gen_CAP_direct(h, acc_fn, config, model_type) -> [str, CAPDirect]:
    redan_q_params= {
        "classifier__C": np.logspace(-3, 3, 7),
        "classifier__class_weight": [None, "balanced"],
    }
    rdan_q_params_sld = redan_q_params | {"recalib": [None, "bcts"]}
    # rdan_q_params_kdey = redan_q_params | {"bandwidth": np.linspace(0.01, 0.2, 5)}
    ### CAP methods ###
    # yield 'SebCAP', SebastianiCAP(h, acc_fn, ACC)
    # yield 'SebCAPweight', SebastianiCAP(h, acc_fn, ACC, alpha=0)
    # yield "PrediQuant(ACC)", PrediQuant(h, acc_fn, ACC)
    # yield "PrediQuantWeight(ACC)", PrediQuant(h, acc_fn, ACC, alpha=0)
    # yield 'PabCAP', PabloCAP(h, acc_fn, ACC)
    if _SLD["PQ"] and model_type == "simple":
        yield "PrediQuant(SLD-ae)", PrediQuant(acc_fn, EMQ(h))
        yield "PrediQuantWeight(SLD-ae)", PrediQuant(acc_fn, EMQ(h), alpha=0)
    if _KDEy["PQ"] and model_type == "simple":
        yield "PrediQuant(KDEy-ae)", PrediQuant(acc_fn, KDEyML(h))
        yield "PrediQuantWeight(KDEy-ae)", PrediQuant(acc_fn, KDEyML(h), alpha=0)

def requa_params(acc_fn, reg, q_class, config, model_type, V2_prot, V2_prot_posteriros):
    quaccs = [
        QuAcc1xN2(acc_fn, q_class),
        QuAccNxN(acc_fn, q_class),
    ]
    if config == "binary":
        quaccs.append(QuAcc1xNp1(acc_fn, q_class))

    quacc_params = {
        # "q_class__classifier__C": np.logspace(-3, 3, 7),
        # "q_class__classifier__class_weight": [None, "balanced"],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }

    if model_type == "simple":
        quacc_params = quacc_params | {
            "add_X": [True, False],
        }
    elif model_type == "large":
        quacc_params = quacc_params | {
            "add_X": [False],
        }

    return acc_fn, reg, quaccs, quacc_params, V2_prot, V2_prot_posteriros

def gen_CAP_regression(acc_fn, config, model_type, V2_prot, V2_prot_posteriors):
    if _SLD["ReQua"]:
        yield "ReQua(SLD-LinReg)", ReQua(*requa_params(acc_fn, LinReg(), sld(), config, model_type, V2_prot, V2_prot_posteriors))
        yield "ReQua(SLD-Ridge)", ReQua(*requa_params(acc_fn, Ridge(), sld(), config, model_type, V2_prot, V2_prot_posteriors))
        yield "ReQua(SLD-KRR)", ReQua(*requa_params(acc_fn, KRR(), sld(), config, model_type, V2_prot, V2_prot_posteriors))
    if _SLD["ReQua_conf"]:
        yield "ReQua(SLD-LinReg)-conf", ReQua(*requa_params(acc_fn, LinReg(), sld(), config, model_type, V2_prot, V2_prot_posteriors), add_conf=True)
        yield "ReQua(SLD-Ridge)-conf", ReQua(*requa_params(acc_fn, Ridge(), sld(), config, model_type, V2_prot, V2_prot_posteriors), add_conf=True)
        yield "ReQua(SLD-KRR)-conf", ReQua(*requa_params(acc_fn, KRR(), sld(), config, model_type, V2_prot, V2_prot_posteriors), add_conf=True)
    if _KDEy["ReQua"]:
        yield "ReQua(KDEy-LinReg)", ReQua(*requa_params(acc_fn, LinReg(), kdey(), config, model_type, V2_prot, V2_prot_posteriors))
        yield "ReQua(KDEy-Ridge)", ReQua(*requa_params(acc_fn, Ridge(), kdey(), config, model_type, V2_prot, V2_prot_posteriors))
        yield "ReQua(KDEy-KRR)", ReQua(*requa_params(acc_fn, KRR(), kdey(), config, model_type, V2_prot, V2_prot_posteriors))
    if _KDEy["ReQua_conf"]:
        yield "ReQua(KDEy-LinReg)-conf", ReQua(*requa_params(acc_fn, LinReg(), kdey(), config, model_type, V2_prot, V2_prot_posteriors), add_conf=True)
        yield "ReQua(KDEy-Ridge)-conf", ReQua(*requa_params(acc_fn, Ridge(), kdey(), config, model_type, V2_prot, V2_prot_posteriors), add_conf=True)
        yield "ReQua(KDEy-KRR)-conf", ReQua(*requa_params(acc_fn, KRR(), kdey(), config, model_type, V2_prot, V2_prot_posteriors), add_conf=True)

def gen_CAP_cont_table(h, acc_fn, config, model_type) -> [str, CAPContingencyTable]:
    yield "Naive", NaiveCAP(acc_fn)
    # yield 'Equations-ACCh', NsquaredEquationsCAP(h, acc_fn, ACC, reuse_h=True)
    # yield 'Equations-ACC', NsquaredEquationsCAP(h, acc_fn, ACC)
    if _CC["LEAP"] and model_type == "simple":
        yield "LEAP(CC)", LEAP(acc_fn, CC(LR()), reuse_h=h)
    if _CC["OCE"] and model_type == "simple":
        yield "OCE(CC)", OCE(acc_fn, CC(LR()), reuse_h=h)
    if _CC["PHD"] and model_type == "simple":
        yield "PHD(CC)", PHD(acc_fn, CC(LR()), reuse_h=h)
    if _ACC["LEAP"] and model_type == "simple":
        yield "LEAP(ACC)", LEAP(acc_fn, ACC(LR()), reuse_h=h)
    if _ACC["OCE"] and model_type == "simple":
        yield "OCE(ACC)", OCE(acc_fn, ACC(LR()), reuse_h=h)
    if _ACC["PHD"] and model_type == "simple":
        yield "PHD(ACC)", PHD(acc_fn, ACC(LR()), reuse_h=h)
    if _SLD["LEAP"]:
        if model_type == "simple":
            yield "LEAP(SLD)", LEAP(acc_fn, sld(), reuse_h=h)
        # yield "LEAP+(SLD)", LEAP(acc_fn, sld())
    if _SLD["OCE"]:
        if model_type == "simple":
            yield "OCE(SLD)", OCE(acc_fn, sld(), reuse_h=h)
    if _SLD["PHD"]:
        yield "PHD(SLD)", PHD(acc_fn, sld(), reuse_h=h)
    if _KDEy["LEAP"]:
        # yield 'CT-PPS-KDE', ContTableTransferCAP(h, acc_fn, KDEyML(LogisticRegression(class_weight='balanced'), bandwidth=0.01))
        # yield 'CT-PPS-KDE05
        if model_type == "simple":
            yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h)
        # yield "LEAP+(KDEy)", LEAP(acc_fn, kdey())
    if _KDEy["OCE"]:
        if model_type == "simple":
            yield "OCE(KDEy)", OCE(acc_fn, kdey(), reuse_h=h)
    if _KDEy["PHD"]:
        yield "PHD(KDEy)", PHD(acc_fn, kdey(), reuse_h=h)
# fmt: on


# fmt: off
def gen_CAP_cont_table_opt(h, acc_fn, config, model_type, V2_prot, V2_prot_posteriors) -> [str, CAPContingencyTable]:
    pacc_lr_params = {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
        "add_posteriors": [True, False],
        "add_y_hat": [True, False],
        "add_maxconf": [True, False],
        "add_negentropy": [True, False],
        "add_maxinfsoft": [True, False],
    }

    if model_type == "simple":
        pacc_lr_params = pacc_lr_params | {
            "add_X": [True],
        }
    elif model_type == "large":
        pacc_lr_params = pacc_lr_params | {
            "add_X": [False]
        }
    
    emq_lr_params = pacc_lr_params | {"q_class__recalib": [None, "bcts"]}
    kde_lr_params = pacc_lr_params | {"q_class__bandwidth": np.linspace(0.01, 0.2, 5)}
    n2e_sld_h0_params = {"q_class__recalib": [None, "bcts"]}
    n2e_sld_hplus_params = n2e_sld_h0_params | {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
    }
    n2e_kde_h0_params = {"q_class__bandwidth": np.linspace(0.01, 0.2, 20)}
    n2e_kde_hplus_params = n2e_kde_h0_params | {
        "q_class__classifier__C": np.logspace(-3, 3, 7),
        "q_class__classifier__class_weight": [None, "balanced"],
    }

    if _SLD["QuAcc"]:
        yield "QuAcc(SLD)1xn2-OPT", GSCAP(QuAcc1xN2(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        yield "QuAcc(SLD)nxn-OPT", GSCAP(QuAccNxN(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        if config == "binary":
            yield "QuAcc(SLD)1xnp1-OPT", GSCAP(QuAcc1xNp1(acc_fn, sld()), emq_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    if _SLD["LEAP_OPT"]:
        if model_type == "simple":
            yield "LEAP(SLD)-OPT", GSCAP(LEAP(acc_fn, sld(), reuse_h=h), n2e_sld_h0_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        # yield "LEAP+(SLD)-OPT", GSCAP(LEAP(acc_fn, sld()), n2e_sld_hplus_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    if _KDEy["QuAcc"]:
        yield "QuAcc(KDEy)1xn2-OPT", GSCAP(QuAcc1xN2(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        yield "QuAcc(KDEy)nxn-OPT", GSCAP(QuAccNxN(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        if config == "binary":
            yield "QuAcc(KDEy)1xnp1-OPT", GSCAP(QuAcc1xNp1(acc_fn, kdey()), kde_lr_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    if _KDEy["LEAP_OPT"]:
        if model_type == "simple":
            yield "LEAP(KDEy)-OPT", GSCAP(LEAP(acc_fn, kdey(), reuse_h=h), n2e_kde_h0_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
        # yield "LEAP+(KDEy)-OPT", GSCAP(LEAP(acc_fn, kdey()), n2e_kde_hplus_params, V2_prot, V2_prot_posteriors, acc_fn, refit=False)
    # return
    # yield
# fmt: on


def gen_methods(h, V, V_posteriors, V1, V1_posteriors, V2_prot, V2_prot_posteriors, config, model_type="simple"):
    config = "multiclass" if config is None else config

    _, acc_fn = next(gen_acc_measure())

    for name, method in gen_CAP_baselines(acc_fn, config, model_type):
        yield name, method, V, V_posteriors
    for name, method in gen_CAP_baselines_vp(acc_fn, config, model_type, V2_prot, V2_prot_posteriors):
        yield name, method, V1, V1_posteriors
    for name, method in gen_CAP_direct(h, acc_fn, config, model_type):
        yield name, method, V, V_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn, config, model_type):
        yield name, method, V, V_posteriors

    for name, method in gen_CAP_cont_table_opt(h, acc_fn, config, model_type, V2_prot, V2_prot_posteriors):
        yield name, method, V1, V1_posteriors
    for name, method in gen_CAP_regression(acc_fn, config, model_type, V2_prot, V2_prot_posteriors):
        yield name, method, V1, V1_posteriors


def get_method_names(config, model_type="simple"):
    mock_h = LR()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_V2_prot = UPP(None)
    mock_V2_post = np.empty((1,))
    return (
        [m for m, _ in gen_CAP_baselines(mock_acc_fn, config, model_type)]
        + [m for m, _ in gen_CAP_baselines_vp(mock_acc_fn, config, model_type, mock_V2_prot, mock_V2_post)]
        + [m for m, _ in gen_CAP_direct(mock_h, mock_acc_fn, config, model_type)]
        + [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn, config, model_type)]
        + [m for m, _ in gen_CAP_cont_table_opt(mock_h, mock_acc_fn, config, model_type, mock_V2_prot, mock_V2_post)]
        + [m for m, _ in gen_CAP_regression(mock_acc_fn, config, model_type, mock_V2_prot, mock_V2_post)]
    )


def gen_acc_measure(multiclass=False):
    if VANILLA:
        yield "vanilla_accuracy", vanilla_acc
    if F1:
        yield "macro-F1", f1_macro if multiclass else f1


def any_missing(rootdir, basedir, cls_name, dataset_name, method_name):
    for acc_name, _ in gen_acc_measure():
        if not os.path.exists(get_results_path(rootdir, basedir, cls_name, acc_name, dataset_name, method_name)):
            return True
    return False
