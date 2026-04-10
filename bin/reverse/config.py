from collections import defaultdict
from typing import Callable, Iterable, Literal, Tuple

import numpy as np
from quapy.method.aggregative import KDEyML
from sklearn.neural_network import MLPClassifier as MLP

from bin.reverse.util import all_results_exist
from cap.error import f1, f1_macro, k_bin, k_macro, smooth, vanilla_acc
from cap.models.base import CAP
from cap.models.cont_table import O_LEAP
from cap.models.direct import DoC
from cap_exp.pretrain.data import DatasetBundle, PretrainInfo, load_info_paths
from cap_exp.pretrain.dataset import sort_datasets_by_size


def kdey():
    return KDEyML(MLP())


def get_acc_names():
    return ["vanilla_accuracy", "macro-F1", "macro-K"]


def get_selection_acc(name: str, multiclass: bool) -> Callable:
    return {
        "vanilla_accuracy": vanilla_acc,
        "macro-F1": (smooth(f1_macro) if multiclass else smooth(f1)),
        "macro-K": (k_macro if multiclass else k_bin),
    }[name]


def get_evaluation_acc(name: str, is_multiclass: bool) -> Callable:
    return {
        "vanilla_accuracy": vanilla_acc,
        "macro-F1": (f1_macro if is_multiclass else f1),
        "macro-K": (k_macro if is_multiclass else k_bin),
    }[name]


def acc_from_ct(acc_name: str, ct: np.ndarray, type: Literal["selection", "evaluation"] = "selection") -> float:
    n_classes = ct.shape[0]
    if type == "selection":
        return get_selection_acc(acc_name, n_classes > 2)(ct)
    elif type == "evaluation":
        return get_evaluation_acc(acc_name, n_classes > 2)(ct)


def gen_acc_measure(is_multiclass: bool):
    for acc in get_acc_names():
        yield acc, get_selection_acc(acc, is_multiclass)


def gen_methods() -> Iterable[Tuple[str, CAP]]:
    _, acc = next(gen_acc_measure(True))
    yield "O-LEAP", O_LEAP(acc, kdey())
    yield "DoC", DoC(acc)


def get_existing_dataset_names(experiment: str, domain: str, sort=True):
    info_paths = load_info_paths(domain=domain)
    dataset_h_map = defaultdict(lambda: True)
    for path in info_paths:
        p = PretrainInfo.load(path, fast=True)
        d_info, h_info = p.d_info, p.h_info
        # if not dataset_h_map[D.name]:
        #     continue
        # problem = "multiclass" if d_info.n_classes > 2 else "binary"
        _key = (d_info.name, d_info.collection)
        dataset_h_map[_key] = dataset_h_map[_key] and all_results_exist(
            p.domain, d_info.name, h_info.full_name, get_method_names(), get_acc_names(), experiment
        )

    datasets = [dc for dc, all_exist in dataset_h_map.items() if all_exist]
    dataset_names, dataset_colls = tuple(map(lambda x: list(x), zip(*datasets)))
    if sort:
        return sort_datasets_by_size(dataset_colls, dataset_names)
    else:
        return dataset_names


def get_method_names():
    names = [m for m, _ in gen_methods()]

    return names
