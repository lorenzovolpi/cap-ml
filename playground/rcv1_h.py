import json

import numpy as np
from quapy.data.base import LabelledCollection
from sklearn.datasets import fetch_rcv1

import quacc as qc
from quacc.utils.dataset import get_rcv1_class_info

class_names, tree, index = get_rcv1_class_info()

training = fetch_rcv1(subset="train", data_home=qc.env["SKLEARN_DATA"])
test = fetch_rcv1(subset="test", data_home=qc.env["SKLEARN_DATA"])

tr_len = training.target.shape[0]
te_len = test.target.shape[0]
print(f"{tr_len=} {te_len=}")


def parse_labels(labels, include_zero=False):
    """
    Parses multi-label structured labels to filter only those that are single labeled,
    collapsing them to a single-label structure and assigning unique values based on
    the available classes.

    :param labels: the labels to parse
    :param include_zero: wheather to include the "0" class from the original labels or not
    :return: a tuple with the valid indexes and the updated labels
    """
    if include_zero:
        valid = np.nonzero(np.sum(labels, axis=-1) <= 1)[0]
    else:
        valid = np.nonzero(np.sum(labels, axis=-1) == 1)[0]

    labels = labels[valid, :]
    ones, nonzero_vals = np.where(labels == np.ones((len(valid), 1)))
    labels = np.sum(labels, axis=-1)

    # if the 0 class must be included, shift remaining classed by 1 to make them unique
    nonzero_vals = nonzero_vals + 1 if include_zero else nonzero_vals

    labels[ones] = nonzero_vals
    return valid, labels


report = {}
for target in class_names + ["Root"]:
    # print(json.dumps({k: v.tolist() for k, v in tree.items()}, indent=2))
    # print(json.dumps({k: v.tolist() for k, v in index.items()}, indent=2))
    if target not in index:
        print(f"{target} is a leaf", end="\n\n")
        continue
    class_idx = index[target]
    tr_labels = training.target[:, class_idx].toarray()
    te_labels = test.target[:, class_idx].toarray()
    tr_valid, tr_labels = parse_labels(tr_labels)
    te_valid, te_labels = parse_labels(te_labels)
    classes = np.unique(np.hstack([tr_labels, te_labels]))

    if len(classes) <= 2:
        print(f"{target} skipped: only {len(classes)} classes", end="\n\n")
        continue

    n_tr_invalid, n_te_invalid = tr_len - len(tr_valid), te_len - len(te_valid)
    T = LabelledCollection(training.data[tr_valid, :], tr_labels, classes=classes)
    U = LabelledCollection(test.data[te_valid, :], te_labels, classes=classes)
    tr_prev, te_prev = T.prevalence(), U.prevalence()
    tr_counts, te_counts = T.counts(), U.counts()
    report[target] = dict(
        classes=classes.tolist(),
        n_tr_invalid=n_tr_invalid,
        n_te_invalid=n_te_invalid,
        tr_prev=tr_prev.tolist(),
        te_prev=te_prev.tolist(),
        tr_counts=tr_counts.tolist(),
        te_counts=te_counts.tolist(),
    )
    print(f"{target}: {json.dumps(report[target])}", end="\n\n")

with open("playground/rcv1_multi.json", "w") as f:
    json.dump(report, f, indent=2)
