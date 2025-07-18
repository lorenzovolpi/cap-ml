import numpy as np
import quapy as qp
from datasets import load_dataset
from quapy.data.base import Dataset, LabelledCollection
from quapy.data.datasets import fetch_lequa2022
from quapy.data.datasets import fetch_UCIBinaryDataset as UCIBin
from quapy.data.datasets import fetch_UCIMulticlassDataset as UCIMulti
from sklearn.datasets import fetch_20newsgroups, fetch_rcv1
from sklearn.feature_extraction.text import TfidfVectorizer

from quacc.data._cifar import fetch_cifar10, fetch_cifar100
from quacc.data.util import get_rcv1_class_info, hf_dataset_map, preprocess_hf_dataset, split_train
from quacc.environment import env

# fmt: off
RCV1_BINARY_DATASETS = [
    "CCAT", "GCAT", "MCAT", "ECAT", 
    # "C151", "GCRIM", "M131", "E41",
]
RCV1_MULTICLASS_DATASETS = [
    # "C18", "C31", "E51", "M14",  # 3 classes
    # "C17", "C2", "C3", "E1", "M1", "Root",  # 4 classes
    # "C1",  # 8 classes
    # "ECAT", # 7 classes
    "Root", # 4 classes
    "CCAT", # 4 classes
    "M1", # 4 classes
    "M14", # 3 classes
]
HF_DATASETS = ["imdb", "rotten_tomatoes", "amazon_polarity"]
# fmt: on


def fetch_IMDBDataset(train_val_split=0.5):
    train, U = qp.datasets.fetch_reviews(
        "imdb", tfidf=True, min_df=10, pickle=True, data_home=env["QUAPY_DATA"]
    ).train_test
    T, V = split_train(train, train_val_split)
    return T, V, U


def fetch_RCV1BinaryDataset(target, train_val_split=0.5):
    training = fetch_rcv1(subset="train", data_home=env["SKLEARN_DATA"])
    test = fetch_rcv1(subset="test", data_home=env["SKLEARN_DATA"])

    assert target in RCV1_BINARY_DATASETS, f"invalid class {target}"

    class_names = training.target_names.tolist()
    class_idx = class_names.index(target)
    tr_labels = training.target[:, class_idx].toarray().flatten()
    te_labels = test.target[:, class_idx].toarray().flatten()
    tr = LabelledCollection(training.data, tr_labels)
    U = LabelledCollection(test.data, te_labels)
    T, V = split_train(tr, train_val_split)
    return T, V, U


def fetch_RCV1MulticlassDataset(target, train_val_split=0.5, include_zero=False):
    """Retrieve a multiclass dataset extracted from the RCV1 taxonomy.

    :param target: the parent of the classes that define the dataset
    :param include_zero: whether to include the datapoints not belonging to target
    :return: a tuple with training, validation and test sets.
    """

    def extend_labels(orig_labels, orig_cns, ext_cns):
        ext_cns = np.asarray(ext_cns)
        sorted_ext_idx = np.argsort(ext_cns)
        sorted_ext = ext_cns[sorted_ext_idx]
        subset_idx = np.searchsorted(sorted_ext, orig_cns)
        ext_labels = np.zeros((orig_labels.shape[0], ext_cns.shape[0]))
        ext_labels[:, subset_idx] = orig_labels

        for name in ext_cns:
            if name not in orig_cns:
                ext_idx = np.where(ext_cns == name)[0][0]
                new_lbl = np.sum(ext_labels[:, index[name]], axis=-1)
                new_lbl[np.where(new_lbl > 0)[0]] = 1.0
                ext_labels[:, ext_idx] = new_lbl

        return ext_labels

    def parse_labels(labels):
        if include_zero:
            valid_idx = np.nonzero(np.sum(labels, axis=-1) <= 1)[0]
        else:
            valid_idx = np.nonzero(np.sum(labels, axis=-1) == 1)[0]

        labels = labels[valid_idx, :]
        ones_idx, nonzero_vals = np.where(labels == np.ones((len(valid_idx), 1)))
        labels = np.sum(labels, axis=-1, dtype=np.int64)

        # if the 0 class must be included, shift remaining classed by 1 to make them unique
        nonzero_vals = nonzero_vals + 1 if include_zero else nonzero_vals

        labels[ones_idx] = nonzero_vals
        return valid_idx, labels

    class_names, _, index = get_rcv1_class_info()

    # assert target in RCV1_MULTICLASS_DATASETS, f"invalid class {target}"

    training = fetch_rcv1(subset="train", data_home=env["SKLEARN_DATA"])
    test = fetch_rcv1(subset="test", data_home=env["SKLEARN_DATA"])
    tr_labels = extend_labels(training.target.toarray(), training.target_names, class_names)
    te_labels = extend_labels(test.target.toarray(), test.target_names, class_names)
    class_idx = index[target]
    tr_labels = tr_labels[:, class_idx]
    te_labels = te_labels[:, class_idx]
    tr_valid_idx, tr_labels = parse_labels(tr_labels)
    te_valid_idx, te_labels = parse_labels(te_labels)
    tr = LabelledCollection(training.data[tr_valid_idx, :], tr_labels)
    U = LabelledCollection(test.data[te_valid_idx, :], te_labels)
    T, V = split_train(tr, train_val_split)
    return T, V, U


def fetch_RCV1WholeDataset(train_val_split=0.5, min_docs=5):
    """Retrieve the whole multiclass dataset extracted from the RCV1 taxonomy.
    For each class, labels from the upper hierarchy are removed, so that each document
    is labelled with the most specific class of competence.
    Other multi-label datapoints are removed from the dataset to keep it single-label.

    :param train_val_split: the propoertion to use for splitting the source set
    in training and validation.
    :return: a tuple with training, validation and test sets.
    """

    def find_parents(c, rev_tree, sorted_classes):
        if c not in rev_tree:
            return []

        try:
            parent = rev_tree[c]
            parent_idx = sorted_classes.index(parent)
            return [parent_idx] + find_parents(parent, rev_tree, sorted_classes)
        except ValueError:
            return find_parents(parent, rev_tree, sorted_classes)

    def filter_multi_label(dset, rev_tree, sorted_classes):
        labels = dset.target.toarray()
        for i, c in enumerate(sorted_classes):
            idx_i = labels[:, i] == 1
            parents = np.asarray(find_parents(c, rev_tree, sorted_classes))
            if len(parents) == 0:
                continue
            labels[idx_i, parents[:, np.newaxis]] = 0

        valid_data = labels.sum(axis=1) == 1
        return LabelledCollection(
            dset.data[valid_data, :],
            np.argmax(labels[valid_data, :], axis=1),
        )

    def filter_min_docs(train: LabelledCollection, U: LabelledCollection):
        all_classes_ = np.sort(np.unique(np.hstack([train.classes_, U.classes_])))
        _bins = np.hstack([all_classes_, [all_classes_.max() + 1]])

        train_valid_classes = np.histogram(train.y, bins=_bins)[0] >= min_docs
        U_valid_classes = np.histogram(U.y, bins=_bins)[0] >= min_docs
        valid_classes = all_classes_[train_valid_classes & U_valid_classes]

        # compute the indexes for the valid documents in train and test given the valid classes
        train_idx = np.any(train.y == valid_classes[:, np.newaxis], axis=0)
        U_idx = np.any(U.y == valid_classes[:, np.newaxis], axis=0)

        train_X = train.X[train_idx, :]
        train_y = train.y[train_idx]
        U_X = U.X[U_idx, :]
        U_y = U.y[U_idx]

        compr_classes = np.zeros(valid_classes.max() + 1)
        compr_classes[valid_classes] = np.arange(len(valid_classes))

        for c in valid_classes:
            train_y[train_y == c] = compr_classes[c]
            U_y[U_y == c] = compr_classes[c]

        return LabelledCollection(train_X, train_y, classes=np.arange(len(valid_classes))), LabelledCollection(
            U_X, U_y, classes=np.arange(len(valid_classes))
        )

    _, tree, _ = get_rcv1_class_info()
    # reverse tree
    rev_tree = {}
    for c, scs in tree.items():
        for sc in scs:
            rev_tree[str(sc)] = c

    og_training = fetch_rcv1(subset="train", data_home=env["SKLEARN_DATA"])
    og_test = fetch_rcv1(subset="test", data_home=env["SKLEARN_DATA"])

    sorted_classes = og_training.target_names.tolist()

    train = filter_multi_label(og_training, rev_tree, sorted_classes)
    U = filter_multi_label(og_test, rev_tree, sorted_classes)

    train, U = filter_min_docs(train, U)

    T, V = split_train(train, train_val_split)
    return T, V, U
    # return len(train), len(U), train.n_classes


def fetch_cifar10Dataset(target, train_val_split=0.5):
    dataset = fetch_cifar10()
    available_targets: list = dataset.label_names

    if target is None or target not in available_targets:
        raise ValueError(f"Invalid target {target}")

    target_idx = available_targets.index(target)
    train_d = dataset.train.data
    train_l = (dataset.train.labels == target_idx).astype(int)
    test_d = dataset.test.data
    test_l = (dataset.test.labels == target_idx).astype(int)
    train = LabelledCollection(train_d, train_l, classes=[0, 1])
    U = LabelledCollection(test_d, test_l, classes=[0, 1])
    T, V = split_train(train, train_val_split)

    return T, V, U


def fetch_cifar100Dataset(target, train_val_split=0.5):
    dataset = fetch_cifar100()
    available_targets: list = dataset.coarse_label_names

    if target is None or target not in available_targets:
        raise ValueError(f"Invalid target {target}")

    target_index = available_targets.index(target)
    train_d = dataset.train.data
    train_l = (dataset.train.coarse_labels == target_index).astype(int)
    test_d = dataset.test.data
    test_l = (dataset.test.coarse_labels == target_index).astype(int)
    train = LabelledCollection(train_d, train_l, classes=[0, 1])
    U = LabelledCollection(test_d, test_l, classes=[0, 1])
    T, V = split_train(train, train_val_split)

    return T, V, U


def fetch_twitterDataset(dataset_name, data_home=env["QUAPY_DATA"], train_val_split=0.5):
    train, U = qp.datasets.fetch_twitter(dataset_name, min_df=3, pickle=True, data_home=data_home)
    T, V = split_train(train, train_val_split)
    return T, V, U


def fetch_UCIBinaryDataset(dataset_name, data_home=env["QUAPY_DATA"], train_val_split=0.5):
    train, U = UCIBin(dataset_name, data_home=data_home).train_test
    T, V = split_train(train, train_val_split)
    return T, V, U


def fetch_UCIMulticlassDataset(dataset_name, data_home=env["QUAPY_DATA"], train_val_split=0.5):
    train, U = UCIMulti(dataset_name, data_home=data_home).train_test
    T, V = split_train(train, train_val_split)
    return T, V, U


def fetch_20newsgroupsDataset(train_val_split=0.5):
    train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), data_home=env["SKLEARN_DATA"])
    test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), data_home=env["SKLEARN_DATA"])
    tfidf = TfidfVectorizer(min_df=5, sublinear_tf=True)
    Xtr = tfidf.fit_transform(train.data)
    Xte = tfidf.transform((test.data))
    train = LabelledCollection(instances=Xtr, labels=train.target)
    U = LabelledCollection(instances=Xte, labels=test.target)
    T, V = split_train(train, train_val_split)
    return T, V, U


def fetch_T1BLequa2022Dataset(train_val_split=0.5, test_split=0.3):
    dataset, _, _ = fetch_lequa2022(task="T1B", data_home=env["QUAPY_DATA"])
    train, U = Dataset.SplitStratified(dataset, train_size=1 - test_split)
    L, V = split_train(train, train_val_split)
    return L, V, U


def fetch_HFDataset(dataset_name, tokenizer, data_collator, train_length=None):
    if dataset_name not in hf_dataset_map:
        raise ValueError(f"HuggingFace dataset {dataset_name} not supported yet")

    text_columns, default_train_length = hf_dataset_map[dataset_name]
    train_length = default_train_length if train_length is None else train_length
    dataset = load_dataset(dataset_name)

    train = preprocess_hf_dataset(dataset, "train", tokenizer, data_collator, text_columns, length=train_length)
    U = preprocess_hf_dataset(dataset, "test", tokenizer, data_collator, text_columns)
    L, V = train.split_stratified(train_prop=0.5, random_state=qp.environ["_R_SEED"])

    return L, V, U


def sort_datasets_by_size(dataset_names, dataset_fun, descending=True):
    def get_dataset_size(name):
        L, V, U = dataset_fun(name)
        return len(L) + len(V) + len(U)

    datasets = [(d, get_dataset_size(d)) for d in dataset_names]
    datasets.sort(key=(lambda d: d[1]), reverse=descending)
    return [d for (d, _) in datasets]
