import functools
import json
import os
import subprocess as sb
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Literal

import datasets
import numpy as np
import quapy as qp
import requests
from datasets import Dataset, DatasetDict, Image, concatenate_datasets
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from tqdm import tqdm

import cap.environment as capenv
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap_exp.util import get_dataset_path, temp_np_seed


def save_dataset(domain, dataset_name, model_name, classes, train_prev, embeds, labels):
    path = get_dataset_path(domain, dataset_name, model_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = dict(
        classes=classes,
        L_prevalence=train_prev,
        validation_X=embeds["validation"],
        validation_y=labels["validation"],
        test_X=embeds["test"],
        test_y=labels["test"],
    )
    np.savez_compressed(path, **d)


def load_dataset(domain, dataset_name, model_name=None):
    path = get_dataset_path(domain, dataset_name, model_name)
    _data = np.load(path)
    _classes = _data["classes"]
    L_prevalence = _data["L_prevalence"]
    V = LabelledCollection(instances=_data["validation_X"], labels=_data["validation_y"], classes=_classes)
    U = LabelledCollection(instances=_data["test_X"], labels=_data["test_y"], classes=_classes)
    return L_prevalence, V, U


def local_dataset_dir():
    cap_dir = capenv["CAP_DATA"]
    return os.path.join(cap_dir, "raw")


def download_imagenet_split(name):
    urls = {}
    basedir = local_dataset_dir()
    with open(os.path.join(basedir, "imagenet_urls.txt"), "r") as f:
        for line in f.readlines():
            _split, _url = tuple(map(lambda s: s.strip(), line.strip().split(" ", maxsplit=1)))
            urls[_split] = _url
    url = urls[name]

    dest_dir = os.path.join(basedir, "imagenet-1k")
    os.makedirs(dest_dir, exist_ok=True)

    filename = url.split("/")[-1]
    path = os.path.join(dest_dir, filename)

    if os.path.exists(path):
        return

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Controlla se ci sono errori HTTP

            with open(path, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(chunk_size=8192),
                    desc=f"{name}",
                    total=int(r.headers.get("content-length", 0)) // 8192,
                ):
                    if chunk:  # filtra i keep-alive chunks
                        f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Errore durante il download di {name}: {e}")


def download_rcv1_raw():
    base_url = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/"
    filenames = [
        "a12-token-files/lyrl2004_tokens_test_pt0.dat.gz",
        "a12-token-files/lyrl2004_tokens_test_pt1.dat.gz",
        "a12-token-files/lyrl2004_tokens_test_pt2.dat.gz",
        "a12-token-files/lyrl2004_tokens_test_pt3.dat.gz",
        "a12-token-files/lyrl2004_tokens_train.dat.gz",
        "a08-topic-qrels/rcv1-v2.topics.qrels.gz",
        "a02-orig-topics-hierarchy/rcv1.topics.hier.orig",
    ]

    dest_dir = os.path.join(local_dataset_dir(), "rcv1_raw")
    os.makedirs(dest_dir, exist_ok=True)

    for fn in filenames:
        url = base_url + fn
        f = fn.split("/")[-1]
        path = os.path.join(dest_dir, f)
        if os.path.exists(path) or os.path.exists(path[:-3]):
            continue

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()  # Controlla se ci sono errori HTTP

                with open(path, "wb") as f:
                    for chunk in tqdm(
                        r.iter_content(chunk_size=8192),
                        desc=f"{f}",
                        total=int(r.headers.get("content-length", 0)) // 8192,
                    ):
                        if chunk:  # filtra i keep-alive chunks
                            f.write(chunk)

        except requests.exceptions.RequestException as e:
            print(f"Errore durante il download di {f}: {e}")

        if path.endswith(".gz"):
            sb.run(f"gzip -d {path}", shell=True)


def download_caltech_256():
    url = "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1"

    dest_dir = os.path.join(local_dataset_dir(), "caltech_256_raw")
    if os.path.exists(dest_dir):
        return
    os.makedirs(dest_dir, exist_ok=True)
    filename = url.split("/")[-1].split("?")[0]
    path = os.path.join(dest_dir, filename)

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()  # Controlla se ci sono errori HTTP

            with open(path, "wb") as f:
                for chunk in tqdm(
                    r.iter_content(chunk_size=8192),
                    desc=f"{filename}",
                    total=int(r.headers.get("content-length", 0)) // 8192,
                ):
                    if chunk:  # filtra i keep-alive chunks
                        f.write(chunk)

    except requests.exceptions.RequestException as e:
        print(f"Errore durante il download di {filename}: {e}")

    sb.run(f"tar -xf {path}", shell=True)
    sb.run(f"rm {path}", shell=True)


def extract_imagenet_train():
    name = "train"
    filename = "ILSVRC2012_img_train.tar"
    dest_dir = os.path.join(local_dataset_dir(), "imagenet-1k")
    os.makedirs(dest_dir, exist_ok=True)

    split_dir = os.path.join(dest_dir, name)
    if os.path.exists(split_dir) and len(glob(os.path.join(split_dir, "*"))) > 1:
        return
    os.makedirs(split_dir, exist_ok=True)

    tar_path = os.path.join(dest_dir, filename)
    if not os.path.exists(tar_path):
        download_imagenet_split(name)

    sb.run(f"tar -xf {tar_path} -C {split_dir}", shell=True)
    sb.run(f"rm {tar_path}", shell=True)
    for path in tqdm(glob(os.path.join(split_dir, "*.tar")), desc=f"extracting {name}"):
        class_dir = os.path.join(split_dir, os.path.basename(path).split(".")[0])
        os.makedirs(class_dir, exist_ok=True)
        sb.run(f"tar -xf {path} -C {class_dir}", shell=True)
        sb.run(f"rm {path}", shell=True)


def extract_imagenet_val():
    name = "val"
    filename = "ILSVRC2012_img_val.tar"
    dest_dir = os.path.join(local_dataset_dir(), "imagenet-1k")
    os.makedirs(dest_dir, exist_ok=True)

    split_dir = os.path.join(dest_dir, name)
    if os.path.exists(split_dir) and len(glob(os.path.join(split_dir, "*"))) > 1:
        return
    os.makedirs(split_dir, exist_ok=True)

    lt_tes_path = os.path.join(local_dataset_dir(), "ImageNet_LT_test.txt")
    class_map = {}
    with open(lt_tes_path, "r") as f:
        for line in f.readlines():
            dest_path, _ = tuple(map(lambda s: s.strip(), line.strip().split(" ", maxsplit=1)))
            parts = dest_path.split("/")
            class_map[parts[2]] = parts[1]

    tar_path = os.path.join(dest_dir, filename)
    if not os.path.exists(tar_path):
        download_imagenet_split(name)

    sb.run(f"tar -xf {tar_path} -C {split_dir}", shell=True)
    sb.run(f"rm {tar_path}", shell=True)
    for path in tqdm(glob(os.path.join(split_dir, "*.JPEG")), desc=f"fixing {name}"):
        img_name = os.path.basename(path)
        class_name = class_map[img_name]
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        dest_path = os.path.join(class_dir, img_name)
        os.rename(path, dest_path)


def get_imagenet():
    extract_imagenet_train()
    extract_imagenet_val()


def build_imagenet_lt():
    def load_set(split) -> tuple[np.ndarray, np.ndarray]:
        path = os.path.join(local_dataset_dir(), f"ImageNet_LT_{split}.txt")
        X, y = [], []
        with open(path, "r") as f:
            for line in f.readlines():
                id, label = tuple(map(lambda s: s.strip(), line.strip().split(" ", maxsplit=1)))
                X.append(id)
                y.append(int(label))

        X = np.asarray(X)
        y = np.asarray(y)

        return X, y

    source_dir = os.path.join(local_dataset_dir(), "imagenet-1k")
    dest_dir = os.path.join(local_dataset_dir(), "imagenet-lt")
    if os.path.exists(dest_dir):
        return
    os.makedirs(dest_dir, exist_ok=True)

    get_imagenet()

    train_X, train_y = load_set("train")
    val_X, val_y = load_set("val")
    test_X, test_y = load_set("test")

    # NOTE: val is concatenated with train and then re-extracted with stratification
    # to ensure that train and val have the same distribution; this flattens the curve
    # and reduces the imbalance ratio (IR), but is essential for our experimental setup
    trainval_X = np.concatenate([train_X, val_X])
    trainval_y = np.concatenate([train_y, val_y])
    # ns_train_X, ns_val_X, ns_train_y, ns_val_y = train_test_split(
    #     trainval_X,
    #     trainval_y,
    #     test_size=val_size,
    #     random_state=qp.environ["_R_SEED"],
    #     stratify=trainval_y,
    # )

    def fix_image_path(p):
        return os.path.join(source_dir, p)

    train_dict = {"image": list(map(fix_image_path, trainval_X.tolist())), "label": trainval_y.tolist()}
    # val_dict = {"image": list(map(fix_image_path, ns_val_X.tolist())), "label": ns_val_y.tolist()}
    test_dict = {"image": list(map(fix_image_path, test_X.tolist())), "label": test_y.tolist()}

    train_ds = Dataset.from_dict(train_dict).cast_column("image", Image())
    test_ds = Dataset.from_dict(test_dict).cast_column("image", Image())
    # val_ds = Dataset.from_dict(val_dict).cast_column("image", Image())

    dataset = DatasetDict(
        {
            "train": train_ds,
            # "validation": val_ds,
            "test": test_ds,
        }
    )

    dataset.save_to_disk(dest_dir)


def cut_imagenet_lt(n_classes=200):
    set_dir = os.path.join(local_dataset_dir(), "imagenet-lt")
    cut_dir = os.path.join(local_dataset_dir(), f"imagenet-lt{n_classes}")
    if os.path.exists(cut_dir):
        return
    os.makedirs(cut_dir, exist_ok=True)
    ds = datasets.load_from_disk(set_dir)

    classes_map = defaultdict(int)
    for lbl in ds["train"]["label"]:
        classes_map[lbl] += 1

    # sort and cut classes based on frequency
    cut_classes_map = sorted(list(classes_map.items()), key=lambda x: x[1], reverse=True)[:n_classes]
    # remap classes to span (0,n_classes-1)
    remap_classes = {k: i for i, (k, _) in enumerate(cut_classes_map)}

    cut_classes = [k for k, _ in cut_classes_map]

    def filter_labels(batch):
        return [label in cut_classes for label in batch["label"]]

    cut_set = DatasetDict(
        {split: d.filter(filter_labels, batched=True, num_proc=8) for split, d in ds.items()},
    )

    def remap_labels(batch):
        batch["label"] = [remap_classes[label] for label in batch["label"]]
        return batch

    remapped_set = DatasetDict({split: d.map(remap_labels, batched=True, num_proc=8) for split, d in cut_set.items()})

    remapped_set.save_to_disk(cut_dir)


def build_yelp_reviews():
    json_path = os.path.join(local_dataset_dir(), "yelp_academic_dataset_review.json")
    dest_path = os.path.join(local_dataset_dir(), "yelp_reviews")

    if os.path.exists(dest_path):
        return

    X, y = [], []
    with open(json_path, "r") as f:
        for line in tqdm(f.readlines(), total=6990280):
            d = json.loads(line)
            X.append(d["text"])
            y.append(int(d["stars"]))

    d = Dataset.from_dict({"text": X, "label": y})

    d = d.train_test_split(train_size=725000, seed=qp.environ["_R_SEED"])["train"]
    _tmp_set = d.train_test_split(train_size=225000, seed=qp.environ["_R_SEED"])
    trainval = _tmp_set["train"]
    test = _tmp_set["test"]
    _tmp_trainval = trainval.train_test_split(train_size=200000, seed=qp.environ["_R_SEED"])
    train = _tmp_trainval["train"]
    val = _tmp_trainval["test"]

    dd = DatasetDict(
        {
            "train": train,
            "validation": val,
            "test": test,
        }
    )

    dd = DatasetDict({split: d.map(lambda x: {"text": x["text"], "label": x["label"] - 1}) for split, d in dd.items()})

    dd.save_to_disk(dest_path)


def build_rcv1(min_examples=10):
    source_dir = os.path.join(local_dataset_dir(), "rcv1_raw")
    target_dir = os.path.join(local_dataset_dir(), "rcv1-v2")
    if os.path.exists(target_dir):
        return
    os.makedirs(target_dir, exist_ok=True)

    filenames = {
        "test": [
            "lyrl2004_tokens_test_pt0.dat",
            "lyrl2004_tokens_test_pt1.dat",
            "lyrl2004_tokens_test_pt2.dat",
            "lyrl2004_tokens_test_pt3.dat",
        ],
        "train": [
            "lyrl2004_tokens_train.dat",
        ],
    }
    labels_fn = "rcv1-v2.topics.qrels"
    hier_fn = "rcv1.topics.hier.orig"

    # ---------------- data ---------------
    data_dict = {"train": {}, "test": {}}
    for split in ["train", "test"]:
        for fn in filenames[split]:
            path = os.path.join(source_dir, fn)
            with open(path, "r") as f:
                lines = f.readlines()

            did, buff = -1, []
            for line in lines:
                if line.startswith(".I"):
                    if did >= 0:
                        data_dict[split][did] = "".join(buff[:-1])
                    buff = []
                    did = int(line.strip().split(" ")[1])
                elif line.startswith(".W"):
                    continue
                else:
                    buff.append(line)

    # ---------------- labels ---------------
    labels_path = os.path.join(source_dir, labels_fn)
    with open(labels_path, "r") as f:
        lines = f.readlines()

    labels = defaultdict(list)
    for line in lines:
        label, did, _ = tuple(line.strip().split(" "))
        labels[int(did)].append(label)

    # ---------------- hierarchy ---------------
    hier_path = os.path.join(source_dir, hier_fn)
    with open(hier_path, "r") as f:
        lines = f.readlines()
    hier = defaultdict(list)
    for line in lines:
        tokens = list(map(lambda s: s.strip(), line.strip().split(":")))
        parent = tokens[1].split(" ")[0]
        child = tokens[2].split(" ")[0]
        if parent != "None":
            hier[parent].append(child)

    # ---------------- hierarchy levels ---------------
    curr_l, hier_lvl_list = 0, [["Root"]]
    while curr_l < len(hier_lvl_list):
        for parent in hier_lvl_list[curr_l]:
            if curr_l + 1 == len(hier_lvl_list):
                hier_lvl_list.append([])
            hier_lvl_list[curr_l + 1].extend(hier[parent])
        curr_l += 1
    hier_levels = {}
    for i, level in enumerate(hier_lvl_list):
        for lbl in level:
            hier_levels[lbl] = i

    # filter data selecting for each example its most specific label; if it has only one label,
    # then it is kept, otherwise it is discarded.
    raw_data = {"train": {"text": [], "label": []}, "test": {"text": [], "label": []}}
    for split, X_dict in data_dict.items():
        for did, text in X_dict.items():
            lbls = labels[did]
            max_depth = max([hier_levels[l] for l in lbls])
            f_lbls = [l for l in lbls if hier_levels[l] == max_depth]
            if len(f_lbls) == 1:
                raw_data[split]["text"].append(text)
                raw_data[split]["label"].append(f_lbls[0])

    # filter out classes that have at least 'min_examples' in train
    lbl_cnt = defaultdict(int)
    for lbl in raw_data["train"]["label"]:
        lbl_cnt[lbl] += 1
    filtered_classes = list(map(lambda x: x[0], filter(lambda x: x[1] >= min_examples, list(lbl_cnt.items()))))
    filtered_data = {"train": {"text": [], "label": []}, "test": {"text": [], "label": []}}
    for split in ["train", "test"]:
        for i in range(len(raw_data[split]["label"])):
            lbl = raw_data[split]["label"][i]
            if lbl in filtered_classes:
                filtered_data[split]["text"].append(raw_data[split]["text"][i])
                filtered_data[split]["label"].append(lbl)

    # filter test data removing examples whose classes are note in train
    unq_labels = set(filtered_data["train"]["label"])
    data = {"train": {"text": [], "label": []}, "test": {"text": [], "label": []}}
    for split in ["train", "test"]:
        for i in range(len(filtered_data[split]["label"])):
            lbl = filtered_data[split]["label"][i]
            if lbl in unq_labels:
                data[split]["text"].append(filtered_data[split]["text"][i])
                data[split]["label"].append(lbl)

    # map classes to numeric counterparts
    unq_labels = set(data["train"]["label"])
    unq_labels_dict = {lbl: i for i, lbl in enumerate(unq_labels)}
    for split in ["train", "test"]:
        data[split]["label"] = [unq_labels_dict[lbl] for lbl in data[split]["label"]]

    dd = DatasetDict(
        {
            "train": Dataset.from_dict(data["train"]),
            "test": Dataset.from_dict(data["test"]),
        }
    )

    dd.save_to_disk(target_dir)


def build_caltech_256():
    source_dir = os.path.join(local_dataset_dir(), "caltech_256_raw", "256_ObjectCategories")
    dest_dir = os.path.join(local_dataset_dir(), "caltech256")
    if os.path.exists(dest_dir):
        return
    os.makedirs(dest_dir, exist_ok=True)

    X, y = [], []
    for path in glob(os.path.join(source_dir, "*", "*.jpg")):
        X.append(path)
        y.append(int(Path(path).parent.name.split(".")[0]))

    d = (
        Dataset.from_dict({"image": X, "label": y})
        .cast_column("image", Image())
        .map(lambda x: {"image": x["image"], "label": x["label"] - 1})
    )
    print(np.unique(d["label"]))
    dd = d.train_test_split(train_size=20000)
    dd.save_to_disk(dest_dir)


def lt_power_law(dataset: Dataset, n_classes: int, imb_factor: float):
    cls_indices = defaultdict(list)
    for i, y in enumerate(dataset["label"]):
        cls_indices[y].append(i)

    n_max = len(cls_indices[0])
    num_per_cls = [int(n_max * (1 / imb_factor) ** (i / (n_classes - 1))) for i in range(n_classes)]

    selected_indices = []
    with temp_np_seed(qp.environ["_R_SEED"]):
        for c, n in zip(range(n_classes), num_per_cls):
            selected_indices.extend(np.random.choice(cls_indices[c], n, replace=False))

    return selected_indices


def build_cifar_lt(name: Literal["cifar10", "cifar100"], imb_factor: float = 5):
    set_path = os.path.join(local_dataset_dir(), f"{name}-lt")
    if os.path.exists(set_path):
        return

    dataset = datasets.load_dataset(name)

    _fields_to_remove = {
        "cifar10": ["img"],
        "cifar100": ["img", "fine_label", "coarse_label"],
    }

    def fix_fields(split):
        if name == "cifar10":
            split["image"] = split["img"]
        if name == "cifar100":
            split["image"] = split["img"]
            split["label"] = split["fine_label"]

        return split

    dataset = dataset.map(fix_fields)
    dataset = dataset.remove_columns(_fields_to_remove.get(name, []))

    cls_num = {
        "cifar10": 10,
        "cifar100": 100,
    }[name]

    selected_indices = lt_power_law(dataset["train"], cls_num, imb_factor)

    dd = DatasetDict(
        {
            "train": dataset["train"].select(selected_indices),
            "test": dataset["test"],
        }
    )

    dd.save_to_disk(set_path)


def build_food_lt(imb_factor: float = 5):
    set_path = os.path.join(local_dataset_dir(), "food101-lt")
    if os.path.exists(set_path):
        return

    dataset_id = "ethz/food101"
    dataset = datasets.load_dataset(dataset_id)

    cls_num = 101
    selected_indices = lt_power_law(dataset["train"], cls_num, imb_factor)

    dd = DatasetDict(
        {
            "train": dataset["train"].select(selected_indices),
            "test": dataset["validation"],
        }
    )

    dd.save_to_disk(set_path)


def build_agnews_lt(imb_factor=5):
    set_path = os.path.join(local_dataset_dir(), "ag_news-lt")
    if os.path.exists(set_path):
        return

    dataset_id = "fancyzhx/ag_news"
    dataset = datasets.load_dataset(dataset_id)

    cls_num = 4
    selected_indices = lt_power_law(dataset["train"], cls_num, imb_factor)

    dd = DatasetDict(
        {
            "train": dataset["train"].select(selected_indices),
            "test": dataset["test"],
        }
    )

    dd.save_to_disk(set_path)


def build_dbpedia14_lt(imb_factor=5):
    set_path = os.path.join(local_dataset_dir(), "dbpedia_14-lt")
    if os.path.exists(set_path):
        return

    dataset_id = "fancyzhx/dbpedia_14"
    dataset = datasets.load_dataset(dataset_id)

    _fields_to_remove = ["title", "content"]

    def fix_fields(split):
        split["text"] = f"{split['title']}\n{split['content']}"

        return split

    dataset = dataset.map(fix_fields)
    dataset = dataset.remove_columns(_fields_to_remove)

    cls_num = 14
    selected_indices = lt_power_law(dataset["train"], cls_num, imb_factor)

    dd = DatasetDict(
        {
            "train": dataset["train"].select(selected_indices),
            "test": dataset["test"],
        }
    )

    dd.save_to_disk(set_path)


def get_local_hf_dataset(
    name: Literal[
        # --- text ---
        "yelp_reviews",
        "rcv1-v2",
        "ag_news-lt",
        "dbpedia_14-lt"
        # --- image ---
        "imagenet-lt",
        "imagenet-lt200",
        "imagenet-lt100",
        "cifar10-lt",
        "cifar100-lt",
        "food101-lt",
        "caltech256",
    ],
):
    _split_size = {
        # --- text ---
        "rcv1-v2": 10000,
        "ag_news-lt": 38803,
        "dbpedia_14-lt": 257798,
        # --- image ---
        "imagenet-lt200": 55932,
        "imagenet-lt100": 23000,
        "cifar10-lt": 12000,
        "cifar100-lt": 12000,
        "food101-lt": 18000,
        "caltech256": 10000,
    }

    dataset_dir = local_dataset_dir()
    if name == "imagenet-lt":
        build_imagenet_lt()
    elif name == "imagenet-lt200":
        build_imagenet_lt()
        cut_imagenet_lt()
    elif name == "imagenet-lt100":
        build_imagenet_lt()
        cut_imagenet_lt(n_classes=100)
    elif name in ["cifar10-lt", "cifar100-lt"]:
        build_cifar_lt(name[:-3])
    elif name == "food101-lt":
        build_food_lt()
    elif name == "caltech256":
        download_caltech_256()
        build_caltech_256()
    elif name == "yelp_reviews":
        build_yelp_reviews()
    elif name == "rcv1-v2":
        download_rcv1_raw()
        build_rcv1()
    elif name == "dbpedia_14-lt":
        build_dbpedia14_lt()
    elif name == "ag_news-lt":
        build_agnews_lt()

    dataset = datasets.load_from_disk(os.path.join(dataset_dir, name))

    if name in _split_size:
        train_size = _split_size[name]
        _tmp_dataset = dataset["train"].train_test_split(train_size=train_size, seed=qp.environ["_R_SEED"])
        dataset["train"] = _tmp_dataset["train"]
        dataset["validation"] = _tmp_dataset["test"]

    return dataset


def get_hf_dataset(
    d_name: Literal[
        # --- text ---
        "stanfordnlp/imdb",
        "stanfordnlp/sst2",
        "fancyzhx/yelp_polarity",
        "fancyzhx/ag_news",
        "fancyzhx/dbpedia_14",
        "community-datasets/yahoo_answers_topics",
        # --- image ---
        "mnist",
        "cifar10",
        "cifar100",
        "ethz/food101",
        "ljnlonoljpiljm/caltech256",
    ],
):
    _split_size = {
        # --- text ---
        "stanfordnlp/imdb": (12500, 12500),
        "stanfordnlp/sst2": (20000, 20000),
        "fancyzhx/yelp_polarity": (60000, 25000),
        "fancyzhx/ag_news": (60000, 25000),
        "fancyzhx/dbpedia_14": (225000, 25000),
        "community-datasets/yahoo_answers_topics": (225000, 25000),
        # --- image ---
        "mnist": (35000, 25000),
        "cifar10": (25000, 25000),
        "cifar100": (25000, 25000),
        "ljnlonoljpiljm/caltech256": (10000, 10000),
        "ethz/food101": (50750, 25000),
    }
    _fields_to_remove = {
        # --- text ---
        "stanfordnlp/sst2": ["sentence"],
        "fancyzhx/dbpedia_14": ["title", "content"],
        "community-datasets/yahoo_answers_topics": ["question_title", "question_content", "best_answer", "topic"],
        # --- image ---
        "cifar10": ["img"],
        "cifar100": ["img", "fine_label", "coarse_label"],
    }

    dataset = datasets.load_dataset(d_name)

    if d_name in [
        "stanfordnlp/sst2",
        "fancyzhx/yelp_polarity",
        "fancyzhx/ag_news",
        "fancyzhx/dbpedia_14",
        "community-datasets/yahoo_answers_topics",
    ]:
        if d_name == "stanfordnlp/sst2":
            _whole_set = concatenate_datasets([dataset["train"], dataset["validation"]])
        else:
            _whole_set = concatenate_datasets([dataset["train"], dataset["test"]])

        train_size, val_size = _split_size[d_name]
        _tmp_split = _whole_set.train_test_split(train_size=train_size + val_size, seed=qp.environ["_R_SEED"])
        dataset["test"] = _tmp_split["test"]
        _tmp_trainval = _tmp_split["train"].train_test_split(train_size=train_size, seed=qp.environ["_R_SEED"])
        dataset["train"] = _tmp_trainval["train"]
        dataset["validation"] = _tmp_trainval["test"]
    elif d_name in ["stanfordnlp/imdb", "mnist", "cifar10", "cifar100", "ethz/food101"]:
        if d_name == "ethz/food101":
            dataset["test"] = dataset["validation"]

        _, val_size = _split_size[d_name]
        _tmp_dataset = dataset["train"].train_test_split(test_size=val_size, seed=qp.environ["_R_SEED"])
        dataset["train"] = _tmp_dataset["train"]
        dataset["validation"] = _tmp_dataset["test"]
        if d_name == "stanfordnlp/imdb":
            del dataset["unsupervised"]
    elif d_name == "ljnlonoljpiljm/caltech256":
        train_size, val_size = _split_size[d_name]
        _tmp_split = dataset["train"].train_test_split(train_size=train_size + val_size, seed=qp.environ["_R_SEED"])
        dataset["test"] = _tmp_split["test"]
        _tmp_trainval = _tmp_split["train"].train_test_split(train_size=train_size, seed=qp.environ["_R_SEED"])
        dataset["train"] = _tmp_trainval["train"]
        dataset["validation"] = _tmp_trainval["test"]

    def fix_fields(split):
        # --- text ---
        if d_name == "stanfordnlp/sst2":
            split["text"] = f"{split['sentence']}"
        elif d_name == "fancyzhx/dbpedia_14":
            split["text"] = f"{split['title']}\n{split['content']}"
        elif d_name == "community-datasets/yahoo_answers_topics":
            split["text"] = f"{split['question_title']}\n{split['question_content']}\n{split['best_answer']}"
            split["label"] = split["topic"]
        # --- image ---
        elif d_name == "cifar10":
            split["image"] = split["img"]
        elif d_name == "cifar100":
            split["image"] = split["img"]
            split["label"] = split["fine_label"]

        return split

    dataset = dataset.map(fix_fields)
    dataset = dataset.remove_columns(_fields_to_remove.get(d_name, []))

    return dataset


def sort_datasets_by_size(dataset_coll: str | list[str], dataset_names: list[str], descending=True):
    @functools.lru_cache(maxsize=len(UCI_BINARY_DATASETS) + len(UCI_MULTICLASS_DATASETS))
    def get_dataset_size(name, coll):
        _len = 0
        if coll == "uci_binary":
            L, V, U = fetch_UCIBinaryDataset(name)
            _len = len(L) + len(V) + len(U)
        elif coll == "uci_multiclass":
            L, V, U = fetch_UCIMulticlassDataset(name)
            _len = len(L) + len(V) + len(U)
        elif coll == "text":
            _, V, U = load_dataset("text", name)
            _len = len(V) + len(U)
        elif coll == "image":
            _, V, U = load_dataset("image", name)
            _len = len(V) + len(U)
        else:
            raise ValueError(f"Unknown dataset collection: {coll}")

        return _len

    dataset_colls = dataset_coll if isinstance(dataset_coll, list) else [dataset_coll] * len(dataset_names)

    datasets = [(d, get_dataset_size(d, c)) for d, c in zip(dataset_names, dataset_colls)]
    datasets.sort(key=(lambda d: d[1]), reverse=descending)
    return [d for (d, _) in datasets]


if __name__ == "__main__":
    qp.environ["_R_SEED"] = 0
