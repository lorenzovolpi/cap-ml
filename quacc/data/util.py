import json
import os
import urllib.request
from collections import defaultdict

import datasets
import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from torch.utils.data import DataLoader

import quacc as qc
from quacc.data.base import TorchLabelledCollection

# fmt: off

# original
# RCV1_HIERARCHY_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a02-orig-topics-hierarchy/rcv1.topics.hier.orig"

# extended
RCV1_HIERARCHY_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a03-expanded-topics-hierarchy/rcv1.topics.hier.expanded"

# fmt: on


def split_train(train: LabelledCollection, train_val_split: float):
    return train.split_stratified(train_prop=train_val_split, random_state=qp.environ["_R_SEED"])


def get_rcv1_class_info():
    os.makedirs(qc.env["QUACC_DATA"], exist_ok=True)
    json_path = os.path.join(qc.env["QUACC_DATA"], "rcv1_class_info.json")
    if not os.path.exists(json_path):
        # retrieve hierarchy file and class names
        hierarchy_tmp_path = os.path.join(qc.env["QUACC_DATA"], "rcv1_hierarchy.tmp")
        urllib.request.urlretrieve(RCV1_HIERARCHY_URL, filename=hierarchy_tmp_path)
        tree = defaultdict(lambda: [])
        class_names = set()
        with open(hierarchy_tmp_path, "r") as tf:
            lines = tf.readlines()
        for line in lines:
            tokens = [s for s in line.strip().split(" ") if len(s) > 0]
            parent, child = tokens[1], tokens[3]
            if parent != "None":
                tree[parent].append(child)
                class_names.add(child)
            class_names.add("Root")

        # sort class names
        class_names = sorted(list(class_names))

        with open(json_path, "w") as jf:
            json.dump(
                {
                    "class_names": class_names,
                    "tree": tree,
                },
                jf,
                indent=2,
            )

        if os.path.exists(hierarchy_tmp_path):
            os.remove(hierarchy_tmp_path)

    with open(json_path, "r") as jf:
        class_info = json.load(jf)

    class_names = np.array(class_info["class_names"])
    tree, index = {}, {}
    for parent, children in class_info["tree"].items():
        children = np.array(children)
        idxs = np.where(np.in1d(class_names, children))[0]
        if len(idxs) == len(children):
            tree[parent] = children
            index[parent] = idxs

    return class_names.tolist(), tree, index


hf_dataset_map = {
    "imdb": (["text"], 25000),
    "rotten_tomatoes": (["text"], 8530),
    "amazon_polarity": (["title", "content"], 25000),
}


def preprocess_hf_dataset(
    dataset: datasets.Dataset, set_name, tokenizer, collator, text_columns: [str], length: int | None = None
) -> TorchLabelledCollection:
    def tokenize(datapoint):
        sentences = [datapoint[c] for c in text_columns]
        return tokenizer(*sentences, truncation=True)

    d = dataset[set_name].shuffle(seed=qp.environ["_R_SEED"])
    if length is not None:
        d = d.select(np.arange(length))
    d = d.map(tokenize, batched=True)

    d = d.remove_columns(text_columns)
    ds = next(iter(DataLoader(d, collate_fn=collator, batch_size=len(d))))
    return TorchLabelledCollection(instances=ds["input_ids"], labels=ds["labels"], attention_mask=ds["attention_mask"])
