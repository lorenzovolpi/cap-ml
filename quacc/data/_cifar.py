import os
import pickle
import tarfile

import numpy as np
from sklearn.utils import Bunch

import quacc.utils.commons as commons


def fetch_cifar10() -> Bunch:
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_home = commons.get_quacc_home()
    unzipped_path = data_home / "cifar-10-batches-py"
    if not unzipped_path.exists():
        downloaded_path = data_home / URL.split("/")[-1]
        commons.download_file(URL, downloaded_path)
        with tarfile.open(downloaded_path) as f:
            f.extractall(data_home)
        os.remove(downloaded_path)

    datas = []
    data_names = sorted([f for f in os.listdir(unzipped_path) if f.startswith("data")])
    for f in data_names:
        with open(unzipped_path / f, "rb") as file:
            datas.append(pickle.load(file, encoding="bytes"))

    tests = []
    test_names = sorted([f for f in os.listdir(unzipped_path) if f.startswith("test")])
    for f in test_names:
        with open(unzipped_path / f, "rb") as file:
            tests.append(pickle.load(file, encoding="bytes"))

    with open(unzipped_path / "batches.meta", "rb") as file:
        meta = pickle.load(file, encoding="bytes")

    return Bunch(
        train=Bunch(
            data=np.concatenate([d[b"data"] for d in datas], axis=0),
            labels=np.concatenate([d[b"labels"] for d in datas]),
        ),
        test=Bunch(
            data=np.concatenate([d[b"data"] for d in tests], axis=0),
            labels=np.concatenate([d[b"labels"] for d in tests]),
        ),
        label_names=[cs.decode("utf-8") for cs in meta[b"label_names"]],
    )


def fetch_cifar100():
    URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    data_home = commons.get_quacc_home()
    unzipped_path = data_home / "cifar-100-python"
    if not unzipped_path.exists():
        downloaded_path = data_home / URL.split("/")[-1]
        commons.download_file(URL, downloaded_path)
        with tarfile.open(downloaded_path) as f:
            f.extractall(data_home)
        os.remove(downloaded_path)

    with open(unzipped_path / "train", "rb") as file:
        train_d = pickle.load(file, encoding="bytes")

    with open(unzipped_path / "test", "rb") as file:
        test_d = pickle.load(file, encoding="bytes")

    with open(unzipped_path / "meta", "rb") as file:
        meta_d = pickle.load(file, encoding="bytes")

    train_bunch = Bunch(
        data=train_d[b"data"],
        fine_labels=np.array(train_d[b"fine_labels"]),
        coarse_labels=np.array(train_d[b"coarse_labels"]),
    )

    test_bunch = Bunch(
        data=test_d[b"data"],
        fine_labels=np.array(test_d[b"fine_labels"]),
        coarse_labels=np.array(test_d[b"coarse_labels"]),
    )

    return Bunch(
        train=train_bunch,
        test=test_bunch,
        fine_label_names=meta_d[b"fine_label_names"],
        coarse_label_names=meta_d[b"coarse_label_names"],
    )
