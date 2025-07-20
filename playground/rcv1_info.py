import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from sklearn.datasets import fetch_rcv1

import cap
from cap.data.datasets import fetch_RCV1MulticlassDataset
from cap.data.util import get_rcv1_class_info

qp.environ["_R_SEED"] = 0


def target_specs(cns):
    for name in cns:
        if name not in index:
            print(f"{name} - excluded")
            continue

        T, V, U = fetch_RCV1MulticlassDataset(name)
        print(
            f"{name} - Tsize: {len(T)}; Tprev: {T.prevalence()}; Usize: {len(U)}; T/Uclasses: {T.n_classes}/{U.n_classes}"
        )


if __name__ == "__main__":
    ext_cns, tree, index = get_rcv1_class_info()
    training = fetch_rcv1(subset="train", data_home=cap.env["SKLEARN_DATA"])
    orig_labels = training.target.toarray()
    print(orig_labels.shape)
    orig_cns = training.target_names
    ext_cns = np.asarray(ext_cns)

    print(ext_cns)
    print(training.target.shape)
    print(len(ext_cns))
    print(tree)
    print(index)

    target_specs(ext_cns)
