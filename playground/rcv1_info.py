import numpy as np
import quapy as qp
from quapy.data.base import LabelledCollection
from sklearn.datasets import fetch_rcv1

import quacc as qc
from quacc.data.datasets import fetch_RCV1MulticlassDataset
from quacc.data.util import get_rcv1_class_info

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
    training = fetch_rcv1(subset="train", data_home=qc.env["SKLEARN_DATA"])
    orig_labels = training.target.toarray()
    print(orig_labels.shape)
    orig_cns = training.target_names
    ext_cns = np.asarray(ext_cns)

    # sorted_ext_idx = np.argsort(ext_cns)
    # sorted_ext = ext_cns[sorted_ext_idx]
    # subset_idx = np.searchsorted(sorted_ext, orig_cns)
    # ext_labels = np.zeros((orig_labels.shape[0], ext_cns.shape[0]))
    # print(ext_labels.shape)
    # ext_labels[:, subset_idx] = orig_labels
    #
    # for name in ext_cns:
    #     if name not in orig_cns:
    #         ext_idx = np.where(ext_cns == name)[0][0]
    #         new_lbl = np.sum(ext_labels[:, index[name]], axis=-1)
    #         new_lbl[np.where(new_lbl > 0)[0]] = 1.0
    #         ext_labels[:, ext_idx] = new_lbl
    # print(ext_labels)

    print(ext_cns)
    print(training.target.shape)
    print(len(ext_cns))
    print(tree)
    print(index)

    target_specs(ext_cns)
