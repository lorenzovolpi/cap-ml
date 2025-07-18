import numpy as np
import quapy as qp
from quapy.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from quapy.functional import uniform_prevalence_sampling, uniform_simplex_sampling
from quapy.protocol import APP, UPP
from sklearn.datasets import fetch_rcv1

import quacc as qc
from quacc.data.datasets import fetch_RCV1MulticlassDataset
from quacc.data.util import get_rcv1_class_info
from quacc.experiments.sampling import get_uniform_prevalences

qp.environ["_R_SEED"] = 0


if __name__ == "__main__":
    L, U = fetch_UCIMulticlassDataset("digits").train_test

    for _ in range(10):
        print(get_uniform_prevalences(L.n_classes, 9))

    # for cn in class_names:
    #     try:
    #         L, _, _ = fetch_RCV1MulticlassDataset(cn)
    #         print(f"{cn} : {L.counts()} - {len(L)}")
    #     except KeyError:
    #         pass

    # class_names, tree, index = get_rcv1_class_info()
    #
    # training = fetch_rcv1(subset="train", data_home=qc.env["SKLEARN_DATA"])
    # t_labels = extend_labels(training.target.toarray(), training.target_names, class_names)
    # print(t_labels[:, 0].sum())
    # print((t_labels[:, [0, 19, 24, 33]].sum(axis=1) > 1).sum())
