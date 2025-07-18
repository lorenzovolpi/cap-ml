from quapy.data.datasets import UCI_BINARY_DATASETS
from sklearn.linear_model import LogisticRegression

from quacc.data.datasets import fetch_UCIBinaryDataset
from quacc.error import vanilla_acc
from quacc.models.direct import KFCV

if __name__ == "__main__":
    for d in UCI_BINARY_DATASETS:
        L, V, U = fetch_UCIBinaryDataset(d)
        h = LogisticRegression()
        h.fit(*L.Xy)

        m = KFCV(h, vanilla_acc)
        m.fit(V)

        print(f"{d}: {m.cv_score}")
