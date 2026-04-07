from collections import defaultdict

import numpy as np
import pandas as pd
from data import PretrainInfo, load_from_collection, load_info_paths
from sklearn.metrics import accuracy_score

DOMAIN = "text"


def main():
    paths = load_info_paths(DOMAIN)
    accs = defaultdict(list)
    for info in paths:
        print(info)
        p = PretrainInfo.load(info, fast=True)
        _, V, U = load_from_collection(p)
        _npz = np.load(p.posteriors_path)
        V_posteriors = _npz["V_posteriors"]
        U_posteriors = _npz["U_posteriors"]
        V_yhat = np.argmax(V_posteriors, axis=-1)
        U_yhat = np.argmax(U_posteriors, axis=-1)
        accs["V"].append(accuracy_score(V_yhat, V.y))
        accs["U"].append(accuracy_score(U_yhat, U.y))
        accs["dataset"].append(p.d_info.name)
        accs["model"].append(p.h_info.name)

    df = pd.DataFrame.from_dict(accs)
    pivot = pd.pivot_table(df, index=["dataset"], columns=["model"], values=["V", "U"])
    pivot.to_json("show_acc.json")


if __name__ == "__main__":
    main()
