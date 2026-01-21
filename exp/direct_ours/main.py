import os
from typing import Tuple

import numpy as np
import pandas as pd

import cap
import exp.direct_ours.env as env
from exp.direct_ours.data import Clsf, DatasetBundle
from exp.direct_ours.generators import gen_acc_measures, gen_classifiers, gen_datasets, gen_methods
from exp.direct_ours.util import gen_method_df, get_exp_name, get_plain_prev


def train_predict(clsf_name, h, dataset_name, L, V, U):
    cl = Clsf(clsf_name, h).fit(*L.Xy)
    D = DatasetBundle(dataset_name, L, V, U).create_sets().get_posteriors(cl.h).get_true_accs(list(gen_acc_measures()))

    return cl, D


if __name__ == "__main__":
    os.makedirs(env.BASEDIR, exist_ok=True)
    cld: list[Tuple[Clsf, DatasetBundle]] = []
    for dataset_name, (L, V, U) in gen_datasets():
        for clsf_name, h in gen_classifiers():
            cld.append(train_predict(clsf_name, h, dataset_name, L, V, U))
            print(f"{clsf_name}@{dataset_name} trained")

    dfs = []
    for cl, D in cld:
        for acc_name, acc_fn in gen_acc_measures():
            for method_name, method, val, val_posteriors in gen_methods(acc_fn, cl, D):
                exp_name = get_exp_name(method_name, cl.name, D.name, acc_name, env.PROBLEM)
                exp_path = os.path.join(env.BASEDIR, f"{exp_name}.json")

                if not os.path.exists(exp_path):
                    val_prev = get_plain_prev(val.prevalence())
                    method.fit(val, val_posteriors)
                    estim_accs = method.batch_predict(D.test_prot, D.test_prot_posteriors)
                    true_accs = D.true_accs[acc_name]
                    aes = cap.error.ae(np.array(true_accs), np.array(estim_accs)).tolist()

                    df_len = D.test_prot.total()
                    method_df = gen_method_df(
                        df_len,
                        estim_accs=estim_accs,
                        true_accs=D.true_accs[acc_name],
                        aes=aes,
                        classifier=cl.name,
                        method=method_name,
                        dataset=D.name,
                        acc_name=acc_name,
                        train_prev=[D.L_prevalence] * df_len,
                        val_prev=[val_prev] * df_len,
                    )
                    method_df.to_json(exp_path)

                method_df = pd.read_json(exp_path)
                dfs.append(method_df)
                print(f"{method_name}@[{cl.name}-{D.name}-{acc_name}] done.")

    df = pd.concat(dfs, ignore_index=True, axis=0)

    df = pd.read_json(env.JSON_PATH)
    for acc_name, _ in gen_acc_measures():
        for c_name, h in gen_classifiers():
            sub_df = df[(df["classifier"] == c_name) & (df["acc_name"] == acc_name)]
            pivot = pd.pivot_table(
                sub_df,
                index=["dataset"],
                columns=["method"],
                values=["aes"],
            )
            pivot.loc["mean"] = pivot.mean(numeric_only=True, axis=0)
            print(f"\n{c_name} - {acc_name}\n", pivot)
