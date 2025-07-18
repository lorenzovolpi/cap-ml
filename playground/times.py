import os

import quacc as qc
from quacc.experiments.generators import gen_classifiers
from quacc.experiments.report import Report

PROBLEM = "binary"

if __name__ == "__main__":
    root_folder = os.path.join(qc.env["OUT_DIR"], "results")
    dir_path = "playground/times"
    os.makedirs(dir_path, exist_ok=True)

    classifiers = [cls_name for cls_name, _ in gen_classifiers()]
    methods = ["ATC-MC", "DoC", "N2E(ACC-h0)", "N2E(KDEy-h0)", "Naive"]
    for cls_name in classifiers:
        rep = Report.load_results(root_folder, PROBLEM, cls_name, "vanilla_accuracy", methods=methods)
        df = rep.time_data()
        df = df.pivot_table(index="dataset", columns="method", values=["t_train", "t_test_ave"])
        df.loc["max", :] = df.max(axis=0)
        df.loc["avg", :] = df.mean(axis=0)
        with open(os.path.join(dir_path, f"{cls_name}.txt"), "w") as f:
            f.write(repr(df))
        print(cls_name, df, end="\n\n")
