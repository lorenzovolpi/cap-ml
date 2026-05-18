import argparse
from typing import Iterable

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS
from quapy.method.aggregative import KDEyML
from quapy.protocol import APP
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import cap
from cap.data.datasets import fetch_UCIBinaryDataset
from cap.error import vanilla_acc
from cap.models.confidence import HD_BayesCAP
from cap.models.cont_table import O_LEAP
from cap_exp.pretrain.dataset import sort_datasets_by_size

qp.environ["_R_SEED"] = 0

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_NUM_TEST = 100


def kdey():
    return KDEyML(MLPClassifier(random_state=qp.environ["_R_SEED"]))


def method_df(num_rows: int, **data) -> pd.DataFrame:
    _data = data | {k: [v] * num_rows for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(_data, orient="columns")


def gen_datasets(
    max_datasets: int | None = None,
) -> Iterable[tuple[str, tuple[LabelledCollection, LabelledCollection, LabelledCollection]]]:
    _uci_bin_native = [
        "breast-cancer",
        "german",
        "haberman",
        "ionosphere",
        "mammographic",
        "semeion",
        "sonar",
        "spambase",
        "spectf",
        "tictactoe",
        "transfusion",
        "wdbc",
        # "yeast",
    ]
    _uci_bin_names = [d for d in UCI_BINARY_DATASETS if d in _uci_bin_native]
    coll = "uci_binary"
    _sorted_bin_names = sort_datasets_by_size(coll, _uci_bin_names, fetch_UCIBinaryDataset)
    for dn in _sorted_bin_names[:max_datasets]:
        dval = fetch_UCIBinaryDataset(dn)
        yield dn, dval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test HD_BayesCAP on binary UCI datasets.")
    parser.add_argument("--max-datasets", type=int, default=5, help="Number of binary datasets to test.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="APP sample size.")
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST, help="Number of APP prevalences per dataset.")
    parser.add_argument("--num-samples", type=int, default=100, help="Stan posterior samples for HD_BayesCAP.")
    parser.add_argument("--num-warmup", type=int, default=100, help="Stan warmup samples for HD_BayesCAP.")
    parser.add_argument("--nbins", type=int, default=4, help="Number of histogram bins for HD_BayesCAP.")
    parser.add_argument(
        "--fixed-bins",
        action="store_true",
        help="Use equally spaced bins instead of validation quantiles.",
    )
    parser.add_argument("--seed", type=int, default=qp.environ["_R_SEED"], help="Random seed.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    if args.num_test < 2:
        raise ValueError("--num-test must be at least 2 when using APP")

    qp.environ["_R_SEED"] = args.seed
    dfs = []

    for dataset_name, (L, V, U) in gen_datasets(args.max_datasets):
        h = MLPClassifier(random_state=args.seed).fit(*L.Xy)

        V_P = h.predict_proba(V.X)

        test_prot = APP(
            U,
            sample_size=args.sample_size,
            n_prevalences=10,
            repeats=args.num_test / 10,
            random_state=qp.environ["_R_SEED"],
            return_type="labelled_collection",
        )
        test_samples = list(test_prot())
        test_prot_post = [h.predict_proba(Ui.X) for Ui in test_samples]
        test_prot_true_accs = [
            accuracy_score(Ui.y, np.argmax(Ui_P, axis=1)) for Ui, Ui_P in zip(test_samples, test_prot_post)
        ]

        # leap
        test_prot_cts, test_prot_estim_accs = [], []
        leap = O_LEAP(vanilla_acc, kdey()).fit(V, V_P)
        for Ui, Ui_P in zip(test_samples, test_prot_post):
            ct = leap.predict_ct(Ui.X, Ui_P)
            test_prot_cts.append(ct)
            test_prot_estim_accs.append(cap.error.vanilla_acc(ct))
        test_prot_ae = [ae_ for ae_ in cap.error.ae(np.array(test_prot_true_accs), np.array(test_prot_estim_accs))]
        leap_df = method_df(
            len(test_samples),
            method="leap",
            dataset=dataset_name,
            ct=test_prot_cts,
            estim_acc=test_prot_estim_accs,
            true_acc=test_prot_true_accs,
            ae=test_prot_ae,
        )
        dfs.append(leap_df)

        # hd bayes
        hd_bayes = HD_BayesCAP(
            vanilla_acc,
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
            nbins=args.nbins,
            fixed_bins=args.fixed_bins,
            random_state=args.seed,
        ).fit(V, V_P)
        test_prot_cts, test_prot_estim_accs, ci_low, ci_high, ci_coverage = [], [], [], [], []
        for Ui, Ui_P, true_acc in zip(test_samples, test_prot_post, test_prot_true_accs):
            ct_samples = hd_bayes.predict_ct_range(Ui.X, Ui_P)
            ct = ct_samples.mean(axis=0)
            ci = hd_bayes.ci_from_cts(ct_samples)
            low, high = ci.interval()

            test_prot_cts.append(ct)
            test_prot_estim_accs.append(vanilla_acc(ct))
            ci_low.append(low)
            ci_high.append(high)
            ci_coverage.append(ci.coverage(true_acc))

        test_prot_ae = [ae_ for ae_ in cap.error.ae(np.array(test_prot_true_accs), np.array(test_prot_estim_accs))]
        hd_bayes_df = method_df(
            len(test_samples),
            method="hd_bayes",
            dataset=dataset_name,
            ct=test_prot_cts,
            estim_acc=test_prot_estim_accs,
            true_acc=test_prot_true_accs,
            ae=test_prot_ae,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_coverage=ci_coverage,
        )
        dfs.append(hd_bayes_df)
        print(f"{dataset_name} done.")

    df = pd.concat(dfs, axis=0)
    pivot = pd.pivot_table(df, index=["dataset"], columns=["method"], values=["ae", "ci_coverage"])
    print(pivot)


if __name__ == "__main__":
    main(parse_args())
