import os

os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
from collections.abc import Iterable

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_MULTICLASS_DATASETS
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import cap
from cap.data.datasets import fetch_UCIMulticlassDataset
from cap.error import vanilla_acc
from cap.models.confidence import ACC_BayesCAP

qp.environ["_R_SEED"] = 0

DEFAULT_DATASETS = ["connect-4"]
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_NUM_TEST = 5
DEFAULT_NUM_SAMPLES = 50
DEFAULT_NUM_WARMUP = 50


Dataset = tuple[str, tuple[LabelledCollection, LabelledCollection, LabelledCollection]]


def parse_csv_arg(value: str) -> list[str]:
    if value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def method_df(num_rows: int, **data) -> pd.DataFrame:
    _data = data | {k: [v] * num_rows for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(_data, orient="columns")


def gen_datasets(dataset_names: list[str]) -> Iterable[Dataset]:
    unknown = sorted(set(dataset_names) - set(UCI_MULTICLASS_DATASETS))
    if unknown:
        raise ValueError(f"Unknown UCI multiclass dataset(s): {unknown}")
    for dataset_name in dataset_names:
        yield dataset_name, fetch_UCIMulticlassDataset(dataset_name)


def random_prevalence_samples(
    data: LabelledCollection,
    sample_size: int,
    repeats: int,
    random_state: int,
) -> list[LabelledCollection]:
    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(repeats):
        prevs = rng.dirichlet(np.ones(data.n_classes))
        sample_index = data.sampling_index(
            sample_size,
            *prevs,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        samples.append(data.sampling_from_index(sample_index))
    return samples


def prior_cases(n_classes: int) -> list[tuple[str, str | np.ndarray]]:
    return [
        ("uniform", "uniform"),
        ("vector", np.linspace(0.5, 2.0, n_classes, dtype=float)),
    ]


def assert_acc_bayes_result(
    method_name: str,
    method: ACC_BayesCAP,
    sample: LabelledCollection,
    posteriors: np.ndarray,
    true_acc: float,
) -> tuple[np.ndarray, float, float, float, float]:
    cts = method.predict_ct_range(sample.X, posteriors)
    expected_shape = (method.num_samples, sample.n_classes, sample.n_classes)
    if cts.shape != expected_shape:
        raise AssertionError(f"{method_name}.predict_ct_range returned {cts.shape}, expected {expected_shape}")
    if not np.isfinite(cts).all():
        raise AssertionError(f"{method_name}.predict_ct_range returned non-finite values")
    if np.any(cts < -1e-8):
        raise AssertionError(f"{method_name}.predict_ct_range returned negative contingency-table values")
    if not np.allclose(cts.sum(axis=(1, 2)), 1.0, atol=1e-5):
        raise AssertionError(f"{method_name}.predict_ct_range samples must sum to 1")

    ct = cts.mean(axis=0)
    estim_acc = vanilla_acc(ct)
    if not np.isfinite(estim_acc) or not 0 <= estim_acc <= 1:
        raise AssertionError(f"{method_name} returned an invalid accuracy estimate: {estim_acc}")

    ci = method.ci_from_cts(cts)
    low, high = ci.interval()
    if not np.isfinite([ci.point_estimate, low, high]).all():
        raise AssertionError(f"{method_name} returned a non-finite confidence interval: {(low, high)}")
    if low > high:
        raise AssertionError(f"{method_name} returned an invalid confidence interval: {(low, high)}")

    return ct, estim_acc, low, high, ci.coverage(true_acc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test ACC_BayesCAP priors on UCI multiclass datasets.")
    parser.add_argument(
        "--datasets",
        type=parse_csv_arg,
        default=DEFAULT_DATASETS,
        help="Comma-separated UCI multiclass datasets to test.",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size for each test bag.")
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST, help="Number of test bags per dataset.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="MCMC posterior samples.")
    parser.add_argument("--num-warmup", type=int, default=DEFAULT_NUM_WARMUP, help="MCMC warmup samples.")
    parser.add_argument("--classifier-max-iter", type=int, default=1000, help="max_iter for LogisticRegression.")
    parser.add_argument("--seed", type=int, default=qp.environ["_R_SEED"], help="Random seed.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.num_test <= 0:
        raise ValueError("--num-test must be positive")
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.num_warmup < 0:
        raise ValueError("--num-warmup must be non-negative")

    qp.environ["_R_SEED"] = args.seed
    dfs = []

    for dataset_i, (dataset_name, (L, V, U)) in enumerate(gen_datasets(args.datasets)):
        classifier = LogisticRegression(max_iter=args.classifier_max_iter, random_state=args.seed).fit(*L.Xy)
        V_P = classifier.predict_proba(V.X)

        test_samples = random_prevalence_samples(
            U,
            sample_size=args.sample_size,
            repeats=args.num_test,
            random_state=args.seed + dataset_i,
        )
        test_prot_post = [classifier.predict_proba(Ui.X) for Ui in test_samples]
        test_prot_true_accs = [
            accuracy_score(Ui.y, np.argmax(Ui_P, axis=1)) for Ui, Ui_P in zip(test_samples, test_prot_post)
        ]

        for prior_name, prior in prior_cases(V.n_classes):
            method_name = f"acc_bayes_{prior_name}"
            acc_bayes = ACC_BayesCAP(
                vanilla_acc,
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
                prior=prior,
                random_state=args.seed,
            ).fit(V, V_P)

            test_prot_cts, test_prot_estim_accs, ci_low, ci_high, ci_coverage = [], [], [], [], []
            for Ui, Ui_P, true_acc in zip(test_samples, test_prot_post, test_prot_true_accs):
                ct, estim_acc, low, high, coverage = assert_acc_bayes_result(
                    method_name,
                    acc_bayes,
                    Ui,
                    Ui_P,
                    true_acc,
                )
                test_prot_cts.append(ct)
                test_prot_estim_accs.append(estim_acc)
                ci_low.append(low)
                ci_high.append(high)
                ci_coverage.append(coverage)

            test_prot_ae = cap.error.ae(np.array(test_prot_true_accs), np.array(test_prot_estim_accs))
            dfs.append(
                method_df(
                    len(test_samples),
                    method=method_name,
                    dataset=dataset_name,
                    n_classes=V.n_classes,
                    prior=prior_name,
                    ct=test_prot_cts,
                    estim_acc=test_prot_estim_accs,
                    true_acc=test_prot_true_accs,
                    ae=list(test_prot_ae),
                    ci_low=ci_low,
                    ci_high=ci_high,
                    ci_coverage=ci_coverage,
                )
            )
        print(f"{dataset_name} done.")

    if not dfs:
        raise ValueError("No datasets selected.")

    df = pd.concat(dfs, axis=0)
    pivot = pd.pivot_table(
        df,
        index=["dataset", "n_classes"],
        columns=["method"],
        values=["ae", "ci_coverage"],
        aggfunc="mean",
    )
    print(pivot)


if __name__ == "__main__":
    main(parse_args())
