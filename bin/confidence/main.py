import argparse
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection
from quapy.method.aggregative import KDEyML
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import cap
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import vanilla_acc
from cap.models.confidence import CBPE, ConfidenceInterval, PrediQuant, RQBS

qp.environ["_R_SEED"] = 0

DEFAULT_BINARY_DATASETS = ["breast-cancer", "ionosphere"]
DEFAULT_MULTICLASS_DATASETS = ["digits", "wine-quality"]
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_NUM_TEST = 10
DEFAULT_NUM_SAMPLES = 50

Dataset = tuple[str, str, tuple[LabelledCollection, LabelledCollection, LabelledCollection]]
MethodFactory = Callable[[LabelledCollection, np.ndarray], object]


def parse_csv_arg(value: str) -> list[str]:
    if value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def method_df(num_rows: int, **data) -> pd.DataFrame:
    expanded = {k: [v] * num_rows for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(data | expanded, orient="columns")


def gen_datasets(binary_names: list[str], multiclass_names: list[str]) -> Iterable[Dataset]:
    for dataset_name in binary_names:
        yield dataset_name, "uci_binary", fetch_UCIBinaryDataset(dataset_name)
    for dataset_name in multiclass_names:
        yield dataset_name, "uci_multiclass", fetch_UCIMulticlassDataset(dataset_name)


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


def make_classifier(seed: int, max_iter: int) -> LogisticRegression:
    return LogisticRegression(max_iter=max_iter, random_state=seed)


def kdey_reusing_classifier(classifier: LogisticRegression, val: LabelledCollection, seed: int) -> KDEyML:
    return KDEyML(classifier=classifier, fit_classifier=False, val_split=val.Xy, random_state=seed)


def method_factories(
    method_names: list[str],
    classifier: LogisticRegression,
    val: LabelledCollection,
    sample_size: int,
    num_samples: int,
    seed: int,
) -> dict[str, MethodFactory]:
    available = {
        "cbpe-bcts": lambda data, posteriors: CBPE("vanilla_accuracy", "bcts").fit(data, posteriors),
        "cbpe-bcts-emq": lambda data, posteriors: CBPE("vanilla_accuracy", "bcts+emq").fit(data, posteriors),
        "cbpe-lascal": lambda data, posteriors: CBPE("vanilla_accuracy", "lascal").fit(data, posteriors),
        "cbpe-switch-bcts": lambda data, posteriors: CBPE(None, "bcts").switch_and_fit(
            vanilla_acc,
            data,
            posteriors,
        ),
        "cbpe-switch-bcts-emq": lambda data, posteriors: CBPE(None, "bcts+emq").switch_and_fit(
            vanilla_acc,
            data,
            posteriors,
        ),
        "cbpe-switch-lascal": lambda data, posteriors: CBPE(None, "lascal").switch_and_fit(
            vanilla_acc,
            data,
            posteriors,
        ),
        "prediquant": lambda data, posteriors: PrediQuant(
            vanilla_acc,
            kdey_reusing_classifier(classifier, val, seed),
            num_samples=num_samples,
            sample_size=sample_size,
            random_state=seed,
        ).fit(data, posteriors),
        "rqbs": lambda data, posteriors: RQBS(
            vanilla_acc,
            kdey_reusing_classifier(classifier, val, seed),
            num_samples=num_samples,
            sample_size=sample_size,
            random_state=seed,
        ).fit(data, posteriors),
    }
    unknown = sorted(set(method_names) - set(available))
    if unknown:
        raise ValueError(f"Unknown method(s): {unknown}. Available methods: {sorted(available)}")
    return {method_name: available[method_name] for method_name in method_names}


def probability_mass_in_interval(values: np.ndarray, probabilities: np.ndarray, low: float, high: float) -> float:
    in_interval = (low <= values) & (values <= high)
    return float(probabilities[in_interval].sum())


def assert_common_confidence_interface(method_name: str, method, sample: LabelledCollection, posteriors: np.ndarray):
    ci = method.predict_with_confidence(sample.X, posteriors)
    if not isinstance(ci, ConfidenceInterval):
        raise AssertionError(f"{method_name}.predict_with_confidence must return ConfidenceInterval, got {type(ci)}")

    point_from_predict = method.predict(sample.X, posteriors)
    if not np.isfinite(point_from_predict):
        raise AssertionError(f"{method_name}.predict must return a finite value, got {point_from_predict}")
    if method_name.startswith("cbpe") and not np.isclose(point_from_predict, ci.point_estimate):
        raise AssertionError(
            f"{method_name}.predict must match predict_with_confidence(...).point_estimate: "
            f"{point_from_predict=} {ci.point_estimate=}"
        )

    low, high = ci.interval()
    if not np.isfinite([ci.point_estimate, low, high]).all():
        raise AssertionError(f"{method_name} returned a non-finite confidence interval: {(low, high)}")
    if low > high:
        raise AssertionError(f"{method_name} returned an invalid confidence interval: {(low, high)}")
    if not 0 <= ci.coverage(ci.point_estimate) <= 1:
        raise AssertionError(f"{method_name}.coverage must return a value in [0, 1]")
    return ci


def assert_cbpe_confidence_interval(method: CBPE, sample: LabelledCollection, posteriors: np.ndarray):
    if method.acc_name != "vanilla_accuracy":
        raise AssertionError(f"CBPE must be configured for vanilla_accuracy in this test, got {method.acc_name}")
    if method.calib_method not in {"bcts", "bcts+emq", "lascal"}:
        raise AssertionError(f"CBPE must use a supported calib_method, got {method.calib_method}")

    acc_distribution = method.predict_range(sample.X, posteriors)
    probabilities = np.asarray(acc_distribution)
    values = np.arange(len(sample) + 1, dtype=float) / len(sample)

    if probabilities.shape != (len(sample) + 1,):
        raise AssertionError(
            "CBPE.predict_range must return the Poisson-binomial accuracy PMF as "
            f"(n + 1,), got {probabilities.shape} for n={len(sample)}"
        )
    if np.any(probabilities < -1e-12):
        raise AssertionError("CBPE probability masses must be non-negative")
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-8):
        raise AssertionError(f"CBPE probability masses must sum to 1, got {probabilities.sum()}")

    ci_90 = method.ci_from_accs(acc_distribution, confidence_level=0.90)
    ci_95 = method.ci_from_accs(acc_distribution, confidence_level=0.95)
    low_90, high_90 = ci_90.interval()
    low_95, high_95 = ci_95.interval()

    expected_accuracy = float(np.dot(values, probabilities))
    if not np.isclose(ci_95.point_estimate, expected_accuracy, atol=1e-10):
        raise AssertionError(
            "CBPE confidence point estimate must be the expectation of the Poisson-binomial distribution: "
            f"{ci_95.point_estimate=} {expected_accuracy=}"
        )
    interval_tol = 1e-8
    if not (-interval_tol <= ci_95.point_estimate <= 1 + interval_tol):
        raise AssertionError(f"CBPE point estimate must be in [0, 1], got {ci_95.point_estimate}")
    if low_95 > low_90 + interval_tol or high_95 < high_90 - interval_tol:
        raise AssertionError(
            "CBPE intervals must be monotonic with the confidence level: "
            f"90%={(low_90, high_90)} 95%={(low_95, high_95)}"
        )

    mass_95 = probability_mass_in_interval(values, probabilities, low_95, high_95)
    if mass_95 + 1e-10 < 0.95:
        raise AssertionError(f"CBPE 95% confidence interval covers only {mass_95:.6f} probability mass")


def assert_cbpe_multiclass_unsupported(method: CBPE, sample: LabelledCollection, posteriors: np.ndarray):
    if getattr(method, "calib", None) is not None or hasattr(method, "val_prev"):
        raise AssertionError("CBPE must not fit calibration state for multiclass problems")

    acc_distribution = method.predict_range(sample.X, posteriors)
    if not np.isnan(acc_distribution):
        raise AssertionError(f"CBPE.predict_range must return np.nan for multiclass problems, got {acc_distribution}")

    point_estimate = method.predict(sample.X, posteriors)
    if not np.isnan(point_estimate):
        raise AssertionError(f"CBPE.predict must return np.nan for multiclass problems, got {point_estimate}")

    ci = method.predict_with_confidence(sample.X, posteriors)
    if not isinstance(ci, ConfidenceInterval):
        raise AssertionError(f"CBPE.predict_with_confidence must return ConfidenceInterval, got {type(ci)}")
    low, high = ci.interval()
    if not np.isnan([ci.point_estimate, low, high]).all():
        raise AssertionError(
            "CBPE.predict_with_confidence must return a nan confidence interval for multiclass problems: "
            f"point={ci.point_estimate} interval={(low, high)}"
        )


def evaluate_method(
    method_name: str,
    method,
    dataset_name: str,
    dataset_kind: str,
    test_samples: list[LabelledCollection],
    test_posteriors: list[np.ndarray],
    true_accs: list[float],
) -> pd.DataFrame:
    estim_accs = []
    ci_low = []
    ci_high = []
    ci_coverage = []

    for sample, posteriors, true_acc in zip(test_samples, test_posteriors, true_accs):
        if method_name.startswith("cbpe") and sample.n_classes > 2:
            assert_cbpe_multiclass_unsupported(method, sample, posteriors)
            estim_accs.append(np.nan)
            ci_low.append(np.nan)
            ci_high.append(np.nan)
            ci_coverage.append(np.nan)
            continue

        ci = assert_common_confidence_interface(method_name, method, sample, posteriors)
        if method_name.startswith("cbpe"):
            assert_cbpe_confidence_interval(method, sample, posteriors)

        low, high = ci.interval()
        estim_accs.append(ci.point_estimate)
        ci_low.append(low)
        ci_high.append(high)
        ci_coverage.append(ci.coverage(true_acc))

    aes = cap.error.ae(np.array(true_accs), np.array(estim_accs))
    return method_df(
        len(test_samples),
        method=method_name,
        dataset=dataset_name,
        dataset_kind=dataset_kind,
        n_classes=test_samples[0].n_classes,
        estim_acc=estim_accs,
        true_acc=true_accs,
        ae=list(aes),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_width=list(np.array(ci_high) - np.array(ci_low)),
        ci_coverage=ci_coverage,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise confidence methods, with strict confidence-interval checks for CBPE."
    )
    parser.add_argument(
        "--binary-datasets",
        type=parse_csv_arg,
        default=DEFAULT_BINARY_DATASETS,
        help="Comma-separated UCI binary datasets, or an empty string to skip them.",
    )
    parser.add_argument(
        "--multiclass-datasets",
        type=parse_csv_arg,
        default=DEFAULT_MULTICLASS_DATASETS,
        help="Comma-separated UCI multiclass datasets, or an empty string to skip them.",
    )
    parser.add_argument(
        "--methods",
        type=parse_csv_arg,
        default=[
            "cbpe-bcts",
            "cbpe-bcts-emq",
            "cbpe-lascal",
            "cbpe-switch-bcts",
            "cbpe-switch-bcts-emq",
            "cbpe-switch-lascal",
            "prediquant",
            "rqbs",
        ],
        help=(
            "Comma-separated methods to test. Available: cbpe-bcts,cbpe-bcts-emq,cbpe-lascal,"
            "cbpe-switch-bcts,cbpe-switch-bcts-emq,cbpe-switch-lascal,prediquant,rqbs."
        ),
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size for each test bag.")
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST, help="Number of test bags per dataset.")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Confidence samples for sampling-based methods.",
    )
    parser.add_argument("--classifier-max-iter", type=int, default=1000, help="max_iter for LogisticRegression.")
    parser.add_argument("--seed", type=int, default=qp.environ["_R_SEED"], help="Random seed.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path where raw results are saved.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.num_test <= 0:
        raise ValueError("--num-test must be positive")

    qp.environ["_R_SEED"] = args.seed
    qp.environ["SAMPLE_SIZE"] = args.sample_size
    dfs = []

    for dataset_i, (dataset_name, dataset_kind, (L, V, U)) in enumerate(
        gen_datasets(args.binary_datasets, args.multiclass_datasets)
    ):
        classifier = make_classifier(args.seed, args.classifier_max_iter).fit(*L.Xy)
        val_posteriors = classifier.predict_proba(V.X)

        test_samples = random_prevalence_samples(
            U,
            sample_size=args.sample_size,
            repeats=args.num_test,
            random_state=args.seed + dataset_i,
        )
        test_posteriors = [classifier.predict_proba(sample.X) for sample in test_samples]
        true_accs = [
            accuracy_score(sample.y, np.argmax(posteriors, axis=1))
            for sample, posteriors in zip(test_samples, test_posteriors)
        ]

        for method_name, method_factory in method_factories(
            args.methods,
            classifier,
            V,
            args.sample_size,
            args.num_samples,
            args.seed,
        ).items():
            method = method_factory(V, val_posteriors)
            dfs.append(
                evaluate_method(
                    method_name,
                    method,
                    dataset_name,
                    dataset_kind,
                    test_samples,
                    test_posteriors,
                    true_accs,
                )
            )

        print(f"{dataset_name} done.")

    if not dfs:
        raise ValueError("No datasets selected.")

    df = pd.concat(dfs, axis=0)
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)

    pivot = pd.pivot_table(
        df,
        index=["dataset", "dataset_kind", "n_classes"],
        columns=["method"],
        values=["ae", "ci_width", "ci_coverage"],
        aggfunc="mean",
    )
    print(pivot)


if __name__ == "__main__":
    main(parse_args())
