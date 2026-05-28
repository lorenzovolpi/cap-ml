import argparse
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.protocol import UPP
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from cap.calib import BCTS
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset, sort_datasets_by_size

EPSILON = 1e-4
MAX_ITER = 1000
DEFAULT_CLASSIFIERS = ("lr", "mlp", "svm")
DEFAULT_NUM_TEST = 100
DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_MAX_BINARY = 5
DEFAULT_MAX_MULTICLASS = 10
NATIVE_UCI_BINARY_DATASETS = [
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
]


def ae(prevs_true, prevs_hat):
    """Computes the absolute error between the two prevalence vectors.
     Absolute error between two prevalence vectors :math:`p` and :math:`\\hat{p}`  is computed as
     :math:`AE(p,\\hat{p})=\\frac{1}{|\\mathcal{Y}|}\\sum_{y\\in \\mathcal{Y}}|\\hat{p}(y)-p(y)|`,
     where :math:`\\mathcal{Y}` are the classes of interest.

    :param prevs_true: array-like of shape `(n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_classes,)` with the predicted prevalence values
    :return: absolute error
    """
    prevs_true = np.asarray(prevs_true)
    prevs_hat = np.asarray(prevs_hat)
    assert prevs_true.shape == prevs_hat.shape, f"wrong shape {prevs_true.shape} vs. {prevs_hat.shape}"
    return abs(prevs_hat - prevs_true).mean(axis=-1)


def mae(prevs_true, prevs_hat):
    """Computes the mean absolute error (see :meth:`quapy.error.ae`) across the sample pairs.

    :param prevs_true: array-like of shape `(n_samples, n_classes,)` with the true prevalence values
    :param prevs_hat: array-like of shape `(n_samples, n_classes,)` with the predicted
        prevalence values
    :return: mean absolute error
    """
    return ae(prevs_true, prevs_hat).mean()


def EM(tr_prev, posterior_probabilities, epsilon=EPSILON):
    """
    Computes the `Expectation Maximization` routine.

    :param tr_prev: array-like, the training prevalence
    :param posterior_probabilities: `np.ndarray` of shape `(n_instances, n_classes,)` with the
        posterior probabilities
    :param epsilon: float, the threshold different between two consecutive iterations
        to reach before stopping the loop
    :return: a tuple with the estimated prevalence values (shape `(n_classes,)`) and
        the corrected posterior probabilities (shape `(n_instances, n_classes,)`)
    """
    Px = posterior_probabilities
    Ptr = np.copy(tr_prev)

    if np.prod(Ptr) == 0:  # some entry is 0; we should smooth the values to avoid 0 division
        Ptr += epsilon
        Ptr /= Ptr.sum()

    qs = np.copy(Ptr)  # qs (the running estimate) is initialized as the training prevalence

    s, converged = 0, False
    qs_prev_ = None
    while not converged and s < MAX_ITER:
        # E-step: ps is Ps(y|xi)
        ps_unnormalized = (qs / Ptr) * Px
        ps = ps_unnormalized / ps_unnormalized.sum(axis=1, keepdims=True)

        # M-step:
        qs = ps.mean(axis=0)

        if qs_prev_ is not None and mae(qs, qs_prev_) < epsilon and s > 10:
            converged = True

        qs_prev_ = qs
        s += 1

    if not converged:
        print("[warning] the method has reached the maximum number of iterations; it might have not converged")

    return qs, ps


DatasetFactory = Callable[[], tuple[LabelledCollection, LabelledCollection, LabelledCollection]]
ClassifierFactory = Callable[[int, int], ClassifierMixin]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    factory: DatasetFactory


@dataclass(frozen=True)
class ExperimentResult:
    dataset: str
    dataset_kind: str
    classifier: str
    total_samples: int
    successful_samples: int
    failures: Counter
    prevalence_ae: list[float]
    invalid_outputs: int


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[np.asarray(y, dtype=int)]


def parse_names(raw_names: str | None) -> list[str]:
    if raw_names is None:
        return []
    names = [name.strip() for name in raw_names.split(",")]
    return [name for name in names if name]


def limit_names(names: list[str], max_datasets: int | None) -> list[str]:
    if max_datasets is None or max_datasets < 0:
        return names
    return names[:max_datasets]


def binary_dataset_spec(dataset_name: str) -> DatasetSpec:
    return DatasetSpec(
        name=dataset_name,
        kind="uci_binary",
        factory=lambda name=dataset_name: fetch_UCIBinaryDataset(name),
    )


def multiclass_dataset_spec(dataset_name: str) -> DatasetSpec:
    return DatasetSpec(
        name=dataset_name,
        kind="uci_multiclass",
        factory=lambda name=dataset_name: fetch_UCIMulticlassDataset(name),
    )


def top_binary_dataset_specs(max_datasets: int | None) -> list[DatasetSpec]:
    names = [name for name in UCI_BINARY_DATASETS if name in NATIVE_UCI_BINARY_DATASETS]
    names = sort_datasets_by_size(names, fetch_UCIBinaryDataset, descending=True)
    return [binary_dataset_spec(name) for name in limit_names(names, max_datasets)]


def top_multiclass_dataset_specs(max_datasets: int | None) -> list[DatasetSpec]:
    names = sort_datasets_by_size(list(UCI_MULTICLASS_DATASETS), fetch_UCIMulticlassDataset, descending=True)
    return [multiclass_dataset_spec(name) for name in limit_names(names, max_datasets)]


def parse_dataset_spec(raw_name: str) -> DatasetSpec:
    if ":" in raw_name:
        kind, dataset_name = raw_name.split(":", maxsplit=1)
    else:
        kind, dataset_name = "uci_binary", raw_name

    if kind == "binary":
        kind = "uci_binary"
    elif kind == "multiclass":
        kind = "uci_multiclass"

    if kind == "uci_binary":
        if dataset_name not in UCI_BINARY_DATASETS:
            raise ValueError(f"Unknown UCI binary dataset: {dataset_name}")
        return binary_dataset_spec(dataset_name)
    if kind == "uci_multiclass":
        if dataset_name not in UCI_MULTICLASS_DATASETS:
            raise ValueError(f"Unknown UCI multiclass dataset: {dataset_name}")
        return multiclass_dataset_spec(dataset_name)

    raise ValueError(f"Unknown dataset kind: {kind}. Use uci_binary:name or uci_multiclass:name")


def dataset_specs(args: argparse.Namespace) -> list[DatasetSpec]:
    if args.datasets:
        return [parse_dataset_spec(name) for name in parse_names(args.datasets)]

    specs = []
    specs.extend(top_binary_dataset_specs(args.max_binary))
    specs.extend(top_multiclass_dataset_specs(args.max_multiclass))
    return specs


def classifier_factories() -> dict[str, ClassifierFactory]:
    return {
        "lr": lambda seed, max_iter: LogisticRegression(random_state=seed, max_iter=max_iter),
        "mlp": lambda seed, max_iter: MLPClassifier(random_state=seed),
        "svm": lambda seed, max_iter: SVC(random_state=seed, probability=True),
    }


def validate_requested_names(requested: list[str], available: dict[str, object], kind: str):
    unknown = sorted(set(requested) - set(available))
    if unknown:
        valid = ", ".join(sorted(available))
        raise ValueError(f"Unknown {kind}: {', '.join(unknown)}. Available {kind}: {valid}")


def nonfinite_count(*arrays: np.ndarray) -> int:
    return sum(not np.all(np.isfinite(array)) for array in arrays)


def run_experiment(
    dataset: DatasetSpec,
    classifier_name: str,
    classifier_factory: ClassifierFactory,
    sample_size: int,
    num_test: int,
    seed: int,
    lr_max_iter: int,
) -> ExperimentResult:
    L, V, U = dataset.factory()
    classifier = classifier_factory(seed, lr_max_iter)
    classifier.fit(*L.Xy)

    V_posteriors = classifier.predict_proba(V.X)
    V_labels = one_hot(V.y, V.n_classes)
    failures = Counter()
    prevalence_ae = []
    invalid_outputs = 0

    try:
        calibrate = BCTS()(V_posteriors, V_labels, posterior_supplied=True)
    except Exception as exc:
        failures["fit_bcts"] += 1
        print(
            f"[failure] dataset={dataset.name} dataset_kind={dataset.kind} "
            f"classifier={classifier_name} stage=fit_bcts error={exc!r}"
        )
        return ExperimentResult(
            dataset=dataset.name,
            dataset_kind=dataset.kind,
            classifier=classifier_name,
            total_samples=num_test,
            successful_samples=0,
            failures=failures,
            prevalence_ae=prevalence_ae,
            invalid_outputs=invalid_outputs,
        )

    test_protocol = UPP(
        U,
        sample_size=sample_size,
        repeats=num_test,
        random_state=seed,
        return_type="labelled_collection",
    )

    for sample_id, Ui in enumerate(test_protocol()):
        Ui_posteriors = classifier.predict_proba(Ui.X)

        try:
            Ui_posteriors_calibrated = calibrate(Ui_posteriors)
        except Exception as exc:
            failures["apply_bcts"] += 1
            print(
                f"[failure] dataset={dataset.name} dataset_kind={dataset.kind} classifier={classifier_name} "
                f"sample={sample_id} stage=apply_bcts error={exc!r}"
            )
            continue

        try:
            estimated_prevalence, estimated_posteriors = EM(V.prevalence(), Ui_posteriors_calibrated)
        except Exception as exc:
            failures["em"] += 1
            print(
                f"[failure] dataset={dataset.name} dataset_kind={dataset.kind} classifier={classifier_name} "
                f"sample={sample_id} stage=em error={exc!r}"
            )
            continue

        invalid_outputs += nonfinite_count(Ui_posteriors_calibrated, estimated_prevalence, estimated_posteriors)
        prevalence_ae.append(ae(Ui.prevalence(), estimated_prevalence))

    return ExperimentResult(
        dataset=dataset.name,
        dataset_kind=dataset.kind,
        classifier=classifier_name,
        total_samples=num_test,
        successful_samples=len(prevalence_ae),
        failures=failures,
        prevalence_ae=prevalence_ae,
        invalid_outputs=invalid_outputs,
    )


def print_result(result: ExperimentResult):
    print(f"\n[{result.dataset_kind}:{result.dataset} / {result.classifier}]")
    print(f"samples: {result.successful_samples}/{result.total_samples} successful")
    print(
        "failures: "
        f"fit_bcts={result.failures['fit_bcts']} "
        f"apply_bcts={result.failures['apply_bcts']} "
        f"em={result.failures['em']}"
    )
    print(f"invalid_outputs_not_counted_as_failures: {result.invalid_outputs}")

    if result.prevalence_ae:
        prev_ae = np.asarray(result.prevalence_ae)
        print(
            "prevalence_ae: "
            f"mean={prev_ae.mean():.6f} "
            f"std={prev_ae.std():.6f} "
            f"min={prev_ae.min():.6f} "
            f"max={prev_ae.max():.6f}"
        )
    else:
        print("prevalence_ae: unavailable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose BCTS calibration failures under label shift.")
    parser.add_argument(
        "--datasets",
        default=None,
        help=(
            "Comma-separated datasets. Use bare names for UCI binary, or prefixes "
            "uci_binary:name / uci_multiclass:name."
        ),
    )
    parser.add_argument(
        "--max-binary",
        type=int,
        default=DEFAULT_MAX_BINARY,
        help="Number of largest UCI binary datasets to test when --datasets is omitted; -1 for all.",
    )
    parser.add_argument(
        "--max-multiclass",
        type=int,
        default=DEFAULT_MAX_MULTICLASS,
        help="Number of largest UCI multiclass datasets to test when --datasets is omitted; -1 for all.",
    )
    parser.add_argument(
        "--classifiers",
        default=",".join(DEFAULT_CLASSIFIERS),
        help="Comma-separated classifier names.",
    )
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST, help="Number of UPP samples from U.")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Size of each UPP sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--lr-max-iter", type=int, default=1000, help="max_iter for LogisticRegression.")
    return parser.parse_args()


def main():
    args = parse_args()
    qp.environ["_R_SEED"] = args.seed

    requested_classifiers = parse_names(args.classifiers)
    classifiers = classifier_factories()
    validate_requested_names(requested_classifiers, classifiers, "classifiers")
    datasets = dataset_specs(args)

    results = []
    for dataset in datasets:
        for classifier_name in requested_classifiers:
            print(f"\nRunning dataset={dataset.name} dataset_kind={dataset.kind} classifier={classifier_name}")
            result = run_experiment(
                dataset=dataset,
                classifier_name=classifier_name,
                classifier_factory=classifiers[classifier_name],
                sample_size=args.sample_size,
                num_test=args.num_test,
                seed=args.seed,
                lr_max_iter=args.lr_max_iter,
            )
            print_result(result)
            results.append(result)

    total_failures = Counter()
    total_successes = 0
    total_samples = 0
    total_invalid = 0
    for result in results:
        total_failures.update(result.failures)
        total_successes += result.successful_samples
        total_samples += result.total_samples
        total_invalid += result.invalid_outputs

    print("\n[overall]")
    print(f"samples: {total_successes}/{total_samples} successful")
    print(
        "failures: "
        f"fit_bcts={total_failures['fit_bcts']} "
        f"apply_bcts={total_failures['apply_bcts']} "
        f"em={total_failures['em']}"
    )
    print(f"invalid_outputs_not_counted_as_failures: {total_invalid}")


if __name__ == "__main__":
    main()
