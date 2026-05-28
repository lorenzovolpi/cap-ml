import argparse
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import KDEyML
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

import cap
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import vanilla_acc
from cap.models.confidence import RQBS, PrediQuant
from cap_exp.pretrain.dataset import sort_datasets_by_size

qp.environ["_R_SEED"] = 0

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_NUM_TEST = 100

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

Dataset = tuple[str, str, tuple[LabelledCollection, LabelledCollection, LabelledCollection]]
MethodFactory = Callable[[], object]


def method_df(num_rows: int, **data) -> pd.DataFrame:
    _data = data | {k: [v] * num_rows for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(_data, orient="columns")


def limit_names(names: list[str], max_datasets: int | None) -> list[str]:
    if max_datasets is None or max_datasets < 0:
        return names
    return names[:max_datasets]


def gen_binary_datasets(max_datasets: int | None) -> Iterable[Dataset]:
    if max_datasets == 0:
        return
    names = [d for d in UCI_BINARY_DATASETS if d in NATIVE_UCI_BINARY_DATASETS]
    names = sort_datasets_by_size("uci_binary", names, descending=True)
    for dataset_name in limit_names(names, max_datasets):
        yield dataset_name, "uci_binary", fetch_UCIBinaryDataset(dataset_name)


def gen_multiclass_datasets(max_datasets: int | None) -> Iterable[Dataset]:
    if max_datasets == 0:
        return
    names = sort_datasets_by_size("uci_multiclass", list(UCI_MULTICLASS_DATASETS), descending=True)
    for dataset_name in limit_names(names, max_datasets):
        yield dataset_name, "uci_multiclass", fetch_UCIMulticlassDataset(dataset_name)


def gen_datasets(max_binary: int | None, max_multiclass: int | None) -> Iterable[Dataset]:
    yield from gen_binary_datasets(max_binary)
    yield from gen_multiclass_datasets(max_multiclass)


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
        sample_index = data.sampling_index(sample_size, *prevs, random_state=int(rng.integers(0, 2**31 - 1)))
        samples.append(data.sampling_from_index(sample_index))
    return samples


def kdey_reusing_classifier(classifier: MLPClassifier, val: LabelledCollection, seed: int) -> KDEyML:
    return KDEyML(classifier=classifier, fit_classifier=False, val_split=val.Xy, random_state=seed)


def method_factories(
    classifier: MLPClassifier,
    val: LabelledCollection,
    sample_size: int,
    num_samples: int,
    alpha: float,
    distance: str,
    seed: int,
) -> dict[str, MethodFactory]:
    return {
        "prediquant": lambda: PrediQuant(
            vanilla_acc,
            kdey_reusing_classifier(classifier, val, seed),
            alpha=alpha,
            num_samples=num_samples,
            sample_size=sample_size,
            distance=distance,
            random_state=seed,
        ),
        "rqbs": lambda: RQBS(
            vanilla_acc,
            kdey_reusing_classifier(classifier, val, seed),
            num_samples=num_samples,
            sample_size=sample_size,
            random_state=seed,
        ),
    }


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
        ci = method.predict_with_confidence(sample.X, posteriors)
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
        description="Compare PrediQuant and RQBS on the largest native UCI binary and UCI multiclass datasets."
    )
    parser.add_argument(
        "--max-binary",
        type=int,
        default=5,
        help="Number of largest native UCI binary datasets to test; -1 for all native binary datasets.",
    )
    parser.add_argument(
        "--max-multiclass",
        type=int,
        default=10,
        help="Number of largest UCI multiclass datasets to test; -1 for all multiclass datasets.",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size for each test bag.")
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST, help="Number of test bags per dataset.")
    parser.add_argument("--num-samples", type=int, default=100, help="Confidence samples per method and test bag.")
    parser.add_argument("--alpha", type=float, default=0.1, help="PrediQuant acceptance fraction.")
    parser.add_argument(
        "--distance",
        choices=["l1", "hellinger", "jensen-shannon"],
        default="l1",
        help="Distance used by PrediQuant to match validation bags to test bags.",
    )
    parser.add_argument("--mlp-max-iter", type=int, default=200, help="max_iter for the MLP classifier.")
    parser.add_argument("--seed", type=int, default=qp.environ["_R_SEED"], help="Random seed.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional path where the raw results are saved.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must be in (0, 1]")

    qp.environ["_R_SEED"] = args.seed
    qp.environ["SAMPLE_SIZE"] = args.sample_size
    dfs = []

    for dataset_i, (dataset_name, dataset_kind, (L, V, U)) in enumerate(
        gen_datasets(args.max_binary, args.max_multiclass)
    ):
        classifier = MLPClassifier(random_state=args.seed, max_iter=args.mlp_max_iter).fit(*L.Xy)
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
            classifier,
            V,
            args.sample_size,
            args.num_samples,
            args.alpha,
            args.distance,
            args.seed,
        ).items():
            method = method_factory().fit(V, val_posteriors)
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

        print(f"{dataset_kind}/{dataset_name} done.")

    df = pd.concat(dfs, axis=0)
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)

    pivot = pd.pivot_table(
        df,
        index=["dataset_kind", "dataset", "n_classes"],
        columns=["method"],
        values=["ae", "ci_coverage", "ci_width"],
        aggfunc="mean",
    )
    pivot.to_markdown(Path(args.output_csv).with_suffix(".md"), index=True)


if __name__ == "__main__":
    main(parse_args())
