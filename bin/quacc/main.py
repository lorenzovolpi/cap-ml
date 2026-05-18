import argparse
from collections.abc import Callable, Iterable

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
from cap.models.cont_table import O_LEAP, QuAcc1xN2, QuAccNxN
from cap_exp.pretrain.dataset import sort_datasets_by_size

qp.environ["_R_SEED"] = 0

DEFAULT_SAMPLE_SIZE = 1000
DEFAULT_NUM_TEST = 100


Dataset = tuple[str, str, tuple[LabelledCollection, LabelledCollection, LabelledCollection]]
MethodFactory = Callable[[], object]


def kdey(seed: int, max_iter: int) -> KDEyML:
    classifier = MLPClassifier(random_state=seed, max_iter=max_iter)
    return KDEyML(classifier)


def method_df(num_rows: int, **data) -> pd.DataFrame:
    _data = data | {k: [v] * num_rows for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(_data, orient="columns")


def limit_names(names: list[str], max_datasets: int | None) -> list[str]:
    if max_datasets is None or max_datasets < 0:
        return names
    return names[:max_datasets]


def gen_binary_datasets(max_datasets: int | None) -> Iterable[Dataset]:
    uci_bin_native = [
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
    names = [d for d in UCI_BINARY_DATASETS if d in uci_bin_native]
    names = sort_datasets_by_size("uci_binary", names, descending=True)
    for dataset_name in limit_names(names, max_datasets):
        yield dataset_name, "uci_binary", fetch_UCIBinaryDataset(dataset_name)


def gen_multiclass_datasets(max_datasets: int | None) -> Iterable[Dataset]:
    names = sort_datasets_by_size(
        "uci_multiclass",
        list(UCI_MULTICLASS_DATASETS),
        descending=True,
    )
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


def method_factories(seed: int, max_iter: int) -> dict[str, MethodFactory]:
    return {
        "o_leap": lambda: O_LEAP(vanilla_acc, kdey(seed, max_iter)),
        "quacc_1xn2": lambda: QuAcc1xN2(
            vanilla_acc, kdey(seed, max_iter), add_maxconf=True, add_negentropy=True, add_maxinfsoft=True
        ),
        "quacc_nxn": lambda: QuAccNxN(
            vanilla_acc, kdey(seed, max_iter), add_maxconf=True, add_negentropy=True, add_maxinfsoft=True
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
    estim_cts = []
    estim_accs = []

    for sample, posteriors in zip(test_samples, test_posteriors):
        ct = method.predict_ct(sample.X, posteriors)
        estim_cts.append(ct)
        estim_accs.append(vanilla_acc(ct))

    aes = cap.error.ae(np.array(true_accs), np.array(estim_accs))
    return method_df(
        len(test_samples),
        method=method_name,
        dataset=dataset_name,
        dataset_kind=dataset_kind,
        n_classes=test_samples[0].n_classes,
        ct=estim_cts,
        estim_acc=estim_accs,
        true_acc=true_accs,
        ae=list(aes),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare QuAcc1xN2 and QuAccNxN with O-LEAP on UCI datasets.")
    parser.add_argument("--max-binary", type=int, default=5, help="Number of UCI binary datasets to test; -1 for all.")
    parser.add_argument(
        "--max-multiclass",
        type=int,
        default=10,
        help="Number of UCI multiclass datasets to test; -1 for all.",
    )
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE, help="Sample size for each test bag.")
    parser.add_argument("--num-test", type=int, default=DEFAULT_NUM_TEST, help="Number of test bags per dataset.")
    parser.add_argument("--mlp-max-iter", type=int, default=200, help="max_iter for the MLP classifiers in KDEyML.")
    parser.add_argument("--seed", type=int, default=qp.environ["_R_SEED"], help="Random seed.")
    return parser.parse_args()


def main(args: argparse.Namespace):
    qp.environ["_R_SEED"] = args.seed
    dfs = []

    for dataset_i, (dataset_name, dataset_kind, (L, V, U)) in enumerate(
        gen_datasets(args.max_binary, args.max_multiclass)
    ):
        h = MLPClassifier(random_state=args.seed, max_iter=args.mlp_max_iter).fit(*L.Xy)
        val_posteriors = h.predict_proba(V.X)

        test_samples = random_prevalence_samples(
            U,
            sample_size=args.sample_size,
            repeats=args.num_test,
            random_state=args.seed + dataset_i,
        )
        test_posteriors = [h.predict_proba(sample.X) for sample in test_samples]
        true_accs = [
            accuracy_score(sample.y, np.argmax(posteriors, axis=1))
            for sample, posteriors in zip(test_samples, test_posteriors)
        ]

        for method_name, method_factory in method_factories(args.seed, args.mlp_max_iter).items():
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
    pivot = pd.pivot_table(
        df,
        index=["dataset_kind", "dataset", "n_classes"],
        columns=["method"],
        values=["ae"],
        aggfunc="mean",
    )
    print(pivot)


if __name__ == "__main__":
    main(parse_args())
