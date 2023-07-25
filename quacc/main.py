import pandas as pd
import quapy as qp
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

import quacc.evaluation as eval
from quacc.estimator import MulticlassAccuracyEstimator

from quacc.data import get_dataset

qp.environ["SAMPLE_SIZE"] = 100

pd.set_option("display.float_format", "{:.4f}".format)


def test_2(dataset_name):
    train, test = get_dataset(dataset_name)

    model = LogisticRegression()

    print(f"fitting model {model.__class__.__name__}...", end=" ", flush=True)
    model.fit(*train.Xy)
    print("fit")

    estimator = MulticlassAccuracyEstimator(model)

    print(
        f"fitting qmodel {estimator.q_model.__class__.__name__}...", end=" ", flush=True
    )
    estimator.fit(train)
    print("fit")

    n_prevalences = 21
    repreats = 1000
    protocol = APP(test, n_prevalences=n_prevalences, repeats=repreats)
    print(
        f"Tests:\n\
        protocol={protocol.__class__.__name__}\n\
        n_prevalences={n_prevalences}\n\
        repreats={repreats}\n\
        executing...\n"
    )
    df = eval.evaluation_report(
        estimator,
        protocol,
        aggregate=True,
    )
    print(df.to_string())


def main():
    for dataset_name in [
        "imdb",
        # "hp",
        # "spambase",
    ]:
        print(dataset_name)
        test_2(dataset_name)
        print("*" * 50)


if __name__ == "__main__":
    main()
