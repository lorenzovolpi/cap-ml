import pandas as pd
import quapy as qp
from quapy.method.aggregative import SLD
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression

import quacc.evaluation as eval
from quacc.estimator import AccuracyEstimator

from .data import get_dataset

qp.environ["SAMPLE_SIZE"] = 100

pd.set_option("display.float_format", "{:.4f}".format)


def test_2(dataset_name):
    train, test = get_dataset(dataset_name)
    model = LogisticRegression()
    model.fit(*train.Xy)
    estimator = AccuracyEstimator(model, SLD(LogisticRegression()))
    estimator.fit(train)
    df = eval.evaluation_report(estimator, APP(test, n_prevalences=11, repeats=100))
    # print(df.to_string())
    print(df.to_string())


def main():
    for dataset_name in [
        # "hp",
        # "imdb",
        "spambase",
    ]:
        print(dataset_name)
        test_2(dataset_name)
        print("*" * 50)


if __name__ == "__main__":
    main()
