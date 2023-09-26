import pandas as pd
import quapy as qp
from quapy.protocol import APP
from sklearn.linear_model import LogisticRegression
from quacc import utils

import quacc.evaluation as eval
import quacc.baseline as baseline
from quacc.estimator import (
    BinaryQuantifierAccuracyEstimator,
    MulticlassAccuracyEstimator,
)

from quacc.dataset import get_imdb, get_rcv1, get_spambase

qp.environ["SAMPLE_SIZE"] = 100

pd.set_option("display.float_format", "{:.4f}".format)

dataset_name = "imdb"


def estimate_multiclass():
    print(dataset_name)
    train, validation, test = get_imdb()

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
    # print(df.to_latex())
    print(df.to_string())
    # print(df.to_html())
    print()


def estimate_binary():
    print(dataset_name)
    train, validation, test = get_imdb()

    model = LogisticRegression()

    print(f"fitting model {model.__class__.__name__}...", end=" ", flush=True)
    model.fit(*train.Xy)
    print("fit")

    estimator = BinaryQuantifierAccuracyEstimator(model)

    print(
        f"fitting qmodel {estimator.q_model_0.__class__.__name__}...",
        end=" ",
        flush=True,
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
    # print(df.to_latex(float_format="{:.4f}".format))
    print(df.to_string())
    # print(df.to_html())
    print()

def estimate_comparison():
    train, validation, test = get_spambase()
    model = LogisticRegression()
    model.fit(*train.Xy)

    n_prevalences = 21
    repreats = 1000
    protocol = APP(test, n_prevalences=n_prevalences, repeats=repreats)

    estimator = BinaryQuantifierAccuracyEstimator(model)
    estimator.fit(validation)
    df = eval.evaluation_report(estimator, protocol, prevalence=False)
    
    df = utils.combine_dataframes(
        baseline.atc_mc(model, validation, protocol),
        baseline.atc_ne(model, validation, protocol),
        baseline.doc_feat(model, validation, protocol),
        baseline.rca_score(model, validation, protocol),
        baseline.rca_star_score(model, validation, protocol),
        baseline.bbse_score(model, validation, protocol),
        df,
        df_index=[("base", "F"), ("base", "T")]
    )

    print(df.to_latex(float_format="{:.4f}".format))
    print(utils.avg_group_report(df).to_latex(float_format="{:.4f}".format))

def main():
    estimate_comparison()

if __name__ == "__main__":
    main()
