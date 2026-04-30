from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import quapy as qp
import quapy.functional as F
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import KDEyML
from quapy.protocol import UPP
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

import cap
from cap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset
from cap.error import vanilla_acc
from cap.models.cont_table import O_LEAP
from cap.models.direct import DoC
from cap_exp.pretrain.dataset import sort_datasets_by_size

qp.environ["_R_SEED"] = 0

SAMPLE_SIZE = 1000
NUM_TEST = 100

P_TEST_Y: str = "P_test(Y)"
P_TEST_C: str = "P_test(C)"
P_C_COND_Y: str = "P(C|Y)"


def kdey():
    return KDEyML(MLPClassifier())


def model(pred_posterior_count: np.ndarray, train_class_cond_count: np.ndarray):
    train_class_count = train_class_cond_count.sum(axis=1)

    K = len(pred_posterior_count)
    L = len(train_class_count)

    pi_ = numpyro.sample(P_TEST_Y, dist.Dirichlet(jnp.ones(L)))
    p_c_cond_y = numpyro.sample(P_C_COND_Y, dist.Dirichlet(jnp.ones(K).repeat(L).reshape(L, K)))

    with numpyro.plate("plate", L):
        numpyro.sample("F_yc", dist.Multinomial(train_class_count, p_c_cond_y), obs=train_class_cond_count)

    p_c = numpyro.deterministic(P_TEST_C, jnp.einsum("yc,y->c", p_c_cond_y, pi_))
    numpyro.sample("N_c", dist.Multinomial(jnp.sum(pred_posterior_count), p_c), obs=pred_posterior_count)


def sample_posterior(
    pred_posterior_count: np.ndarray,
    train_class_cond_count: np.ndarray,
    num_warmup: int,
    num_samples: int,
    seed: int = 0,
) -> dict:
    mcmc = numpyro.infer.MCMC(
        numpyro.infer.NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    rng_key = jax.random.PRNGKey(seed)
    mcmc.run(rng_key, pred_posterior_count=pred_posterior_count, train_class_cond_count=train_class_cond_count)
    return mcmc.get_samples()


def get_ct_samples(
    posterior_count: np.ndarray, train_ct: np.ndarray, seed: int = 0
) -> tuple[np.ndarray, float, float, float]:
    samples = sample_posterior(posterior_count, train_ct, num_warmup=500, num_samples=1000, seed=seed)

    # compute P(Y,C) for all samples from P(Y) and P(C|Y)
    p_y_and_c_test = jnp.einsum("sy,syc->syc", samples[P_TEST_Y], samples[P_C_COND_Y])

    ct_mean = np.array(jax.device_get(jnp.mean(p_y_and_c_test, axis=0)))
    ct_lb = np.array(jax.device_get(jnp.percentile(p_y_and_c_test, 5, axis=0)))
    ct_ub = np.array(jax.device_get(jnp.percentile(p_y_and_c_test, 95, axis=0)))

    return p_y_and_c_test, ct_mean, ct_lb, ct_ub


def method_df(**data) -> pd.DataFrame:
    _data = data | {k: [v] * NUM_TEST for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(_data, orient="columns")


def gen_datasets() -> Iterable[tuple[str, tuple[LabelledCollection, LabelledCollection, LabelledCollection]]]:
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
    for dn in _sorted_bin_names[:5]:
        dval = fetch_UCIBinaryDataset(dn)
        yield dn, dval
    _uci_mul_names = [d for d in UCI_MULTICLASS_DATASETS]
    coll = "uci_multiclass"
    _sorted_mul_names = sort_datasets_by_size(coll, _uci_mul_names, fetch_UCIMulticlassDataset)
    for dn in _sorted_mul_names:
        dval = fetch_UCIMulticlassDataset(dn)
        yield dn, dval


def main():
    dfs = []

    for dataset_name, (L, V, U) in gen_datasets():
        h = MLPClassifier().fit(*L.Xy)

        V_P = h.predict_proba(V.X)
        V_yhat = np.argmax(V_P, axis=1)
        val_ct = confusion_matrix(V.y, V_yhat, labels=h.classes_)

        test_prot = UPP(
            U,
            sample_size=SAMPLE_SIZE,
            repeats=NUM_TEST,
            random_state=qp.environ["_R_SEED"],
            return_type="labelled_collection",
        )
        test_prot_post = [h.predict_proba(Ui.X) for Ui in test_prot()]
        test_prot_true_accs = [
            accuracy_score(Ui.y, np.argmax(Ui_P, axis=1)) for Ui, Ui_P in zip(test_prot(), test_prot_post)
        ]

        # bayes
        test_prot_cts, test_prot_estim_accs = [], []
        for Ui_P in test_prot_post:
            Ui_yhat = np.argmax(Ui_P, axis=1)
            Ui_post_count = F.counts_from_labels(Ui_yhat, h.classes_)

            p_y_and_c_test, ct_mean, ct_lb, ct_ub = get_ct_samples(Ui_post_count, val_ct)
            test_prot_cts.append(ct_mean)
            test_prot_estim_accs.append(cap.error.vanilla_acc(ct_mean))
        test_prot_ae = [ae_ for ae_ in cap.error.ae(np.array(test_prot_true_accs), np.array(test_prot_estim_accs))]
        bayes_df = method_df(
            method="bayes",
            dataset=dataset_name,
            ct=test_prot_cts,
            estim_acc=test_prot_estim_accs,
            true_acc=test_prot_true_accs,
            ae=test_prot_ae,
        )
        dfs.append(bayes_df)

        # leap
        test_prot_cts, test_prot_estim_accs = [], []
        leap = O_LEAP(vanilla_acc, kdey()).fit(V, V_P)
        for Ui, Ui_P in zip(test_prot(), test_prot_post):
            ct = leap.predict_ct(Ui.X, Ui_P)
            test_prot_cts.append(ct)
            test_prot_estim_accs.append(cap.error.vanilla_acc(ct))
        test_prot_ae = [ae_ for ae_ in cap.error.ae(np.array(test_prot_true_accs), np.array(test_prot_estim_accs))]
        leap_df = method_df(
            method="leap",
            dataset=dataset_name,
            ct=test_prot_cts,
            estim_acc=test_prot_estim_accs,
            true_acc=test_prot_true_accs,
            ae=test_prot_ae,
        )
        dfs.append(leap_df)
        print(f"{dataset_name} done.")

    df = pd.concat(dfs, axis=0)
    pivot = pd.pivot_table(df, index=["dataset"], columns=["method"], values=["ae"])
    print(pivot)


if __name__ == "__main__":
    main()
