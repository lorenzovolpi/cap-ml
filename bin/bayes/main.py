import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import quapy as qp
import quapy.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier

import cap
from cap.data.datasets import fetch_UCIBinaryDataset

qp.environ["_R_SEED"] = 0

P_TEST_Y: str = "P_test(Y)"
P_TEST_C: str = "P_test(C)"
P_C_COND_Y: str = "P(C|Y)"


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


def main():
    L, V, U = fetch_UCIBinaryDataset("spambase")

    h = MLPClassifier().fit(*L.Xy)

    V_P = h.predict_proba(V.X)
    V_yhat = np.argmax(V_P, axis=1)
    val_ct = confusion_matrix(V.y, V_yhat, labels=h.classes_)

    U_P = h.predict_proba(U.X)
    U_yhat = np.argmax(U_P, axis=1)
    posterior_count = F.counts_from_labels(U_yhat, h.classes_)

    p_y_and_c_test, ct_mean, ct_lb, ct_ub = get_ct_samples(posterior_count, val_ct)
    print("Confusion table mean")
    print(ct_mean, type(ct_mean))

    acc = cap.error.vanilla_acc(ct_mean)
    true_acc = accuracy_score(U.y, U_yhat)

    print("Predicted and true accuracy")
    print(acc, true_acc)
    print("Confusion table lb")
    print(ct_lb)
    print("Confusion table ub")
    print(ct_ub)


if __name__ == "__main__":
    main()
