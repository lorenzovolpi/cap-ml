import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

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


def main():
    pass


if __name__ == "__main__":
    main()
