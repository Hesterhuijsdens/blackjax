import jax.scipy.stats as stats
import jax.numpy as jnp

import blackjax.hmc as hmc


def logpdf(epsilon, volatility, nu, returns):
    logpdf = 0
    logpdf += stats.expon(10.).logpdf(epsilon)

    volat = volatility[1:]
    volat_lagged = volatility[:-1]
    logpdf += jnp.sum(stats.norm(volat_lagged, epsilon).logpdf(volat))

    logpdf += stats.expon(0.1).logpdf(nu)
    logpdf += jnp.sum(stats.t(nu, jnp.exp(-2. * volatility).logpdf(returns)))

    return logpdf
