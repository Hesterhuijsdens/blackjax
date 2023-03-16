# Copyright 2020- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Public API for the Elliptical Slice sampling Kernel"""
from typing import Callable, NamedTuple, Tuple, Union

import jax
import jax.numpy as jnp
import jax.random as jrnd

from blackjax.types import Array, PRNGKey, PyTree
from blackjax.util import generate_gaussian_noise
from blackjax.mcmc.elliptical_slice import EllipSliceState, EllipSliceInfo


def kernel(k: Array, mean: Array, shared_hyperparameters=True, D=1, nu=1):

    # Get number of dimensions and N:
    ndim = jnp.ndim(k)
    N = k.shape[-1]

    # k is only the first column of a Toeplitz matrix. To sample from this matrix, we need to extend k to its circulant
    # column. This is described in more detail in Eq. 4.40 in:
    # Wilson (2014). Covariance kernels for fast automatic pattern discovery and extrapolation with Gaussian processes.
    if ndim == 1 and shared_hyperparameters: # first column of cingulant matrix.
        c = jnp.concatenate((k, jnp.flip(k)[1:-1]), axis=0)
        c_tilde_sqrt = jnp.sqrt(jnp.fft.fft(c))
    elif ndim == 2 and not shared_hyperparameters: # c should then be (num_latent, N).
        c = jnp.concatenate((k, jnp.flip(k)[:, 1:-1]), axis=1)
        c_tilde_sqrt = jax.vmap(jnp.fft.fft, in_axes=0)(c)
    else:
        raise ValueError('Input c has the wrong number of dimensions. It should be the first column of the circulant '
                         'matrix of K, or two-dimensional if shared_hyperparameters is set to False.')

    def momentum_generator(rng_key, position):
        def generate_gaussian_noise_latent(key, v_i, cov):
            return generate_gaussian_noise_toeplitz(key, v_i, mean, cov)

        def generate_gaussian_noise_toeplitz(rng_key: PRNGKey, v_i: Array, mu: Union[float, Array] = 0.0,
                                             sqrt_c_tilde: Union[float, Array] = 1.0) -> PyTree:
            return jnp.fft.ifft(sqrt_c_tilde * jnp.fft.fft(v_i))[:N].real

        keys = jrnd.split(rng_key, nu * D)
        u = jnp.reshape(position, (nu * D, N))

        if ndim == 1 and shared_hyperparameters: # use same covariance for all latent dimensions.
            v = jrnd.normal(rng_key, shape=(c.shape[-1], D * nu))
            res = jax.vmap(generate_gaussian_noise_latent, in_axes=(0, 1, None))(keys, v, c_tilde_sqrt)
            return jnp.reshape(res, (nu * D * N), order='F')

        elif ndim == 2 and not shared_hyperparameters: # use separate covariance for each latent dimension.
            v = jrnd.normal(rng_key, shape=(c.shape[-1], D*nu))
            res = jax.vmap(generate_gaussian_noise_latent, in_axes=(0, 1, 0))(keys, v, c_tilde_sqrt)
            return jnp.reshape(res, (nu * D * N), order='F')

        else:
            v = jrnd.normal(rng_key, shape=(c.shape[-1],))
            return generate_gaussian_noise_toeplitz(rng_key, v, mean, c_tilde_sqrt)

    def one_step(rng_key: PRNGKey, state: EllipSliceState, logdensity_fn: Callable) \
            -> Tuple[EllipSliceState, EllipSliceInfo]:
        proposal_generator = elliptical_proposal(logdensity_fn, momentum_generator, mean)
        return proposal_generator(rng_key, state)

    return one_step


def elliptical_proposal(
    logdensity_fn: Callable,
    momentum_generator: Callable,
    mean: Array,
) -> Callable:
    """Build an Ellitpical slice sampling kernel.

    The algorithm samples a latent parameter, traces an ellipse connecting the
    initial position and the latent parameter and does slice sampling on this
    ellipse to output a new sample from the posterior distribution.

    Parameters
    ----------
    logdensity_fn
        A function that returns the log-likelihood at a given position.
    momentum_generator
        A function that generates a new latent momentum variable.

    Returns
    -------
    A kernel that takes a rng_key and a Pytree that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def generate(
        rng_key: PRNGKey, state: EllipSliceState
    ) -> Tuple[EllipSliceState, EllipSliceInfo]:
        position, logdensity = state
        key_momentum, key_uniform, key_theta = jax.random.split(rng_key, 3)
        # step 1: sample momentum
        momentum = momentum_generator(key_momentum, position)
        # step 2: get slice (y)
        logy = logdensity + jnp.log(jax.random.uniform(key_uniform))
        # step 3: get theta (ellipsis move), set inital interval
        theta = 2 * jnp.pi * jax.random.uniform(key_theta)
        theta_min = theta - 2 * jnp.pi
        theta_max = theta
        # step 4: proposal
        p, m = ellipsis(position, momentum, theta, mean)
        # step 5: acceptance
        logdensity = logdensity_fn(p)

        def slice_fn(vals):
            """Perform slice sampling around the ellipsis.

            Checks if the proposed position's likelihood is larger than the slice
            variable. Returns the position if True, shrinks the bracket for sampling
            `theta` and samples a new proposal if False.

            As the bracket `[theta_min, theta_max]` shrinks, the proposal gets closer
            to the original position, which has likelihood larger than the slice variable.
            It is guaranteed to stop in a finite number of iterations as long as the
            likelihood is continuous with respect to the parameter being sampled.

            """
            rng, _, subiter, theta, theta_min, theta_max, *_ = vals
            rng, thetak = jax.random.split(rng)
            theta = jax.random.uniform(thetak, minval=theta_min, maxval=theta_max)
            p, m = ellipsis(position, momentum, theta, mean)
            logdensity = logdensity_fn(p)
            theta_min = jnp.where(theta < 0, theta, theta_min)
            theta_max = jnp.where(theta > 0, theta, theta_max)
            subiter += 1
            return rng, logdensity, subiter, theta, theta_min, theta_max, p, m

        _, logdensity, subiter, theta, *_, position, momentum = jax.lax.while_loop(
            lambda vals: vals[1] <= logy,
            slice_fn,
            (rng_key, logdensity, 1, theta, theta_min, theta_max, p, m),
        )
        return (
            EllipSliceState(position, logdensity),
            EllipSliceInfo(momentum, theta, subiter),
        )

    return generate


def ellipsis(position, momentum, theta, mean):
    """Generate proposal from the ellipsis.

    Given a scalar theta indicating a point on the circumference of the ellipsis
    and the shared mean vector for both position and momentum variables,
    generate proposed position and momentum to later accept or reject
    depending on the slice variable.

    """
    position, unravel_fn = jax.flatten_util.ravel_pytree(position)
    momentum, _ = jax.flatten_util.ravel_pytree(momentum)
    position_centered = position - mean
    momentum_centered = momentum - mean
    return (
        unravel_fn(
            position_centered * jnp.cos(theta)
            + momentum_centered * jnp.sin(theta)
            + mean
        ),
        unravel_fn(
            momentum_centered * jnp.cos(theta)
            - position_centered * jnp.sin(theta)
            + mean
        ),
    )