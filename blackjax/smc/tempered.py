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
from typing import Any, Callable, NamedTuple, Tuple, Optional, Hashable, Union, Sequence
import jax
import jax.numpy as jnp
from math import ceil
import blackjax.smc as smc
from blackjax.smc.base import SMCState, SMCInfo
from blackjax.types import PRNGKey, PyTree

__all__ = ["TemperedSMCState", "init", "kernel"]


class TemperedSMCState(NamedTuple):
    """Current state for the tempered SMC algorithm.

    particles: PyTree
        The particles' positions.
    lmbda: float
        Current value of the tempering parameter.

    """

    particles: PyTree
    weights: jax.Array
    lmbda: float


def init(particles: PyTree):
    num_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    weights = jnp.ones(num_particles) / num_particles
    return TemperedSMCState(particles, weights, 0.0)


def kernel(logprior_fn: Callable, loglikelihood_fn: Callable, mcmc_step_fn: Callable, mcmc_init_fn: Callable,
           resampling_fn: Callable) -> Callable:
    """Build the base Tempered SMC kernel.

    Tempered SMC uses tempering to sample from a distribution given by

    .. math::
        p(x) \\propto p_0(x) \\exp(-V(x)) \\mathrm{d}x

    where :math:`p_0` is the prior distribution, typically easy to sample from
    and for which the density is easy to compute, and :math:`\\exp(-V(x))` is an
    unnormalized likelihood term for which :math:`V(x)` is easy to compute
    pointwise.

    Parameters
    ----------
    logprior_fn
        A function that computes the log density of the prior distribution
    loglikelihood_fn
        A function that returns the probability at a given
        position.
    mcmc_step_fn
        A function that creates a mcmc kernel from a log-probability density function.
    mcmc_init_fn: Callable
        A function that creates a new mcmc state from a position and a
        log-probability density function.
    resampling_fn
        A random function that resamples generated particles based of weights
    num_mcmc_iterations
        Number of iterations in the MCMC chain.

    Returns
    -------
    A callable that takes a rng_key and a TemperedSMCState that contains the current state
    of the chain and that returns a new state of the chain along with
    information about the transition.

    """

    def one_step(rng_key: PRNGKey, state: TemperedSMCState, num_mcmc_steps: int, lmbda: float, mcmc_parameters: dict,
                 batch_size: int) -> Tuple[TemperedSMCState, smc.base.SMCInfo]:
        """Move the particles one step using the Tempered SMC algorithm.

        Parameters
        ----------
        rng_key
            JAX PRNGKey for randomness
        state
            Current state of the tempered SMC algorithm
        lmbda
            Current value of the tempering parameter

        Returns
        -------
        state
            The new state of the tempered SMC algorithm
        info
            Additional information on the SMC step

        """
        new_lmbda = state.lmbda
        delta = lmbda - new_lmbda

        def log_weights_fn(position: PyTree) -> float:
            return delta * loglikelihood_fn(position)

        def tempered_logposterior_fn(position: PyTree) -> float:
            logprior = logprior_fn(position)
            tempered_loglikelihood = state.lmbda * loglikelihood_fn(position)
            return logprior + tempered_loglikelihood

        def mcmc_kernel(rng_key, position):
            state = mcmc_init_fn(position, tempered_logposterior_fn)

            def body_fn(state, rng_key):
                new_state, info = mcmc_step_fn(rng_key, state, new_lmbda, **mcmc_parameters)
                return new_state, info

            keys = jax.random.split(rng_key, num_mcmc_steps)
            last_state, info = jax.lax.scan(body_fn, state, keys)
            return last_state.position, info

        def base_smc_step(rng_key: PRNGKey, state: SMCState, update_fn: Callable, weigh_fn: Callable,
                          resample_fn: Callable, num_resampled: Optional[int] = None) -> Tuple[SMCState, SMCInfo]:
            """
            ...
            """
            updating_key, resampling_key = jax.random.split(rng_key, 2)
            num_particles = state.weights.shape[0]
            if num_resampled is None:
                num_resampled = num_particles
            resampling_idx = resample_fn(resampling_key, state.weights, num_resampled)
            particles = jax.tree_map(lambda x: x[resampling_idx], state.particles)
            keys = jax.random.split(updating_key, num_resampled)

            particles, update_info = update_fn(keys, particles)

            log_weights = weigh_fn(particles)
            logsum_weights = jax.scipy.special.logsumexp(log_weights)
            normalizing_constant = logsum_weights - jnp.log(num_particles)
            weights = jnp.exp(log_weights - logsum_weights)

            return SMCState(particles, weights), SMCInfo(resampling_idx, normalizing_constant, update_info)

        if batch_size is None:
            mcmc_kernel_fn = jax.vmap(mcmc_kernel)
        else:
            mcmc_kernel_fn = sequential_vmap(mcmc_kernel, in_axes=(0, 0), batch_size=batch_size)

        smc_state, info = base_smc_step(rng_key, SMCState(state.particles, state.weights), mcmc_kernel_fn,
                                        jax.vmap(log_weights_fn), resampling_fn)

        tempered_state = TemperedSMCState(smc_state.particles, smc_state.weights, state.lmbda + delta)
        return tempered_state, info

    return one_step


def get_batch(i, data, batch_size):
    """
    This function generates a batch (batch number i) from the given data.
    """

    # For dictionaries, we need to divide all dictionary values in batches:
    if type(data) == dict:
        batch_dict = {}
        for key, value in data.items():
            if type(value) == dict:
                subdict = {}
                for subkey, subvalue in value.items():
                    subdict[subkey] = jax.lax.dynamic_slice(subvalue, (i*batch_size, 0), (batch_size,
                                                                                          subvalue.shape[1]))
                batch_dict[key] = subdict
            else:
                batch_dict[key] = jax.lax.dynamic_slice(value, (i*batch_size, 0), (batch_size, value.shape[1]))
        return batch_dict
    else:
        # We assume the data is of type DeviceArray:
        return jax.lax.dynamic_slice(data, (i*batch_size, 0), (batch_size, data.shape[1]))


def combine_dictionaries(dict1, dict2):
    """
    This function combines two dictionaries, with the same keys, into a single dictionary.
    """
    combined = {}
    for key, value in dict1.items():
        if type(dict1[key]) == dict:
            subdict = {}
            for key2, value2 in dict1[key].items():
                subdict[key2] = jnp.concatenate([value2, dict2[key][key2]], axis=0)
            combined[key] = subdict
        else:
            combined[key] = jnp.concatenate([value, dict2[key]], axis=0)
    return combined


def combine_batches(batches):
    """
    This function combines all batches based on their data type.
    """

    n_batches = len(batches)
    if type(batches[0]) == dict:
        outputs = combine_dictionaries(batches[0], batches[1])
        for i in range(2, n_batches):
            outputs = combine_dictionaries(outputs, batches[i])
    elif type(batches[0]) == tuple:
        output_1 = combine_dictionaries(batches[0][0], batches[1][0])
        for i in range(2, n_batches):
            output_1 = combine_dictionaries(output_1, batches[i][0])
        outputs = (output_1, 0)
    else:
        raise Exception('I have not implemented the concatenation of batches of type ' + str(type) + ' yet.')
    return outputs


def sequential_vmap(f: Callable, in_axes: Union[int, Sequence[Any]] = 0, out_axes: Any = 0, batch_size: int = 100,
                    axis_name: Optional[Hashable] = None, axis_size: Optional[int] = None,
                    spmd_axis_name: Optional[Hashable] = None) -> Callable:
    """
    A sequential version of vmap from Jax (https://jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html), which
    divides the data in batches. This was implemented to avoid OOM issues.
    """

    def sequential_vmap_f(*args):
        # Compute the number of batches for the given data size and batch size:
        n = args[in_axes.index(0)].shape[0]
        n_batches = ceil(n / batch_size)

        batch_outputs = []
        for i in range(n_batches):
            batched_args = ()

            # Check for every vmap argument whether we want to divide the argument in batches, and get a batch:
            for d in range(len(args)):
                if in_axes[d] is not None:
                    batch = get_batch(i, args[d], batch_size)
                    batched_args += (batch,)
                else:
                    batched_args += (args[d],)

            # Apply the current batch to the function:
            batch_outputs += [
                jax.vmap(f, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name, axis_size=axis_size,
                         spmd_axis_name=spmd_axis_name)(*batched_args)]

        # Combine all batch outputs:
        outputs = combine_batches(batch_outputs)
        return outputs

    return sequential_vmap_f