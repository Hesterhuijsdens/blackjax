"""Implementation of the Pathinder warmup for the HMC family of sampling algorithms."""
from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from blackjax.adaptation.step_size import (
    DualAveragingAdaptationState,
    dual_averaging_adaptation,
)
from blackjax.optimizers.lbfgs import lbfgs_inverse_hessian_formula_1
from blackjax.types import Array, PRNGKey, PyTree
from blackjax.vi.pathfinder import init as pathfinder_init_fn
from blackjax.vi.pathfinder import sample_from_state

__all__ = ["base"]


class PathfinderAdaptationState(NamedTuple):
    ss_state: DualAveragingAdaptationState
    step_size: float
    inverse_mass_matrix: Array


def base(
    logprob_fn: Callable,
    target_acceptance_rate: float = 0.80,
):
    """Warmup scheme for sampling procedures based on euclidean manifold HMC.

    This adaptation runs in two steps:

    1. The Pathfinder algorithm is ran and we subsequently compute an estimate
    for the value of the inverse mass matrix, as well as a new initialization
    point for the markov chain that is supposedly closer to the typical set.
    2. We then start sampling with the MCMC algorithm and use the samples to
    adapt the value of the step size using an optimization algorithm so that
    the mcmc algorithm reaches a given target acceptance rate.

    Parameters
    ----------
    logprob_fn
        The log-probability density function from which we wish to
        sample.
    target_acceptance_rate:
        The target acceptance rate for the step size adaptation.

    Returns
    -------
    init
        Function that initializes the warmup.
    update
        Function that moves the warmup one step.
    final
        Function that returns the step size and mass matrix given a warmup state.

    """
    da_init, da_update, da_final = dual_averaging_adaptation(target_acceptance_rate)

    def init(
        rng_key: PRNGKey, position: PyTree, initial_step_size: float
    ) -> Tuple[PathfinderAdaptationState, PyTree]:
        """Initialze the adaptation state and parameter values.

        We use the Pathfinder algorithm to compute an estimate of the inverse
        mass matrix that will stay constant throughout the rest of the
        adaptation.

        """
        pathfinder_key, sample_key = jax.random.split(rng_key, 2)
        pathfinder_state = pathfinder_init_fn(pathfinder_key, logprob_fn, position)
        new_position, _ = sample_from_state(sample_key, pathfinder_state)
        inverse_mass_matrix = lbfgs_inverse_hessian_formula_1(
            pathfinder_state.alpha, pathfinder_state.beta, pathfinder_state.gamma
        )

        da_state = da_init(initial_step_size)

        warmup_state = PathfinderAdaptationState(
            da_state, initial_step_size, inverse_mass_matrix
        )

        return warmup_state, new_position

    def update(
        adaptation_state: PathfinderAdaptationState,
        position: PyTree,
        acceptance_rate: float,
    ) -> PathfinderAdaptationState:
        """Update the adaptation state and parameter values.

        Since the value of the inverse mass matrix is already known we only
        update the step size state.

        Parameters
        ----------
        adaptation_state
            Current adptation state.
        adaptation_stage
            The current stage of the warmup: whether this is a slow window,
            a fast window and if we are at the last step of a slow window.
        position
            Current value of the model parameters.
        acceptance_rate
            Value of the acceptance rate for the last mcmc step.

        Returns
        -------
        The updated states of the chain and the warmup.

        """
        new_ss_state = da_update(adaptation_state.ss_state, acceptance_rate)
        new_step_size = jnp.exp(new_ss_state.log_step_size)

        return PathfinderAdaptationState(
            new_ss_state, new_step_size, adaptation_state.inverse_mass_matrix
        )

    def final(warmup_state: PathfinderAdaptationState) -> Tuple[float, Array]:
        """Return the final values for the step size and mass matrix."""
        step_size = jnp.exp(warmup_state.ss_state.log_step_size_avg)
        inverse_mass_matrix = warmup_state.inverse_mass_matrix
        return step_size, inverse_mass_matrix

    return init, update, final
