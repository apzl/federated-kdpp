"""Stochastic swap optimization, not MCMC sampling.

Reference: Diverse Data Expansion with Semi-Supervised k-DPPs, Section III,
Algorithm 1, with an injectable ordinary or forced-set logdet objective. The
caller's candidate index space excludes immutable forced items. A fixed
proposal budget follows the paper's stopping rule; patience is an extension. This has
no MAP-optimality, local-optimality, or Markov-chain mixing guarantee.
"""

from dataclasses import dataclass, replace
import math
from time import perf_counter
from typing import Callable

import numpy as np

from .config import SearchConfig
from .objectives import ForcedSetObjective


@dataclass(frozen=True)
class SearchStep:
    iteration: int
    logdet: float
    accepted: bool
    improved: bool
    stalled_iterations: int


@dataclass(frozen=True)
class SearchResult:
    selected: tuple[int, ...]
    initial_selection: tuple[int, ...]
    initial_logdet: float
    logdet: float
    iterations: int
    accepted_swaps: int
    strict_improvements: int
    stalled_iterations: int
    stop_reason: str
    search_seconds: float
    history: tuple[SearchStep, ...]


def stochastic_local_search(
    population_size: int,
    selection_size: int,
    objective: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    settings: SearchConfig,
    initial_selection: np.ndarray | None = None,
    save_history: bool = True,
    on_step: Callable[[SearchStep, np.ndarray], str | None] | None = None,
    proposal_budget: int | None = None,
) -> SearchResult:
    """Optimize an injected objective over fixed-size sets of local indices.

    ``objective`` receives a sorted int64 (selection_size,) array in the index
    space range(population_size). It must be deterministic for a given set.
    Choose an incoming item uniformly from the complement and an outgoing item
    uniformly from the current set. Accept only nondecreasing scores, including
    ties. Reset patience only for an accepted increase greater than the absolute
    improvement_tolerance; smaller increases and ties count toward patience.

    Return unique sorted local indices, a monotone history if requested, and
    the actual proposal count. Fail clearly for a non-finite initial utility.
    No hidden RNG, automatic numerical repair, or MCMC acceptance is used.
    Optional on_step receives each completed step and a COPY of eligible-local
    selected IDs; returning a nonempty string stops with that reason. It must
    not consume the proposal RNG. Default stopping and proposal streams are
    unchanged. This hook supports separately documented convergence checks.

    A positive proposal_budget runs that many attempted swaps with patience
    disabled, unless on_step explicitly requests termination. This recovers
    the paper's fixed t. max_iterations must not be smaller than that budget.
    If no swap exists (empty/full selection), return zero proposals rather
    than inventing repeated swaps. A budget does not certify convergence.
    """
    if settings.algorithm != "stochastic_local_search":
        raise ValueError("stochastic_local_search requires its own algorithm setting, not MCMC")
    if type(population_size) is not int or population_size < 0:
        raise ValueError("population_size must be a nonnegative integer")
    if type(selection_size) is not int or not 0 <= selection_size <= population_size:
        raise ValueError("selection_size must be between zero and population_size")
    if proposal_budget is not None:
        if type(proposal_budget) is not int or proposal_budget < 1:
            raise ValueError("proposal_budget must be a positive integer")
        if settings.max_iterations is not None and proposal_budget > settings.max_iterations:
            raise ValueError("proposal_budget exceeds the max_iterations safety cap")
    if initial_selection is None:
        selected = np.sort(rng.choice(population_size, selection_size, replace=False))
    else:
        initial = np.asarray(initial_selection)
        if initial.ndim != 1 or not np.issubdtype(initial.dtype, np.integer):
            raise ValueError("initial_selection must be a one-dimensional integer array")
        selected = np.sort(initial.astype(np.int64, copy=True))
    if len(selected) != selection_size or len(np.unique(selected)) != selection_size:
        raise ValueError("initial_selection must have the requested number of unique items")
    if len(selected) and (selected[0] < 0 or selected[-1] >= population_size):
        raise ValueError("initial_selection contains an out-of-range local index")
    current = float(objective(selected.copy()))
    if not math.isfinite(current):
        raise ValueError(
            "Initial selection has non-finite logdet; check kernel conditioning, "
            "duplicates, gamma, or explicitly configure jitter"
        )
    initial_score = current
    initial_tuple = tuple(int(value) for value in selected)
    complement = np.setdiff1d(np.arange(population_size, dtype=np.int64), selected)
    stalled = iterations = accepted = improvements = 0
    history = [SearchStep(0, current, False, False, 0)] if save_history else []
    reason = "patience"
    start = perf_counter()
    if selection_size == 0 or selection_size == population_size:
        reason = "no_possible_swap"
    else:
        while proposal_budget is not None or stalled < settings.patience:
            if proposal_budget is not None and iterations >= proposal_budget:
                reason = "proposal_budget"
                break
            if settings.max_iterations is not None and iterations >= settings.max_iterations:
                reason = "max_iterations"
                break
            incoming_position = int(rng.integers(len(complement)))
            outgoing_position = int(rng.integers(selection_size))
            outgoing = selected[outgoing_position]
            proposal = selected.copy()
            proposal[outgoing_position] = complement[incoming_position]
            proposal.sort()
            score = float(objective(proposal.copy()))
            is_accepted = math.isfinite(score) and score >= current
            improved = is_accepted and score - current > settings.improvement_tolerance
            if is_accepted:
                selected = proposal
                complement[incoming_position] = outgoing
                current = score
                accepted += 1
            stalled = 0 if improved else stalled + 1
            improvements += int(improved)
            iterations += 1
            if save_history or on_step is not None:
                step = SearchStep(iterations, current, is_accepted, improved, stalled)
                if save_history:
                    history.append(step)
                if on_step is not None:
                    requested_stop = on_step(step, selected.copy())
                    if requested_stop is not None:
                        if not isinstance(requested_stop, str) or not requested_stop:
                            raise ValueError("on_step must return None or a nonempty stop reason")
                        reason = requested_stop
                        break
    return SearchResult(
        selected=tuple(int(value) for value in selected),
        initial_selection=initial_tuple,
        initial_logdet=initial_score, logdet=current, iterations=iterations,
        accepted_swaps=accepted, strict_improvements=improvements,
        stalled_iterations=stalled, stop_reason=reason,
        search_seconds=perf_counter() - start, history=tuple(history),
    )


def semi_supervised_local_search(
    matrix: np.ndarray, forced_indices: np.ndarray, selection_size: int,
    rng: np.random.Generator, settings: SearchConfig,
    initial_selection: np.ndarray | None = None, save_history: bool = True,
    symmetry_tolerance: float = 1e-12, psd_tolerance: float = 1e-10,
    proposal_budget: int | None = None,
) -> SearchResult:
    """Paper Algorithm 1 with a fixed G0 and swaps ONLY in G1.

    ``matrix`` is an (N, N) float64 public L kernel; ``forced_indices`` is a
    unique (a,) integer array of matrix indices. ``selection_size`` counts NEW
    items, not a+b. Optional initial_selection and returned selected/initial
    indices use the MATRIX index space. Every score/history value is joint
    logdet(L[G0 union G1]), with G0 unchanged and G1 disjoint from it.

    Uniform proposals and nondecreasing acceptance follow Section III,
    Algorithm 1 of Diverse Data Expansion with Semi-Supervised k-DPPs. The
    configured patience stopping rule is the user's extension of the paper's
    fixed proposal budget. Passing proposal_budget uses the paper's fixed t
    instead. Invalid context/initialization or non-finite initial
    joint logdet fails explicitly; no numerical repairs are implicit.

    A dense matrix is appropriate for a centralized reference or local client
    collection, not a scalable server-side private-context representation.
    """
    objective = ForcedSetObjective(matrix, forced_indices, symmetry_tolerance, psd_tolerance)
    initial_local = None
    if initial_selection is not None:
        initial = np.asarray(initial_selection)
        if initial.ndim != 1 or not np.issubdtype(initial.dtype, np.integer):
            raise ValueError("initial_selection must be a one-dimensional integer array")
        if not np.isin(initial, objective.eligible_indices).all():
            raise ValueError("initial_selection must exclude G0 and contain only eligible matrix indices")
        initial_local = np.searchsorted(objective.eligible_indices, initial)
    result = stochastic_local_search(
        len(objective.eligible_indices), selection_size, objective, rng, settings,
        initial_selection=initial_local, save_history=save_history,
        proposal_budget=proposal_budget,
    )
    return replace(
        result,
        selected=tuple(int(objective.eligible_indices[i]) for i in result.selected),
        initial_selection=tuple(int(objective.eligible_indices[i]) for i in result.initial_selection),
    )
