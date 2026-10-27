"""DPPy's lazy exchange MCMC for fixed-size, forced-set conditional DPPs.

References: k-DPPs: Fixed-Size Determinantal Point Processes, Section 3,
Eq. (12); Monte Carlo Markov Chain Algorithms, Section 1's lazy Metropolis
chain; Diverse Data Expansion, Section III's immutable G0 constraint.
This is sampling, NOT the expansion paper's improvement-only Algorithm 1.

DPPy's native basis_exchange_sampler implements the exchange chain but uses
raw determinants internally. An isolated function-global copy replaces ONLY
its determinant arithmetic with Cholesky log-weights. The installed module
and its RNG/proposal/acceptance code are never modified. Saturating an uphill
ratio at one leaves the native uniform acceptance test unchanged. This is a
documented numerical correction, not a new sampler or mixing guarantee.
"""

from time import perf_counter
from types import FunctionType

import numpy as np

from .config import NumericsConfig, SearchConfig
from .kernels import stable_logdet, validate_kernel
from .objectives import ForcedSetObjective
from .search import SearchResult, SearchStep


class _LogWeight:
    """Only division is needed by DPPy's exchange acceptance test."""

    def __init__(self, value: float) -> None:
        if np.isnan(value) or value == float("inf"):
            raise ValueError("Invalid log-weight in DPPy exchange sampler")
        self.value = value

    def __truediv__(self, other: "_LogWeight") -> float:
        if not np.isfinite(other.value):
            raise ValueError("MCMC current state has zero or invalid probability")
        return float(np.exp(min(0.0, self.value - other.value)))


def conditional_l_kernel(
    matrix: np.ndarray, forced_indices: np.ndarray,
    numerics: NumericsConfig = NumericsConfig(),
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (Schur-complement L, eligible matrix indices, context logdet).

    Input L is (N,N) float64; immutable G0 is a unique integer array (a,).
    Returned L is (N-a,N-a), in the returned eligible-index order. For any
    eligible b-set Y, det(L[G0 union Y]) = det(L[G0]) det(L_cond[Y]).
    Hence a b-DPP on L_cond samples precisely the forced-set distribution.
    A Cholesky solve avoids an explicit inverse. Singular G0 fails; no jitter
    or clipping is added. Only within-tolerance floating drift is symmetrized.
    """
    objective = ForcedSetObjective(
        matrix, forced_indices, numerics.symmetry_tolerance, numerics.psd_tolerance,
    )
    eligible = objective.eligible_indices
    conditional = matrix[np.ix_(eligible, eligible)].copy()
    if len(objective.forced_indices):
        context = objective.forced_indices
        factor = np.linalg.cholesky(matrix[np.ix_(context, context)])
        projected = np.linalg.solve(factor, matrix[np.ix_(context, eligible)])
        conditional -= projected.T @ projected
    if not np.allclose(conditional, conditional.T, atol=numerics.symmetry_tolerance, rtol=0):
        raise ValueError("Schur-complement asymmetry exceeds the configured tolerance")
    conditional = (conditional + conditional.T) * 0.5
    validate_kernel(conditional, numerics.symmetry_tolerance, numerics.psd_tolerance)
    return conditional, eligible.copy(), objective.context_logdet


def _native_exchange(log_weight):
    """Bind DPPy's actual sampler body to a private logdet arithmetic helper.

    Fail explicitly if a future library version changes the helper contract.
    No monkey-patching of shared module state or third-party files is used.
    """
    try:
        from dppy.mcmc_sampling import basis_exchange_sampler
    except ImportError as exc:
        raise RuntimeError("DPPy MCMC requires the optional dppy dependencies") from exc
    native = basis_exchange_sampler
    if "det_ST" not in native.__code__.co_names:
        raise RuntimeError("Unsupported DPPy exchange implementation: review the logdet adapter")
    namespace = dict(native.__globals__)
    namespace["det_ST"] = log_weight
    return FunctionType(native.__code__, namespace, native.__name__, native.__defaults__, native.__closure__)


def sample_conditional_k_dpp(
    matrix: np.ndarray, forced_indices: np.ndarray, selection_size: int,
    rng: np.random.Generator, settings: SearchConfig,
    initial_selection: np.ndarray | None = None, save_history: bool = True,
    numerics: NumericsConfig = NumericsConfig(),
) -> tuple[SearchResult, np.ndarray, np.ndarray, dict]:
    """Return terminal G1, the whole sorted chain, joint scores and diagnostics.

    Input kernel/G0 use matrix indices. Initial and returned G1 indices use
    ELIGIBLE-local positions (as in ForcedSetObjective). Chain shape is
    (chain_steps+1,b), int64: initial state plus exactly chain_steps transitions
    including lazy/rejected steps. Scores are (chain_steps+1,) float64 joint
    logdets, not log probabilities. The terminal state is returned, never the
    best-scoring state. Burn-in marks diagnostics only; no states are deleted.

    DPPy takes an explicit RandomState derived from the supplied Generator.
    Single-state problems produce the same requested-length constant chain.
    Singular proposals have zero probability; a singular initial state fails.
    A finite budget does not certify stationarity or independence of samples.
    """
    if settings.algorithm != "dppy_mcmc":
        raise ValueError("sample_conditional_k_dpp requires algorithm=dppy_mcmc")
    conditional, eligible, context_score = conditional_l_kernel(matrix, forced_indices, numerics)
    population_size = len(eligible)
    if type(selection_size) is not int or not 0 <= selection_size <= population_size:
        raise ValueError("selection_size must be between zero and the eligible population size")
    if initial_selection is None:
        initial = np.sort(rng.choice(population_size, selection_size, replace=False))
    else:
        initial = np.asarray(initial_selection)
        if initial.ndim != 1 or not np.issubdtype(initial.dtype, np.integer):
            raise ValueError("initial_selection must be a one-dimensional integer array")
        initial = np.sort(initial.astype(np.int64, copy=True))
    if len(initial) != selection_size or len(np.unique(initial)) != selection_size:
        raise ValueError("initial_selection must have the requested number of unique items")
    if len(initial) and (initial[0] < 0 or initial[-1] >= population_size):
        raise ValueError("initial_selection contains an out-of-range eligible index")
    initial_score = stable_logdet(conditional[np.ix_(initial, initial)])
    if not np.isfinite(initial_score):
        raise ValueError("Initial selection has non-finite logdet; explicitly configure regularization")
    scores = {initial.tobytes(): initial_score}
    determinant_evaluations = 0

    def log_weight(kernel, selected, other=None):
        nonlocal determinant_evaluations
        if other is not None:
            raise RuntimeError("Exchange sampler requested a non-principal minor")
        determinant_evaluations += 1
        positions = np.sort(np.asarray(selected, dtype=np.int64))
        key = positions.tobytes()
        if key not in scores:
            scores[key] = stable_logdet(kernel[np.ix_(positions, positions)])
        return _LogWeight(scores[key])

    # Adapter creation also checks the optional dependency for degenerate sets.
    native = _native_exchange(log_weight)
    random_state_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    started = perf_counter()
    if selection_size in {0, population_size}:
        chain = np.repeat(initial[None, :], settings.chain_steps + 1, axis=0)
        reason = "single_state_chain"
    else:
        chain = np.sort(np.asarray(native(
            conditional, s_init=initial.tolist(), nb_iter=settings.chain_steps + 1,
            random_state=np.random.RandomState(random_state_seed),
        ), dtype=np.int64), axis=1)
        reason = "chain_steps"
    chain_scores = np.asarray([scores[row.tobytes()] + context_score for row in chain], dtype=np.float64)
    changes = np.any(chain[1:] != chain[:-1], axis=1)
    differences = np.diff(chain_scores)
    significant = changes & (differences > settings.improvement_tolerance)
    history = [SearchStep(0, float(chain_scores[0]), False, False, 0)] if save_history else []
    stalled = 0
    for step in range(1, len(chain)):
        stalled = 0 if significant[step - 1] else stalled + 1
        if save_history:
            history.append(SearchStep(
                step, float(chain_scores[step]), bool(changes[step - 1]),
                bool(significant[step - 1]), stalled,
            ))
    result = SearchResult(
        selected=tuple(int(i) for i in chain[-1]), initial_selection=tuple(int(i) for i in initial),
        initial_logdet=float(chain_scores[0]), logdet=float(chain_scores[-1]),
        iterations=settings.chain_steps, accepted_swaps=int(changes.sum()),
        strict_improvements=int(significant.sum()), stalled_iterations=stalled,
        stop_reason=reason, search_seconds=perf_counter() - started, history=tuple(history),
    )
    proposals = max(0, determinant_evaluations - 1)
    diagnostics = {
        "sampler": "dppy.mcmc_sampling.basis_exchange_sampler",
        "mode": "E", "laziness": 0.5, "chain_steps": settings.chain_steps,
        "chain_states": len(chain), "burn_in": settings.burn_in,
        "returned_state": "terminal; not best logdet", "mixing_guarantee": False,
        "proposed_swaps": proposals, "downhill_swaps": int((changes & (differences < 0)).sum()),
        "acceptance_rate_per_proposal": float(changes.sum() / proposals) if proposals else 0.0,
        "post_burn_in_mean_joint_logdet": float(chain_scores[settings.burn_in + 1:].mean()),
        "random_state_seed": random_state_seed,
        "numerical_adapter": "isolated native sampler globals; Cholesky log-weight acceptance; no raw determinants",
        "conditional_kernel_bytes": conditional.nbytes,
    }
    return result, chain, chain_scores, diagnostics
