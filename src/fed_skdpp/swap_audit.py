"""Numerical exhaustive one-swap checks for an immutable forced-set logdet.

The swap neighborhood is the expansion paper's Section III, Algorithm 1.
Auditing every pair is an extension, not that paper's stopping rule. A passed
audit means no one-swap improvement beyond tolerance, NOT global MAP optimality.
Cholesky/linear solves batch all pair gains without computing raw determinants,
explicitly inverting L, or applying any audit-selected move.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import NumericsConfig, _nonnegative_float
from .kernels import KernelError, stable_logdet, validate_kernel
from .objectives import _indices


@dataclass(frozen=True)
class SwapAudit:
    total_swaps: int
    improving_swaps: int
    max_logdet_gain: float | None
    directly_verified_best_gain: float | None
    verification_error: float
    certified: bool
    best_out: int | None
    best_in: int | None
    audit_seconds: float


def audit_one_swap_neighborhood(
    matrix: np.ndarray, forced_indices: np.ndarray, selected_indices: np.ndarray,
    improvement_tolerance: float = 1e-10,
    candidate_indices: np.ndarray | None = None,
    numerics: NumericsConfig = NumericsConfig(),
) -> SwapAudit:
    """Check ALL (out,in) pairs with fixed G0, using matrix-index-space IDs.

    Inputs: validated float64 (N,N) PSD kernel; unique integer (a,) forced and
    (b,) selected IDs, disjoint; candidates are all non-forced IDs by default.
    G1 must be contained in candidates. Singular current joint kernels fail;
    singular neighbors have logdet -inf and are never improvements. Null max
    gain means no swaps or no finite neighbor. No jitter/clipping is introduced.

    For H=G0 union G1, z outside H, w solves L_H w=L_Hz, and residual
    s=L_zz-L_zH w, the swap log-gain for removing o in G1 is
    log(precision_oo*s+w_o**2). The precision diagonal is obtained from squared
    column norms of Cholesky solves; no inverse of L is formed. Verify the
    predicted best swap by a fresh Cholesky logdet. Excess discrepancy or
    negative residual arithmetic fails rather than issuing a certificate.
    This is a float64 certificate at the configured tolerance, not exact algebra.
    """
    started = perf_counter()
    _nonnegative_float("improvement_tolerance", improvement_tolerance)
    validate_kernel(matrix, numerics.symmetry_tolerance, numerics.psd_tolerance)
    forced = _indices(np.asarray(forced_indices), len(matrix), "forced_indices")
    selected = _indices(np.asarray(selected_indices), len(matrix), "selected_indices")
    candidates = (np.setdiff1d(np.arange(len(matrix), dtype=np.int64), forced)
                  if candidate_indices is None else
                  _indices(np.asarray(candidate_indices), len(matrix), "candidate_indices"))
    if np.intersect1d(forced, candidates).size or not np.isin(selected, candidates).all():
        raise ValueError("Candidates must exclude G0 and contain every selected G1 item")
    joint = np.concatenate((forced, selected))
    current_matrix = matrix[np.ix_(joint, joint)]
    current = stable_logdet(current_matrix)
    if not np.isfinite(current):
        raise KernelError("One-swap audit requires a nonsingular current G0 union G1 kernel")
    incoming = np.setdiff1d(candidates, selected)
    pair_count = len(selected) * len(incoming)
    if not pair_count:
        return SwapAudit(0, 0, None, None, 0.0, True, None, None, perf_counter() - started)
    factor = np.linalg.cholesky(current_matrix)
    columns = np.linalg.solve(factor, np.eye(len(joint), dtype=np.float64))
    precision_diag = np.sum(columns[:, len(forced):] ** 2, axis=0)
    cross = matrix[np.ix_(joint, incoming)]
    whitened = np.linalg.solve(factor, cross)
    weights = np.linalg.solve(factor.T, whitened)[len(forced):]
    residuals = np.diag(matrix)[incoming] - np.sum(whitened ** 2, axis=0)
    ratios = precision_diag[:, None] * residuals[None, :] + weights ** 2
    if not np.isfinite(ratios).all() or (ratios < 0).any():
        raise KernelError("Unreliable swap arithmetic; no clipping or certificate was applied")
    gains = np.full(ratios.shape, -np.inf, dtype=np.float64)
    np.log(ratios, out=gains, where=ratios > 0)
    improved_count = int(np.count_nonzero(gains > improvement_tolerance))
    position = np.unravel_index(np.argmax(gains), gains.shape)
    predicted = float(gains[position])
    if not np.isfinite(predicted):
        return SwapAudit(pair_count, 0, None, None, 0.0, True, None, None, perf_counter() - started)
    outgoing_id, incoming_id = int(selected[position[0]]), int(incoming[position[1]])
    proposal = joint.copy()
    proposal[len(forced) + position[0]] = incoming_id
    verified = stable_logdet(matrix[np.ix_(proposal, proposal)]) - current
    error = abs(verified - predicted)
    if not np.isfinite(verified) or error > numerics.psd_tolerance:
        raise KernelError(f"Swap-audit verification discrepancy {error}; no certificate issued")
    certified = improved_count == 0 and verified <= improvement_tolerance
    if verified > improvement_tolerance and improved_count == 0:
        improved_count = 1
    return SwapAudit(pair_count, improved_count, predicted, float(verified), float(error),
                     certified, outgoing_id, incoming_id, perf_counter() - started)
