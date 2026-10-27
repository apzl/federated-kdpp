"""Ordinary DPP draws and a coupled, exact fixed-size rejection sweep.

Kulesza--Taskar (2011), Eq. (4), Algorithm 1: the local L-ensemble law is
det(L_A)/det(I+L), with no cardinality constraint. Server draws target their
Eq. (12), conditional on the realized pool. The server rejection construction
below is an implementation extension, NOT their spectral Algorithm 2.
"""

from time import perf_counter

import numpy as np

from .config import NumericsConfig
from .kernels import stable_logdet, validate_kernel


class OrdinaryDPPSampler:
    """Cache a real float64 (n,n) L eigendecomposition; draw variable-size sets.

    Returned sorted int64 indices are in this matrix's row space. The matrix
    is copied so subsequent caller mutations cannot invalidate the spectrum.
    DPPy's GS projection sampler is exact in ideal arithmetic, like KuTa12,
    and avoids repeated eigendecompositions. Negatives are never clipped.
    Empty and singular PSD kernels are allowed; nonfinite scores fail.
    """

    def __init__(self, matrix: np.ndarray, numerics: NumericsConfig = NumericsConfig()):
        from dppy.finite_dpps import FiniteDPP
        from scipy.linalg import eigh

        started = perf_counter()
        validate_kernel(matrix, numerics.symmetry_tolerance, numerics.psd_tolerance)
        self.matrix = matrix.copy()
        n = len(matrix)
        values, vectors = eigh(matrix) if n else (np.empty(0), np.empty((0, 0)))
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Ordinary DPP requires nonnegative eigenvalues; no clipping is applied")
        self.dpp = FiniteDPP("likelihood", L_eig_dec=(values, vectors)) if n else None
        probabilities = values / (1 + values)
        self.diagnostics = {
            "population_size": n,
            "log_normalizer": float(np.log1p(values).sum()),
            "expected_size": float(probabilities.sum()),
            "size_variance": float(np.sum(probabilities * (1 - probabilities))),
            "min_eigenvalue": float(values.min()) if n else None,
            "max_eigenvalue": float(values.max()) if n else None,
            "eigenvalue_policy": "no clipping, truncation, rescaling or implicit jitter",
            "preparation_seconds": perf_counter() - started,
        }

    def sample(self, rng: np.random.Generator) -> tuple[np.ndarray, float, dict]:
        """One ordinary DPP draw; no hidden global RNG state or size filtering."""
        started = perf_counter()
        seed = int(rng.integers(0, 2**32, dtype=np.uint64))
        selected = (np.sort(np.asarray(self.dpp.sample_exact(
            mode="GS", random_state=np.random.RandomState(seed)), dtype=np.int64))
            if self.dpp is not None else np.empty(0, dtype=np.int64))
        if self.dpp is not None:
            self.dpp.flush_samples()
        if (len(np.unique(selected)) != len(selected)
                or (len(selected) and (selected[0] < 0 or selected[-1] >= len(self.matrix)))):
            raise RuntimeError("DPPy returned invalid ordinary-DPP indices")
        score = stable_logdet(self.matrix[np.ix_(selected, selected)])
        if not np.isfinite(score):
            raise ValueError("Ordinary DPP draw has a nonfinite original-kernel logdet")
        return selected, score, {
            **self.diagnostics, "selection_size": len(selected),
            "random_state_seed": seed, "sampler": "DPPy.sample_exact(mode=GS)",
            "sample_log_probability": score - self.diagnostics["log_normalizer"],
            "draw_seconds": perf_counter() - started,
        }


def prefix_logdets(matrix: np.ndarray, order: np.ndarray, stop: int) -> np.ndarray:
    """Logdets for prefixes 0..stop, using one float64 Cholesky factorization.

    matrix is a previously validated SPD (n,n) array. order contains unique
    matrix-local indices. Output[i] = logdet(matrix[order[:i],order[:i]]).
    Empty prefixes have score zero. This helper does not regularize failures.
    """
    from scipy.linalg import cholesky

    if type(stop) is not int or not 0 <= stop <= len(order):
        raise ValueError("stop must be an integer in [0, len(order)]")
    if not stop:
        return np.zeros(1)
    ids = np.asarray(order)[:stop]
    factor = cholesky(matrix[np.ix_(ids, ids)], lower=True, check_finite=False)
    return np.r_[0.0, np.cumsum(2 * np.log(np.diag(factor)))]


def sweep_sizes(pool_size: int, minimum: int = 10, interval: int = 5) -> list[int]:
    """Strict interval grid <=m, plus endpoint m when it is off the grid.

    If m < minimum, the grid is empty; the endpoint is still included, even
    for m=0. Thus no rare local-DPP draw is rejected just to obtain a big pool.
    """
    if type(pool_size) is not int or pool_size < 0:
        raise ValueError("pool_size must be a nonnegative integer")
    if type(minimum) is not int or minimum < 1 or type(interval) is not int or interval < 1:
        raise ValueError("minimum and interval must be positive integers")
    sizes = list(range(minimum, pool_size + 1, interval))
    if not sizes or sizes[-1] != pool_size:
        sizes.append(pool_size)
    return sizes


def selected_indices(record: dict, permutations: list[list[int]]) -> np.ndarray:
    """Reconstruct one server draw's MATRIX-local IDs from a compact record."""
    return np.asarray(permutations[record["permutation_index"]][
        record["slice_start"]:record["slice_stop"]], dtype=np.int64)


def exact_rejection_sweep(
    matrix: np.ndarray, selection_sizes: list[int], rng: np.random.Generator,
    numerics: NumericsConfig = NumericsConfig(), progress=None,
) -> dict:
    """Exact marginal k-DPP draws for MANY sizes; draws across sizes are coupled.

    Requires an SPD float64 (m,m) matrix L with constant positive diagonal d.
    Write Q=L/d, so Q has unit diagonal and defines the SAME fixed-size law.
    For k<=m/2, propose uniform k-subsets Y and accept with det(Q_Y)<=1
    (Hadamard). One uniform permutation supplies a uniform prefix for every k.

    For k>m/2, propose uniform (m-k)-subsets Z and return Y=S\\Z. Jacobi's
    complementary-minor identity gives det(Q_Y)=det(Q)det((Q^-1)_Z).
    Hadamard bounds the latter by the product of the largest m-k diagonal
    entries of Q^-1. Accept using this fixed, size-dependent upper bound.
    A Cholesky solve constructs Q^-1 once; it is not a private-context inverse.

    Each size uses its FIRST accepted proposal in an unbounded rejection loop.
    Shared permutations and uniform thresholds couple sizes, but do not change
    any size's uniform proposal or rejection acceptance law. Samples are NOT
    independent across k, NOT a single nested sample, and NOT MAP solutions.
    No approximation, eigenvalue clipping, truncation or kernel jitter occurs.
    Exactness is in ideal arithmetic, subject to recorded float64 errors.

    Output records reference compact permutation/slice representations, with
    matrix-local indices. No global (10000,10000) matrix is required. The dense
    pool kernel and its inverse are appropriate only for this baseline.
    """
    from scipy.linalg import cholesky, cho_solve

    started = perf_counter()
    validate_kernel(matrix, numerics.symmetry_tolerance, numerics.psd_tolerance)
    n = len(matrix)
    if (not selection_sizes or len(set(selection_sizes)) != len(selection_sizes)
            or any(type(size) is not int or not 0 <= size <= n for size in selection_sizes)):
        raise ValueError("selection_sizes must be distinct integers between zero and population size")
    if not n:
        return {"records": [{"selection_size": 0, "logdet": 0.0,
                            "permutation_index": 0, "slice_start": 0, "slice_stop": 0,
                            "proposal_round": 0, "method": "deterministic_endpoint"}],
                "permutations": [[]], "diagnostics": {"population_size": 0, "rounds": {}}}
    diagonal = np.diag(matrix)
    scale = float(diagonal[0])
    # Exact equality is intentional: a nearly constant diagonal is NOT enough
    # to remove quality weights without changing the target.
    if scale <= 0 or not np.all(diagonal == scale):
        raise ValueError("Rejection sweep requires a constant positive diagonal")
    q = matrix if scale == 1 else matrix / scale
    factor = cholesky(q, lower=True, check_finite=False)
    full_logdet = float(2 * np.log(np.diag(factor)).sum())
    log_scale = float(np.log(scale))
    permutations: list[list[int]] = []
    records: list[dict] = []
    rounds = {}
    inverse = None
    inverse_symmetry_error = None
    inverse_identity_error = None
    if any(n // 2 < size < n for size in selection_sizes):
        inverse = cho_solve((factor, True), np.eye(n), check_finite=False)
        inverse_symmetry_error = float(np.max(np.abs(inverse - inverse.T)))
        if inverse_symmetry_error > numerics.symmetry_tolerance:
            raise ValueError("Computed pool inverse lost symmetry beyond tolerance")
        inverse = (inverse + inverse.T) * 0.5  # floating-point drift only
        # Check solve residual on deterministic columns without a second O(n^3)
        # multiplication. Never silently fix a failed complementary identity.
        probes = np.unique(np.linspace(0, n - 1, min(n, 8), dtype=int))
        residual = q @ inverse[:, probes] - np.eye(n)[:, probes]
        inverse_identity_error = float(np.max(np.abs(residual)))
        if inverse_identity_error > numerics.psd_tolerance:
            raise ValueError("Pool inverse solve residual exceeds tolerance")
    del factor
    preparation_seconds = perf_counter() - started
    for group in ("direct", "complement"):
        pending = {size for size in selection_sizes if 0 < size < n
                   and (size <= n // 2) == (group == "direct")}
        round_number = 0
        if group == "complement" and pending:
            if inverse is None:
                raise RuntimeError("Missing complementary kernel")
            if np.any(np.diag(inverse) <= 0):
                raise ValueError("Pool inverse has a nonpositive diagonal")
            log_bounds = np.r_[0., np.cumsum(np.sort(np.log(np.diag(inverse)))[::-1])]
        while pending:
            round_number += 1
            order = rng.permutation(n)
            # 1-U is in (0,1], avoiding log(0) without clipping randomness.
            log_uniform = float(np.log1p(-rng.random()))
            stop = max(pending) if group == "direct" else n - min(pending)
            scores = prefix_logdets(q if group == "direct" else inverse, order, stop)
            accepted = []
            for size in sorted(pending):
                minor_size = size if group == "direct" else n - size
                bound = 0.0 if group == "direct" else float(log_bounds[minor_size])
                log_accept = float(scores[minor_size] - bound)
                if log_accept > numerics.psd_tolerance:
                    raise ValueError("Hadamard rejection bound violated beyond tolerance")
                if log_uniform <= log_accept:
                    start, end = (0, size) if group == "direct" else (minor_size, n)
                    score = (scores[size] if group == "direct"
                             else full_logdet + scores[minor_size]) + size * log_scale
                    records.append({
                        "selection_size": size, "logdet": float(score),
                        "permutation_index": len(permutations),
                        "slice_start": start, "slice_stop": end,
                        "proposal_round": round_number, "method": group,
                        "log_acceptance_probability": min(0.0, log_accept),
                        "log_uniform": log_uniform,
                        "acceptance_bound_roundoff": max(0.0, log_accept),
                    })
                    accepted.append(size)
            if accepted:
                permutations.append(order.tolist())
                pending.difference_update(accepted)
            if progress is not None:
                progress({"stage": "server_rejection", "branch": group,
                          "round": round_number, "accepted": len(accepted), "remaining": len(pending)})
        rounds[group] = round_number
    if 0 in selection_sizes or n in selection_sizes:
        index = len(permutations)
        permutations.append(list(range(n)))
        for size in (0, n):
            if size in selection_sizes:
                records.append({"selection_size": size,
                                "logdet": full_logdet + n * log_scale if size else 0.0,
                                "permutation_index": index, "slice_start": 0, "slice_stop": size,
                                "proposal_round": 0, "method": "deterministic_endpoint"})
    records.sort(key=lambda row: row["selection_size"])
    return {"records": records, "permutations": permutations, "diagnostics": {
        "population_size": n, "full_pool_logdet": full_logdet + n * log_scale,
        "constant_diagonal": scale, "rounds": rounds,
        "sampler": "uniform rejection with Hadamard bounds and Jacobi complementary minors",
        "target": "P(Y | realized pool, |Y|=selection_size) proportional to det(L_Y)",
        "cross_size_coupling": "shared uniform permutations and thresholds; first acceptance per size",
        "independent_across_sizes": False, "nested_across_sizes": False,
        "inverse_symmetry_error": inverse_symmetry_error,
        "inverse_probe_residual_max": inverse_identity_error,
        "preparation_seconds": preparation_seconds,
        "total_seconds": perf_counter() - started,
        "numerical_policy": "float64, no clipping/truncation/jitter; tolerance bounds roundoff only",
    }}
