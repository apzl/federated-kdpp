"""Native DPPy spectral sampling for a fixed-size, forced-set k-DPP.

Kulesza and Taskar (2011), Section 3.1 Algorithm 2: select eigenvectors using
elementary symmetric polynomials, then use Algorithm 1's projection sampler.
DPPy's KuTa12 mode implements the latter as in their 2012 survey Algorithm 1;
the survey's Algorithms 7/8 provide the positive ESP recurrence/selection.
Conditioning on the immutable G0 uses the shared Schur-complement helper.
This is one random sample, with no optimization, initialization, or chain.
"""

from time import perf_counter

import numpy as np

from .config import NumericsConfig
from .kernels import stable_logdet
from .mcmc import conditional_l_kernel


def sample_conditional_exact_k_dpp(
    matrix: np.ndarray, forced_indices: np.ndarray, selection_size: int,
    rng: np.random.Generator, mode: str = "KuTa12",
    numerics: NumericsConfig = NumericsConfig(),
) -> tuple[np.ndarray, float, dict]:
    """Return (eligible-local G1 indices, original joint logdet, diagnostics).

    matrix is a finite symmetric PSD float64 (N,N) array; forced_indices is
    a unique integer (a,) array in MATRIX index space. The returned sorted
    int64 (selection_size,) array indexes the sorted complement of G0.
    Exactly one native FiniteDPP.sample_exact_k_dpp call is made for nonempty
    samples. Empty selection is the deterministic empty sample. An explicit
    RandomState seed is derived from rng, without touching global RNG state.

    Singular G0, negative computed eigenvalues, insufficient numerical rank,
    and invalid/overflowing ESP normalizers fail explicitly. No eigenvalue
    clipping, truncation, rescaling, or extra jitter is applied. The numerical
    rank check follows DPPy's n * eps * largest_eigenvalue rule. Full float64
    eigenvalues are passed to DPPy. Exactness refers to the sampling algorithm;
    eigen-decomposition and orthogonalization use floating-point arithmetic.
    """
    if mode not in {"KuTa12", "GS", "GS_bis"}:
        raise ValueError("mode must be KuTa12, GS, or GS_bis")
    started = perf_counter()
    conditional, eligible, context_logdet = conditional_l_kernel(matrix, forced_indices, numerics)
    n = len(eligible)
    if type(selection_size) is not int or not 0 <= selection_size <= n:
        raise ValueError("selection_size must be between zero and the eligible population size")
    conditioned = perf_counter()
    try:
        from dppy.exact_sampling import elementary_symmetric_polynomials
        from dppy.finite_dpps import FiniteDPP
        from scipy.linalg import eigh
    except ImportError as exc:
        raise RuntimeError("Exact k-DPP sampling requires the optional dppy dependencies") from exc

    seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    min_eigenvalue = max_eigenvalue = rank_tolerance = None
    rank = None
    log_normalizer = 0.0
    eig_seconds = esp_seconds = draw_seconds = 0.0
    if selection_size == 0:
        selected = np.empty(0, dtype=np.int64)
    else:
        before = perf_counter()
        values, vectors = eigh(conditional)
        eig_seconds = perf_counter() - before
        min_eigenvalue, max_eigenvalue = float(values[0]), float(values[-1])
        if not np.isfinite(values).all() or min_eigenvalue < 0:
            raise ValueError(
                "Exact sampling found a negative or non-finite eigenvalue; "
                "no clipping is applied. Review the kernel or explicitly configure jitter."
            )
        rank_tolerance = float(max_eigenvalue * n * np.finfo(np.float64).eps)
        rank = int(np.count_nonzero(values > rank_tolerance))
        if selection_size > rank:
            raise ValueError(f"selection_size={selection_size} exceeds numerical rank={rank}")
        before = perf_counter()
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                polynomials = elementary_symmetric_polynomials(values, selection_size)
        except FloatingPointError as exc:
            raise ValueError("DPPy ESP table overflowed; no automatic kernel rescaling is applied") from exc
        normalizer = polynomials[selection_size, n]
        if not np.isfinite(polynomials).all() or normalizer <= 0:
            raise ValueError("DPPy ESP normalizer is zero or non-finite")
        log_normalizer = float(np.log(normalizer))
        esp_seconds = perf_counter() - before
        # Reuse the validated eigendecomposition and the native ESP recurrence.
        dpp = FiniteDPP("likelihood", L_eig_dec=(values, vectors))
        dpp.E_poly = polynomials
        before = perf_counter()
        selected = np.sort(np.asarray(dpp.sample_exact_k_dpp(
            size=selection_size, mode=mode, random_state=np.random.RandomState(seed),
        ), dtype=np.int64))
        draw_seconds = perf_counter() - before

    if (selected.shape != (selection_size,) or len(np.unique(selected)) != selection_size
            or (len(selected) and (selected[0] < 0 or selected[-1] >= n))):
        raise RuntimeError("DPPy's exact sample violated cardinality, uniqueness, or index bounds")
    conditional_score = stable_logdet(conditional[np.ix_(selected, selected)])
    joint_ids = np.sort(np.concatenate((np.asarray(forced_indices, dtype=np.int64), eligible[selected])))
    joint_score = stable_logdet(matrix[np.ix_(joint_ids, joint_ids)])
    if not np.isfinite(conditional_score) or not np.isfinite(joint_score):
        raise ValueError("Exact sample has a non-finite determinant under the original kernel")
    identity_error = abs(joint_score - context_logdet - conditional_score)
    if identity_error > numerics.psd_tolerance:
        raise ValueError("Conditional and original joint logdet disagree beyond the numerical tolerance")
    return selected, joint_score, {
        "sampler": "dppy.finite_dpps.FiniteDPP.sample_exact_k_dpp",
        "mode": mode, "sample_count": 1,
        "native_sampler_calls": int(selection_size > 0),
        "returned_state": "single exact draw",
        "random_state_seed": seed, "context_logdet": context_logdet,
        "conditional_logdet": conditional_score,
        "conditional_log_normalizer": log_normalizer,
        "sample_log_probability": conditional_score - log_normalizer,
        "schur_identity_error": identity_error,
        "schur_identity_tolerance": numerics.psd_tolerance,
        "min_eigenvalue": min_eigenvalue, "max_eigenvalue": max_eigenvalue,
        "numerical_rank": rank, "rank_tolerance": rank_tolerance,
        "eigenvalue_policy": "error on negatives; no clipping, truncation, or rescaling",
        "conditional_kernel_bytes": conditional.nbytes,
        "conditioning_seconds": conditioned - started,
        "eigendecomposition_seconds": eig_seconds, "esp_seconds": esp_seconds,
        "draw_seconds": draw_seconds, "sampling_seconds": perf_counter() - started,
    }
