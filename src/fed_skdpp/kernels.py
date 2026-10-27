"""Float64 reference kernels and stable log-determinant evaluation."""

import numpy as np


class KernelError(ValueError):
    """A kernel is malformed or violates the numerical PSD contract."""


def euclidean_rbf(points: np.ndarray, gamma: float, jitter: float = 0.0) -> np.ndarray:
    """Return an (N, N) float64 RBF kernel for finite real (N, d) points.

    Uses exp(-gamma * squared Euclidean distance), unit quality, and explicit
    jitter only. Unlike hamming_rbf, Euclidean distance IS squared. This is
    the numerical oracle for the Euclidean MPC feasibility experiment.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 1 or not np.isfinite(points).all():
        raise KernelError("points must be a finite (N, positive dimension) array")
    if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(jitter) or jitter < 0:
        raise KernelError("gamma must be positive and jitter nonnegative, both finite")
    difference = points[:, None, :] - points[None, :, :]
    kernel = np.exp(-float(gamma) * np.sum(difference * difference, axis=2))
    kernel[np.diag_indices(len(points))] += jitter
    return kernel


def hamming_rbf(encoded: np.ndarray, gamma: float, jitter: float = 0.0) -> np.ndarray:
    """Return an (N, N) float64 L kernel, exp(-gamma * mismatch_count).

    ``encoded`` has shape (N, kmer_length). Counts are unnormalized Hamming
    distances, NOT squared Hamming distances. Unit quality scores are used.
    Positive gamma gives strict PD for distinct fixed-length sequences in exact
    arithmetic. ``jitter * I`` is an explicit model/numerical choice.

    This is an extension of the expansion paper, Section IV-B, Equation (6),
    whose linear match-fraction kernel has rank at most 3*length+1 for DNA.
    """
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or encoded.shape[1] == 0:
        raise KernelError("encoded sequences must have shape (N, positive length)")
    if not np.isfinite(encoded).all():
        raise KernelError("encoded sequences must be finite")
    if not np.isfinite(gamma) or gamma <= 0:
        raise KernelError("gamma must be finite and positive")
    if not np.isfinite(jitter) or jitter < 0:
        raise KernelError("jitter must be finite and nonnegative")
    n = len(encoded)
    distances = np.zeros((n, n), dtype=np.int64)
    for position in range(encoded.shape[1]):
        distances += encoded[:, position, None] != encoded[None, :, position]
    kernel = np.exp(-float(gamma) * distances)
    if jitter:
        kernel[np.diag_indices(n)] += jitter
    return np.asarray(kernel, dtype=np.float64)


def validate_kernel(
    kernel: np.ndarray, symmetry_tolerance: float = 1e-12,
    psd_tolerance: float = 1e-10,
) -> None:
    """Check a finite real square float64 kernel for symmetry and numerical PSD.

    Cholesky cheaply certifies positive definiteness; if it fails, eigenvalues
    distinguish a singular PSD matrix from an invalid indefinite matrix.
    No jitter, symmetrization, or eigenvalue clipping is applied implicitly.
    """
    if not isinstance(kernel, np.ndarray) or kernel.dtype != np.float64:
        raise KernelError("reference kernels must be float64 NumPy arrays")
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise KernelError("kernel must be square")
    if not np.isfinite(kernel).all():
        raise KernelError("kernel contains non-finite values")
    if not np.allclose(kernel, kernel.T, atol=symmetry_tolerance, rtol=0):
        raise KernelError("kernel is not symmetric within the configured tolerance")
    if not len(kernel):
        return
    try:
        np.linalg.cholesky(kernel)
    except np.linalg.LinAlgError:
        smallest = float(np.linalg.eigvalsh(kernel).min())
        if smallest < -psd_tolerance:
            raise KernelError(f"kernel is indefinite: smallest eigenvalue {smallest}")


def stable_logdet(kernel: np.ndarray) -> float:
    """Log determinant of a previously validated PSD float64 square matrix.

    Use Cholesky, never a raw determinant. An empty matrix has logdet 0;
    a failed positive-definite factorization returns -inf. Callers must reject
    a non-finite initial selection and report/configure regularization rather
    than silently treating near-singular results as meaningful utilities.
    """
    if not len(kernel):
        return 0.0
    try:
        factor = np.linalg.cholesky(kernel)
    except np.linalg.LinAlgError:
        return float("-inf")
    return float(2.0 * np.log(np.diag(factor)).sum())
