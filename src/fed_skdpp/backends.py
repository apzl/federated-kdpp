"""Kernel representations used by the two thin experiment scripts.

DPPy does not expose the paper's improvement-only local search. The DPPy path
wraps the same kernel in FiniteDPP('likelihood', L=L) and evaluates that L with
the shared stable NumPy logdet routine. It is NOT a separate DPPy optimizer,
nor does it secretly substitute exact sampling or DPPy's MCMC sampler.
"""

import numpy as np

from .config import KernelConfig, NumericsConfig
from .kernels import hamming_rbf, stable_logdet, validate_kernel


class KernelObjective:
    """Dense candidate-local reference L kernel with stable subset scoring.

    Candidate positions are local; mapping them back to public global IDs is
    the experiment runner's responsibility. Matrices are float64. The dense
    global matrix is used only by the labeled centralized reference. Clients
    and server build matrices over their own candidate collections only.
    """

    def __init__(
        self, encoded: np.ndarray, kernel: KernelConfig,
        numerics: NumericsConfig, backend: str,
    ) -> None:
        if backend not in {"numpy", "dppy"}:
            raise ValueError("backend must be numpy or dppy")
        matrix = hamming_rbf(encoded, kernel.gamma, kernel.jitter)
        validate_kernel(matrix, numerics.symmetry_tolerance, numerics.psd_tolerance)
        self.dpp = None
        if backend == "dppy":
            try:
                from dppy.finite_dpps import FiniteDPP
            except ImportError as exc:
                raise RuntimeError(
                    "The DPPy script requires optional dependencies. "
                    "Install with: python -m pip install -e '.[dppy]'"
                ) from exc
            self.dpp = FiniteDPP("likelihood", L=matrix)
            matrix = self.dpp.L
        self.matrix = matrix

    def __call__(self, selected: np.ndarray) -> float:
        """Return logdet for unique local integer indices, shape (b,)."""
        return stable_logdet(self.matrix[np.ix_(selected, selected)])

