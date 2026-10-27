"""Forced-set objectives from the expansion paper, Section III, Algorithm 1.

The context is immutable. The optimizer's index space contains ONLY eligible
new items, so a forced item cannot be swapped out or proposed as an addition.
These numerical objects are local reference computations, not a privacy
protocol or a mechanism for transmitting client contexts.
"""

import numpy as np

from .kernels import stable_logdet, validate_kernel


def _indices(values: np.ndarray, population_size: int, name: str) -> np.ndarray:
    """Validate a unique, one-dimensional integer index array; return a copy."""
    values = np.asarray(values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"{name} must contain unique indices")
    if len(values) and (values.min() < 0 or values.max() >= population_size):
        raise ValueError(f"{name} contains an out-of-range index")
    return np.sort(values.astype(np.int64, copy=True))


class ForcedSetObjective:
    """Score logdet(L[G0 union G1]) while keeping G0 fixed.

    ``matrix`` is a validated (N, N) float64 L kernel in the input item index
    space. ``forced_indices`` is an (a,) integer array in that space. The
    callable receives unique indices into ``eligible_indices`` (NOT matrix
    indices), shape (selection_size,). Its output is the joint logdet; ``gain``
    subtracts the constant context logdet for evaluation.

    Empty G0 reduces exactly to ordinary logdet. Singular/numerically
    unfactorable G0 fails explicitly, since its conditional gain is undefined.
    No inverse, implicit jitter, clipping, or modification of G0 is used.
    The caller owns ``matrix`` and must not mutate it during optimization.
    """

    def __init__(
        self, matrix: np.ndarray, forced_indices: np.ndarray,
        symmetry_tolerance: float = 1e-12, psd_tolerance: float = 1e-10,
    ) -> None:
        validate_kernel(matrix, symmetry_tolerance, psd_tolerance)
        self.matrix = matrix
        self.forced_indices = _indices(forced_indices, len(matrix), "forced_indices")
        self.forced_indices.setflags(write=False)
        self.eligible_indices = np.setdiff1d(
            np.arange(len(matrix), dtype=np.int64), self.forced_indices,
        )
        self.eligible_indices.setflags(write=False)
        self.context_logdet = stable_logdet(matrix[np.ix_(self.forced_indices, self.forced_indices)])
        if not np.isfinite(self.context_logdet):
            raise ValueError(
                "Forced-set kernel has non-finite logdet; conditional gain is "
                "undefined. Check duplicates, gamma, or explicitly configure jitter"
            )

    def selected_indices(self, selected: np.ndarray) -> np.ndarray:
        """Map eligible-local G1 indices to matrix indices, without overlap."""
        positions = _indices(selected, len(self.eligible_indices), "selected")
        return self.eligible_indices[positions]

    def __call__(self, selected: np.ndarray) -> float:
        added = self.selected_indices(selected)
        joint = np.sort(np.concatenate((self.forced_indices, added)))
        return stable_logdet(self.matrix[np.ix_(joint, joint)])

    def gain(self, selected: np.ndarray) -> float:
        """Conditional gain: joint logdet minus the fixed context logdet."""
        return self(selected) - self.context_logdet
