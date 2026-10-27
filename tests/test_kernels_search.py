from itertools import combinations

import numpy as np
import pytest

from fed_skdpp.config import SearchConfig
from fed_skdpp.kernels import KernelError, hamming_rbf, stable_logdet, validate_kernel
from fed_skdpp.search import stochastic_local_search


def test_hamming_rbf_counts_mismatches_without_squaring():
    encoded = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])
    matrix = hamming_rbf(encoded, gamma=0.5)
    assert matrix.dtype == np.float64
    assert np.allclose(np.diag(matrix), 1)
    assert matrix[0, 1] == pytest.approx(np.exp(-0.5))
    assert matrix[0, 2] == pytest.approx(np.exp(-1.5))
    assert np.array_equal(matrix, matrix.T)
    assert np.linalg.eigvalsh(matrix).min() > 0
    validate_kernel(matrix)


@pytest.mark.parametrize("matrix", [
    np.array([[1.0, 2.0], [2.0, 1.0]]),
    np.array([[1.0, 0.5], [0.0, 1.0]]),
    np.array([[float("nan")]]),
    np.ones((2, 3), dtype=np.float64),
    np.eye(2, dtype=np.float32),
])
def test_kernel_validation_rejects_malformed_input(matrix):
    with pytest.raises(KernelError):
        validate_kernel(matrix)


def test_singular_kernels_fail_clearly_and_jitter_is_explicit():
    duplicated = np.zeros((2, 6), dtype=np.int64)
    matrix = hamming_rbf(duplicated, 0.5)
    validate_kernel(matrix)  # PSD, but not positive definite.
    assert stable_logdet(matrix) == float("-inf")
    with pytest.raises(ValueError, match="non-finite"):
        stochastic_local_search(2, 2, lambda _: stable_logdet(matrix), np.random.default_rng(1), SearchConfig())
    repaired = hamming_rbf(duplicated, 0.5, jitter=1e-6)
    assert np.isfinite(stable_logdet(repaired))
    assert np.allclose(repaired - matrix, 1e-6 * np.eye(2))
    assert stable_logdet(np.empty((0, 0))) == 0
    assert stable_logdet(np.zeros((2, 2))) == float("-inf")


def test_logdet_agrees_with_slogdet_and_tiny_normalization():
    encoded = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    matrix = hamming_rbf(encoded, 0.7)
    eigenvalues = np.linalg.eigvalsh(matrix)
    total = 0.0
    for subset in combinations(range(4), 2):
        principal = matrix[np.ix_(subset, subset)]
        sign, reference = np.linalg.slogdet(principal)
        assert sign == 1
        assert stable_logdet(principal) == pytest.approx(reference)
        total += np.exp(stable_logdet(principal))
    elementary_symmetric = sum(eigenvalues[i] * eigenvalues[j] for i, j in combinations(range(4), 2))
    assert total == pytest.approx(elementary_symmetric)
    probabilities = [np.exp(stable_logdet(matrix[np.ix_(s, s)])) / total for s in combinations(range(4), 2)]
    assert sum(probabilities) == pytest.approx(1.0)


def test_nearly_identical_sequences_and_zero_quality_fixture():
    encoded = np.zeros((2, 6), dtype=np.int64)
    encoded[1, -1] = 1
    matrix = hamming_rbf(encoded, gamma=1e-6)
    validate_kernel(matrix)
    assert stable_logdet(matrix) == pytest.approx(np.linalg.slogdet(matrix)[1], abs=1e-8)
    qualities = np.array([1.0, 0.0])
    zero_quality_kernel = qualities[:, None] * matrix * qualities[None, :]
    validate_kernel(zero_quality_kernel)
    assert stable_logdet(zero_quality_kernel) == float("-inf")


def test_patience_counts_fifty_consecutive_non_improving_proposals():
    result = stochastic_local_search(10, 3, lambda _: 0.0, np.random.default_rng(9), SearchConfig())
    assert result.iterations == 50
    assert result.stalled_iterations == 50
    assert result.stop_reason == "patience"
    assert result.accepted_swaps == 50  # Ties are allowed by Algorithm 1.
    assert result.strict_improvements == 0
    assert len(result.history) == 51
    assert len(result.selected) == len(set(result.selected)) == 3


def test_search_is_monotone_reproducible_and_bounded_by_exact_optimum():
    encoded = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3]])
    matrix = hamming_rbf(encoded, 0.5)
    objective = lambda selected: stable_logdet(matrix[np.ix_(selected, selected)])
    settings = SearchConfig(patience=20)
    left = stochastic_local_search(6, 3, objective, np.random.default_rng(3), settings)
    right = stochastic_local_search(6, 3, objective, np.random.default_rng(3), settings)
    assert left.selected == right.selected
    assert left.history == right.history
    values = [step.logdet for step in left.history]
    assert np.all(np.diff(values) >= 0)
    assert left.logdet >= left.initial_logdet
    exact = max(objective(np.array(subset)) for subset in combinations(range(6), 3))
    assert left.logdet <= exact + 1e-12
    assert left.stalled_iterations == 20
    assert left.strict_improvements == sum(step.improved for step in left.history)


def test_optional_cap_and_no_swap_edge_cases():
    capped = stochastic_local_search(10, 2, lambda _: 0.0, np.random.default_rng(1), SearchConfig(max_iterations=5))
    assert capped.iterations == 5 and capped.stop_reason == "max_iterations"
    full = stochastic_local_search(3, 3, lambda _: 0.0, np.random.default_rng(1), SearchConfig())
    empty = stochastic_local_search(3, 0, lambda _: 0.0, np.random.default_rng(1), SearchConfig())
    assert full.selected == (0, 1, 2) and full.iterations == 0
    assert empty.selected == () and empty.iterations == 0
    assert full.stop_reason == empty.stop_reason == "no_possible_swap"


def test_initialization_validation_and_tolerance():
    with pytest.raises(ValueError, match="unique"):
        stochastic_local_search(3, 2, lambda _: 0.0, np.random.default_rng(1), SearchConfig(), np.array([1, 1]))
    with pytest.raises(ValueError, match="out-of-range"):
        stochastic_local_search(3, 2, lambda _: 0.0, np.random.default_rng(1), SearchConfig(), np.array([0, 3]))
    result = stochastic_local_search(
        4, 2, lambda selected: 1e-12 * float(selected.sum()),
        np.random.default_rng(1), SearchConfig(patience=20, improvement_tolerance=1e-10),
    )
    assert result.iterations == 20 and result.strict_improvements == 0
