from itertools import combinations

import numpy as np
import pytest

from fed_skdpp.config import (
    DataConfig, ExperimentConfig, PartitionConfig, SearchConfig, SelectionConfig,
)
from fed_skdpp.data import sample_kmers, split_context
from fed_skdpp.experiment import _rng, _run_stage, run_trial
from fed_skdpp.kernels import hamming_rbf, stable_logdet
from fed_skdpp.objectives import ForcedSetObjective
from fed_skdpp.search import semi_supervised_local_search, stochastic_local_search


@pytest.fixture
def matrix():
    return hamming_rbf(np.array([
        [0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3],
    ]), 0.5)


def test_forced_objective_uses_union_and_preserves_context(matrix):
    original = np.array([3, 0], dtype=np.int64)
    objective = ForcedSetObjective(matrix, original)
    original[:] = 5  # Mutating the caller's array cannot change G0.
    assert tuple(objective.forced_indices) == (0, 3)
    assert tuple(objective.eligible_indices) == (1, 2, 4, 5)
    assert not objective.forced_indices.flags.writeable
    for selected in combinations(range(4), 2):
        local = np.array(selected, dtype=np.int64)
        joint = np.union1d(objective.forced_indices, objective.selected_indices(local))
        assert len(joint) == 4
        assert objective(local) == pytest.approx(stable_logdet(matrix[np.ix_(joint, joint)]))
        assert objective.gain(local) == pytest.approx(objective(local) - objective.context_logdet)
    assert tuple(objective.forced_indices) == (0, 3)


def test_empty_context_reduces_exactly_to_regular_objective(matrix):
    objective = ForcedSetObjective(matrix, np.empty(0, dtype=np.int64))
    assert objective.context_logdet == 0
    selected = np.array([1, 2, 4], dtype=np.int64)
    reference = stable_logdet(matrix[np.ix_(selected, selected)])
    assert objective(selected) == reference
    assert objective.gain(selected) == reference


def test_conditional_gain_agrees_with_schur_complement_using_solve(matrix):
    context = np.array([0, 3], dtype=np.int64)
    objective = ForcedSetObjective(matrix, context)
    for selected in combinations(range(4), 2):
        local = np.array(selected, dtype=np.int64)
        added = objective.selected_indices(local)
        cross = matrix[np.ix_(context, added)]
        schur = matrix[np.ix_(added, added)] - cross.T @ np.linalg.solve(
            matrix[np.ix_(context, context)], cross,
        )
        assert objective.gain(local) == pytest.approx(stable_logdet(schur), abs=1e-12)


def test_forced_sls_never_swaps_context_and_is_monotone(matrix):
    objective = ForcedSetObjective(matrix, np.array([0, 3], dtype=np.int64))
    seen = []

    def checked(local):
        added = objective.selected_indices(local)
        assert len(added) == len(set(added)) == 2
        assert not np.intersect1d(added, objective.forced_indices).size
        seen.append(tuple(added))
        return objective(local)

    result = stochastic_local_search(4, 2, checked, np.random.default_rng(17), SearchConfig())
    assert len(seen) == result.iterations + 1
    assert np.all(np.diff([step.logdet for step in result.history]) >= 0)
    assert tuple(objective.forced_indices) == (0, 3)
    scores = [objective(np.array(s, dtype=np.int64)) for s in combinations(range(4), 2)]
    assert result.logdet <= max(scores) + 1e-12
    # Conditional fixed-size probabilities normalize over eligible additions.
    context = objective.forced_indices
    eligible = objective.eligible_indices
    cross = matrix[np.ix_(context, eligible)]
    schur = matrix[np.ix_(eligible, eligible)] - cross.T @ np.linalg.solve(
        matrix[np.ix_(context, context)], cross,
    )
    eigenvalues = np.linalg.eigvalsh(schur)
    e2 = sum(eigenvalues[i] * eigenvalues[j] for i, j in combinations(range(4), 2))
    normalizer = np.exp(objective.context_logdet) * e2
    assert np.exp(scores).sum() == pytest.approx(normalizer)
    assert (np.exp(scores) / normalizer).sum() == pytest.approx(1.0)


@pytest.mark.parametrize("context", [
    np.array([0, 0]), np.array([6]), np.array([-1]),
    np.array([0.0]), np.array([[0]]),
])
def test_invalid_forced_indices_fail(matrix, context):
    with pytest.raises(ValueError):
        ForcedSetObjective(matrix, context)


def test_singular_context_fails_instead_of_undefined_gain():
    matrix = np.ones((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Forced-set kernel.*non-finite"):
        ForcedSetObjective(matrix, np.array([0, 1], dtype=np.int64))
    # An explicitly declared jitter permits factorization.
    objective = ForcedSetObjective(matrix + 1e-6 * np.eye(3), np.array([0, 1], dtype=np.int64))
    assert np.isfinite(objective.gain(np.array([0], dtype=np.int64)))


def test_local_stage_maps_only_g1_to_public_ids_and_hides_g0():
    encoded = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3]])
    context = np.array([1, 4], dtype=np.int64)
    candidates = np.array([0, 2, 3, 5], dtype=np.int64)
    config = ExperimentConfig(
        data=DataConfig(kmer_length=2, global_collection_size=6),
        partition=PartitionConfig(num_clients=1),
        selection=SelectionConfig(client_selection_size=2, final_selection_size=2, forced_set_size=2),
    )
    original = context.copy()
    stage = _run_stage(encoded, candidates, 2, config, "numpy", np.random.default_rng(3), forced_ids=context)
    assert np.array_equal(context, original)
    assert stage["forced_set_size"] == 2 and stage["joint_selection_size"] == 4
    assert set(stage["selected_ids"]) <= set(candidates)
    assert not set(stage["selected_ids"]) & set(context)
    assert "forced_ids" not in stage and "g0_ids" not in stage
    joint = np.union1d(context, stage["selected_ids"])
    reference = stable_logdet(hamming_rbf(encoded[joint], config.kernel.gamma))
    assert stage["logdet"] == pytest.approx(reference)
    assert stage["conditional_gain"] == pytest.approx(reference - stage["context_logdet"])


def test_context_and_addition_sizes_must_fit_the_client_pool():
    with pytest.raises(ValueError, match="plus forced_set_size"):
        ExperimentConfig(selection=SelectionConfig(client_selection_size=91, forced_set_size=10))
    for size in (-1, True, 1.5):
        with pytest.raises(ValueError, match="nonnegative integer"):
            SelectionConfig(forced_set_size=size)


def test_context_sampling_is_uniform_seeded_disjoint_and_local():
    original = np.arange(100, 200, dtype=np.int64)
    context, eligible = split_context(original, 10, np.random.default_rng(42))
    again = split_context(original, 10, np.random.default_rng(42))
    assert len(context) == 10 and len(eligible) == 90
    assert not np.intersect1d(context, eligible).size
    assert np.array_equal(np.union1d(context, eligible), original)
    assert np.array_equal(context, again[0]) and np.array_equal(eligible, again[1])
    assert np.array_equal(original, np.arange(100, 200))
    empty, full = split_context(original, 0, np.random.default_rng(1))
    assert not len(empty) and np.array_equal(full, original)
    with pytest.raises(ValueError):
        split_context(original, 101, np.random.default_rng(1))


def test_public_forced_search_uses_matrix_indices_and_preserves_g0(matrix):
    context = np.array([3, 0], dtype=np.int64)
    initial = np.array([1, 4], dtype=np.int64)
    result = semi_supervised_local_search(
        matrix, context, 2, np.random.default_rng(3), SearchConfig(), initial,
    )
    assert result.initial_selection == (1, 4)
    assert np.array_equal(context, np.array([3, 0]))
    assert len(result.selected) == 2 and not set(result.selected) & set(context)
    joint = np.union1d(context, result.selected)
    assert result.logdet == pytest.approx(stable_logdet(matrix[np.ix_(joint, joint)]))
    with pytest.raises(ValueError, match="exclude G0"):
        semi_supervised_local_search(matrix, context, 2, np.random.default_rng(3), SearchConfig(), np.array([0, 1]))
    with pytest.raises(ValueError, match="selection_size"):
        semi_supervised_local_search(matrix, context, 5, np.random.default_rng(3), SearchConfig())


def test_public_forced_search_empty_context_matches_regular(matrix):
    regular = stochastic_local_search(
        len(matrix), 2, lambda s: stable_logdet(matrix[np.ix_(s, s)]),
        np.random.default_rng(7), SearchConfig(),
    )
    forced = semi_supervised_local_search(
        matrix, np.empty(0, dtype=np.int64), 2, np.random.default_rng(7), SearchConfig(),
    )
    assert regular.selected == forced.selected
    assert regular.logdet == forced.logdet and regular.history == forced.history


def test_ten_fixed_items_and_one_hundred_new_items():
    data = DataConfig(kmer_length=6, global_collection_size=120)
    encoded = sample_kmers(data, np.random.default_rng(42))
    matrix = hamming_rbf(encoded, 0.5)
    context = np.arange(10, dtype=np.int64)
    result = semi_supervised_local_search(
        matrix, context, 100, np.random.default_rng(3), SearchConfig(max_iterations=3),
    )
    assert len(result.selected) == len(set(result.selected)) == 100
    assert not set(result.selected) & set(context)
    joint = np.union1d(context, result.selected)
    assert len(joint) == 110
    assert result.logdet == pytest.approx(stable_logdet(matrix[np.ix_(joint, joint)]))


def test_client_pipeline_conditions_local_search_and_excludes_all_contexts():
    config = ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=5, final_selection_size=6, forced_set_size=2),
        search=SearchConfig(patience=5),
    )
    trial = run_trial(config, "numpy", 0)
    assert len(trial["server_pool_ids"]) == 20
    assert trial["centralized"]["candidate_count"] == 32
    assert trial["centralized"]["selection_size"] == trial["server"]["selection_size"] == 6
    alphabet_codes = {letter: index for index, letter in enumerate(config.data.alphabet)}
    encoded = np.array([[alphabet_codes[letter] for letter in kmer] for kmer in trial["global_collection"]])
    for client, (partition, stage) in enumerate(zip(trial["client_partitions"], trial["clients"])):
        context, eligible = split_context(np.array(partition), 2, _rng(config.experiment.seed, 0, 5, client))
        assert stage["forced_set_size"] == 2 and stage["joint_selection_size"] == 7
        assert stage["candidate_count"] == 8
        assert set(stage["selected_ids"]) <= set(eligible)
        joint = np.union1d(context, stage["selected_ids"])
        assert stage["logdet"] == pytest.approx(stable_logdet(hamming_rbf(encoded[joint], config.kernel.gamma)))
        for final in (trial["centralized"], trial["server"]):
            assert not set(context) & set(final["selected_ids"])
        scores = trial["trusted_evaluation_client_scores"][client]
        base = stable_logdet(hamming_rbf(encoded[context], config.kernel.gamma))
        for name, stage_name in (("centralized", "centralized"), ("federated", "server")):
            augmented = np.union1d(context, trial[stage_name]["selected_ids"])
            expected = stable_logdet(hamming_rbf(encoded[augmented], config.kernel.gamma))
            assert scores[f"{name}_joint_logdet"] == pytest.approx(expected)
            assert scores[f"{name}_conditional_gain"] == pytest.approx(expected - base)
    assert trial["centralized"]["initial_ids"] == trial["server"]["initial_ids"]
    # Private membership is absent from the returned/evaluator record schema.
    def check_keys(value):
        if isinstance(value, dict):
            assert not {"g0_ids", "forced_ids", "context_ids", "eligible_ids", "membership_mask"} & set(value)
            for child in value.values():
                check_keys(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_keys(child)
    check_keys(trial)


def test_private_context_pipeline_agrees_between_backends():
    pytest.importorskip("dppy")
    config = ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=5, final_selection_size=6, forced_set_size=2),
        search=SearchConfig(patience=5),
    )
    numpy_trial = run_trial(config, "numpy", 0)
    dppy_trial = run_trial(config, "dppy", 0)
    assert numpy_trial["trusted_evaluation_client_scores"] == dppy_trial["trusted_evaluation_client_scores"]
    for stage in ("centralized", "server"):
        assert numpy_trial[stage]["selected_ids"] == dppy_trial[stage]["selected_ids"]
        assert numpy_trial[stage]["logdet"] == dppy_trial[stage]["logdet"]
    for a, b in zip(numpy_trial["clients"], dppy_trial["clients"]):
        assert a["selected_ids"] == b["selected_ids"] and a["logdet"] == b["logdet"]
