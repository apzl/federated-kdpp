from dataclasses import replace
from itertools import combinations
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.cli import main
from fed_skdpp.config import (
    DataConfig, ExperimentConfig, NumericsConfig, OutputConfig, PartitionConfig, SearchConfig,
    SelectionConfig, TrialConfig, load_config,
)
from fed_skdpp.exact import sample_conditional_exact_k_dpp
from fed_skdpp.experiment import _rng, run_experiment, run_trial
from fed_skdpp.kernels import hamming_rbf, stable_logdet
from fed_skdpp.mcmc import conditional_l_kernel


pytest.importorskip("dppy")


def fixture_kernel():
    features = np.array([[1., 0.2], [0.3, 1.1], [1.2, 0.8], [-0.4, 0.6], [0.5, -0.7]])
    return features @ features.T + np.diag([0.5, 1., 1.5, 2., 0.8])


def test_exact_conditional_frequencies_and_esp_normalization_against_enumeration():
    matrix = fixture_kernel()
    context = np.array([2])
    before = matrix.copy()
    eligible = np.array([0, 1, 3, 4])
    subsets = list(combinations(range(4), 2))
    joint_scores = []
    for subset in subsets:
        joint = np.r_[context, eligible[list(subset)]]
        joint_scores.append(stable_logdet(matrix[np.ix_(joint, joint)]))
    joint_scores = np.array(joint_scores)
    weights = np.exp(joint_scores)
    probabilities = weights / weights.sum()
    counts = dict.fromkeys(subsets, 0)
    rng = np.random.default_rng(31415)
    for _ in range(6000):
        selected, score, info = sample_conditional_exact_k_dpp(matrix, context, 2, rng)
        counts[tuple(selected)] += 1
        index = subsets.index(tuple(selected))
        assert score == pytest.approx(joint_scores[index], abs=1e-12)
        assert info["sample_log_probability"] == pytest.approx(np.log(probabilities[index]), abs=1e-12)
    assert info["conditional_log_normalizer"] == pytest.approx(
        np.log(weights.sum()) - stable_logdet(matrix[np.ix_(context, context)]), abs=1e-12,
    )
    assert sum(probabilities) == pytest.approx(1)
    assert np.allclose([counts[s] / 6000 for s in subsets], probabilities, atol=0.025, rtol=0)
    assert np.array_equal(matrix, before) and np.array_equal(context, [2])


@pytest.mark.parametrize("mode", ["KuTa12", "GS", "GS_bis"])
def test_matches_unmodified_native_dppy_and_is_reproducible(mode):
    from dppy.finite_dpps import FiniteDPP
    matrix = fixture_kernel()
    context = np.array([1])
    first = sample_conditional_exact_k_dpp(matrix, context, 2, np.random.default_rng(42), mode)
    second = sample_conditional_exact_k_dpp(matrix, context, 2, np.random.default_rng(42), mode)
    conditional, _, _ = conditional_l_kernel(matrix, context)
    native = FiniteDPP("likelihood", L=conditional)
    selected = native.sample_exact_k_dpp(
        size=2, mode=mode, random_state=np.random.RandomState(first[2]["random_state_seed"]),
    )
    assert np.array_equal(first[0], np.sort(selected))
    assert np.array_equal(first[0], second[0]) and first[1] == second[1]
    assert first[2]["native_sampler_calls"] == 1


@pytest.mark.parametrize("size", [0, 4])
def test_empty_and_full_eligible_selection(size):
    selected, score, info = sample_conditional_exact_k_dpp(
        fixture_kernel(), np.array([2]), size, np.random.default_rng(1),
    )
    assert np.array_equal(selected, np.arange(size))
    assert info["sample_log_probability"] == pytest.approx(0, abs=1e-12)
    assert np.isfinite(score) and info["sample_count"] == 1


def test_empty_universe_and_empty_context_regular_distribution():
    selected, score, _ = sample_conditional_exact_k_dpp(
        np.empty((0, 0)), np.array([], dtype=np.int64), 0, np.random.default_rng(1),
    )
    assert len(selected) == 0 and score == 0
    matrix = fixture_kernel()
    selected, score, info = sample_conditional_exact_k_dpp(
        matrix, np.array([], dtype=np.int64), 3, np.random.default_rng(1),
    )
    assert info["context_logdet"] == 0
    assert score == info["conditional_logdet"] == stable_logdet(matrix[np.ix_(selected, selected)])


def test_rank_deficient_zero_quality_and_duplicate_items():
    matrix = np.diag([1., 2., 0., 0.])
    empty = np.array([], dtype=np.int64)
    selected, _, _ = sample_conditional_exact_k_dpp(matrix, empty, 2, np.random.default_rng(1))
    assert np.array_equal(selected, [0, 1])
    with pytest.raises(ValueError, match="numerical rank"):
        sample_conditional_exact_k_dpp(matrix, empty, 3, np.random.default_rng(1))
    with pytest.raises(ValueError, match="Forced-set"):
        sample_conditional_exact_k_dpp(matrix, np.array([2]), 1, np.random.default_rng(1))
    duplicate = np.array([[1., 1., 0.], [1., 1., 0.], [0., 0., 2.]])
    for seed in range(20):
        selected, _, _ = sample_conditional_exact_k_dpp(duplicate, empty, 2, np.random.default_rng(seed))
        assert 2 in selected and len(set(selected)) == 2


def test_nearly_identical_candidates_and_negative_eigenvalues_are_explicit():
    matrix = np.ones((4, 4)) + 1e-7 * np.eye(4)
    with pytest.raises(ValueError, match="joint logdet disagree"):
        sample_conditional_exact_k_dpp(matrix, np.array([0]), 2, np.random.default_rng(1))
    # This ill-conditioned fixture needs an explicitly looser identity check.
    selected, score, info = sample_conditional_exact_k_dpp(
        matrix, np.array([0]), 2, np.random.default_rng(1),
        numerics=NumericsConfig(psd_tolerance=1e-7),
    )
    assert len(selected) == 2 and np.isfinite(score)
    assert info["min_eigenvalue"] > 0
    with pytest.raises(ValueError, match="negative"):
        sample_conditional_exact_k_dpp(
            np.diag([1., -1e-12]), np.array([], dtype=np.int64), 1, np.random.default_rng(1),
        )


@pytest.mark.parametrize("matrix,context,size", [
    (np.array([[1., 2.], [2., 1.]]), np.array([], dtype=np.int64), 1),
    (np.array([[1., 0.], [0.1, 1.]]), np.array([], dtype=np.int64), 1),
    (np.diag([1., np.nan]), np.array([], dtype=np.int64), 1),
    (np.eye(3), np.array([0, 0]), 1),
    (np.eye(3), np.array([3]), 1),
    (np.eye(3), np.array([0.]), 1),
    (np.eye(3), np.array([0]), 3),
    (np.eye(3), np.array([0]), -1),
])
def test_invalid_inputs_fail(matrix, context, size):
    with pytest.raises(ValueError):
        sample_conditional_exact_k_dpp(matrix, context, size, np.random.default_rng(1))


def test_global_numpy_rng_is_untouched():
    before = np.random.get_state()
    sample_conditional_exact_k_dpp(fixture_kernel(), np.array([0]), 2, np.random.default_rng(9))
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.fixture
def exact_config():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=5, final_selection_size=8,
                                  forced_set_size=2, final_objective="pooled_context"),
        search=SearchConfig(algorithm="dppy_exact"), experiment=TrialConfig(seed=42, repetitions=2),
        output=OutputConfig(save_history=False),
    )


def test_experiment_uses_native_exact_at_all_stages_and_preserves_contexts(exact_config, monkeypatch):
    from dppy.finite_dpps import FiniteDPP
    original = FiniteDPP.sample_exact_k_dpp
    calls = []

    def record(self, size, mode, **kwargs):
        calls.append((size, mode))
        return original(self, size, mode, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("SLS/MCMC must not be called by the exact path")

    monkeypatch.setattr(FiniteDPP, "sample_exact_k_dpp", record)
    monkeypatch.setattr("fed_skdpp.experiment.stochastic_local_search", forbidden)
    monkeypatch.setattr("fed_skdpp.experiment.sample_conditional_k_dpp", forbidden)
    result = run_experiment(exact_config, "dppy")
    assert len(calls) == 12 and all(mode == "KuTa12" for _, mode in calls)
    for trial in result["trials"]:
        contexts = set(trial["server_shared_context_ids"])
        encoded = np.array([[exact_config.data.alphabet.index(x) for x in word]
                            for word in trial["global_collection"]])
        for stage in [trial["centralized"], trial["server"], *trial["clients"]]:
            selected = stage["selected_ids"]
            assert len(set(selected)) == stage["selection_size"]
            assert not set(selected) & contexts
            assert stage["sample_count"] == 1 and stage["iterations"] == 0
            assert stage["initial_ids"] is None and stage["initial_logdet"] is None
            assert stage["history"] == []
            assert stage["sample_log_probability"] <= 1e-12
        assert set(trial["server"]["selected_ids"]) <= set(trial["server_pool_ids"])
        assert trial["centralized"]["random_state_seed"] != trial["server"]["random_state_seed"]
        for name in ("centralized", "server"):
            ids = np.array(sorted(contexts | set(trial[name]["selected_ids"])))
            score = stable_logdet(hamming_rbf(encoded[ids], exact_config.kernel.gamma))
            assert score == pytest.approx(trial[name]["logdet"], abs=1e-12)
        a, b = set(trial["centralized"]["selected_ids"]), set(trial["server"]["selected_ids"])
        assert trial["metrics"]["intersection_size"] == len(a & b)
        assert trial["metrics"]["jaccard"] == len(a & b) / len(a | b)


def test_data_pairing_with_sls_and_seed_reproducibility(exact_config):
    exact = run_trial(exact_config, "dppy", 0)
    again = run_trial(exact_config, "dppy", 0)
    sls = run_trial(replace(exact_config, search=SearchConfig(patience=5)), "numpy", 0)
    for field in ("collection_sha256", "client_partitions", "client_disclosed_context_ids"):
        assert exact[field] == sls[field]
    for stage in ("centralized", "server"):
        assert exact[stage]["selected_ids"] == again[stage]["selected_ids"]
        assert exact[stage]["logdet"] == again[stage]["logdet"]


def test_central_sample_independent_of_client_suggestion_budget(exact_config):
    first = run_trial(exact_config, "dppy", 0)
    second = run_trial(replace(exact_config, selection=replace(
        exact_config.selection, client_selection_size=8)), "dppy", 0)
    assert first["centralized"]["selected_ids"] == second["centralized"]["selected_ids"]


def test_one_client_no_compression_has_same_conditional_distribution():
    config = ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=20),
        partition=PartitionConfig(num_clients=1),
        selection=SelectionConfig(client_selection_size=18, final_selection_size=4,
                                  forced_set_size=2, final_objective="pooled_context"),
        search=SearchConfig(algorithm="dppy_exact"), output=OutputConfig(save_history=False),
    )
    trial = run_trial(config, "dppy", 0)
    for key in ("context_logdet", "conditional_log_normalizer", "min_eigenvalue", "max_eigenvalue"):
        assert trial["centralized"][key] == pytest.approx(trial["server"][key], abs=1e-12)
    # Independent draws are intentional. Reusing central RNG on the same
    # conditional distribution must exactly recover the central sample.
    context = np.array(trial["server_shared_context_ids"])
    eligible = np.array(trial["server_pool_ids"])
    words = trial["global_collection"]
    encoded = np.array([[config.data.alphabet.index(x) for x in words[i]] for i in np.r_[context, eligible]])
    selected, score, _ = sample_conditional_exact_k_dpp(
        hamming_rbf(encoded, config.kernel.gamma), np.arange(2), 4, _rng(config.experiment.seed, 0, 3),
    )
    assert eligible[selected].tolist() == trial["centralized"]["selected_ids"]
    assert score == pytest.approx(trial["centralized"]["logdet"])


def test_configuration_and_backend_rejections(exact_config):
    with pytest.raises(ValueError, match="backend=dppy"):
        run_trial(exact_config, "numpy", 0)
    with pytest.raises(ValueError, match="save_chain"):
        replace(exact_config, output=OutputConfig(save_chain=True))
    with pytest.raises(ValueError, match="exact_mode"):
        SearchConfig(exact_mode="invalid")
    current, prior = load_config("configs/dppy_exact.yml"), load_config("configs/kmer_sls.yml")
    assert current == replace(prior, search=current.search, output=current.output)


def test_cli_serializes_single_samples_and_does_not_overwrite(tmp_path, exact_config):
    config_file = tmp_path / "exact.yml"
    config_file.write_text(yaml.safe_dump(exact_config.as_dict()))
    for _ in range(2):
        assert main("dppy", ["--config", str(config_file), "--output-dir", str(tmp_path / "runs")]) == 0
    files = list((tmp_path / "runs").glob("dppy_exact_*/results.json"))
    assert len(files) == 2
    for path in files:
        saved = json.loads(path.read_text())
        assert saved["config"]["search"]["algorithm"] == "dppy_exact"
        assert saved["trials"][0]["centralized"]["sample_count"] == 1
        assert "intersection_size" in saved["summary"]
        assert not (path.parent / "search_history.csv").exists()
        assert (path.parent / "exact_comparison.png").stat().st_size > 1000
        assert (path.parent / "exact_comparison.svg").stat().st_size > 1000


def test_client_size_sweep_supports_exact_sampling_without_false_sls_labels(tmp_path, exact_config):
    from fed_skdpp.sweep_cli import main as sweep_main
    config_path, sweep_path = tmp_path / "exact.yml", tmp_path / "sweep.yml"
    config_path.write_text(yaml.safe_dump(exact_config.as_dict()))
    sweep_path.write_text(yaml.safe_dump({"client_selection_sizes": [2, 5, 8]}))
    assert sweep_main(["--config", str(config_path), "--sweep-config", str(sweep_path),
                       "--backend", "dppy", "--output-dir", str(tmp_path / "runs")]) == 0
    path = next((tmp_path / "runs").glob("dppy_exact_*/results.json"))
    result = json.loads(path.read_text())
    assert "no initialization" in result["initialization_caveat"]
    first = result["points"][0]["experiment"]["trials"]
    for point in result["points"]:
        for reference, trial in zip(first, point["experiment"]["trials"]):
            assert trial["centralized"]["selected_ids"] == reference["centralized"]["selected_ids"]
    svg = (path.parent / "client_selection_size_overlap.svg").read_text()
    assert "DPPy exact" in svg and "stalled proposals" not in svg
    assert (path.parent / "client_selection_size_exact_metrics.png").stat().st_size > 1000
