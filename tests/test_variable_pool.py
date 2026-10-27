from dataclasses import replace
from itertools import combinations
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.config import DataConfig, KernelConfig, PartitionConfig, TrialConfig
from fed_skdpp.kernels import hamming_rbf, stable_logdet
from fed_skdpp.variable_dpp import (
    OrdinaryDPPSampler, exact_rejection_sweep, prefix_logdets, selected_indices, sweep_sizes,
)
from fed_skdpp.variable_pool import (
    PoolSweepConfig, VariablePoolConfig, attach_client_diagnostics,
    load_variable_pool_config, run_variable_pool_experiment,
)
from fed_skdpp.variable_pool_reporting import (
    plot_variable_pool, write_variable_pool_report, write_variable_pool_results,
)


pytest.importorskip("dppy")


def kernel():
    words = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 1], [2, 2, 2]])
    return hamming_rbf(words, .6)


def probabilities(matrix, size=None):
    subsets = [s for count in range(len(matrix) + 1) if size is None or count == size
               for s in combinations(range(len(matrix)), count)]
    weights = np.array([np.exp(stable_logdet(matrix[np.ix_(s, s)])) for s in subsets])
    return subsets, weights / weights.sum(), float(weights.sum())


def test_ordinary_dpp_normalization_frequencies_size_moments_and_native_api():
    from dppy.finite_dpps import FiniteDPP

    matrix = kernel()
    original = matrix.copy()
    subsets, probs, normalizer = probabilities(matrix)
    sampler = OrdinaryDPPSampler(matrix)
    assert sampler.diagnostics["log_normalizer"] == pytest.approx(np.log(normalizer), abs=1e-12)
    assert normalizer == pytest.approx(np.linalg.det(np.eye(4) + matrix))
    counts = dict.fromkeys(subsets, 0)
    sizes = []
    rng = np.random.default_rng(435)
    for _ in range(6000):
        ids, score, info = sampler.sample(rng)
        counts[tuple(ids)] += 1
        sizes.append(len(ids))
        assert score == stable_logdet(matrix[np.ix_(ids, ids)])
        assert info["sample_log_probability"] == pytest.approx(score - np.log(normalizer))
    assert np.allclose([counts[s] / 6000 for s in subsets], probs, atol=.024, rtol=0)
    assert np.mean(sizes) == pytest.approx(sampler.diagnostics["expected_size"], abs=.06)
    assert np.var(sizes) == pytest.approx(sampler.diagnostics["size_variance"], abs=.06)
    assert len(set(sizes)) > 2  # no fixed local sample size
    assert np.array_equal(original, matrix)
    ids, _, info = sampler.sample(np.random.default_rng(9))
    native = FiniteDPP("likelihood", L=matrix)
    expected = native.sample_exact(mode="GS", random_state=np.random.RandomState(info["random_state_seed"]))
    assert np.array_equal(ids, np.sort(expected))
    assert sampler.dpp.list_of_samples == []


def test_exact_rejection_all_sizes_against_exhaustive_weights_and_logdets():
    matrix = kernel()
    before = matrix.copy()
    expected = {size: probabilities(matrix, size) for size in range(5)}
    counts = {size: dict.fromkeys(expected[size][0], 0) for size in range(5)}
    rng = np.random.default_rng(178)
    for _ in range(6000):
        result = exact_rejection_sweep(matrix, list(range(5)), rng)
        assert result["diagnostics"]["independent_across_sizes"] is False
        assert result["diagnostics"]["inverse_probe_residual_max"] < 1e-12
        for row in result["records"]:
            selected = selected_indices(row, result["permutations"])
            size = row["selection_size"]
            assert len(selected) == len(set(selected)) == size
            counts[size][tuple(sorted(selected))] += 1
            assert row["logdet"] == pytest.approx(stable_logdet(matrix[np.ix_(selected, selected)]), abs=1e-12)
    for size, (subsets, probs, normalizer) in expected.items():
        assert sum(probs) == pytest.approx(1)
        assert normalizer > 0
        assert np.allclose([counts[size][s] / 6000 for s in subsets], probs, atol=.024, rtol=0)
    assert np.array_equal(matrix, before)


def test_constant_scale_preserves_exact_draws_and_adds_size_times_log_scale():
    first = exact_rejection_sweep(kernel(), [0, 1, 2, 3, 4], np.random.default_rng(74))
    second = exact_rejection_sweep(2 * kernel(), [0, 1, 2, 3, 4], np.random.default_rng(74))
    for a, b in zip(first["records"], second["records"]):
        assert np.array_equal(selected_indices(a, first["permutations"]), selected_indices(b, second["permutations"]))
        assert b["logdet"] - a["logdet"] == pytest.approx(a["selection_size"] * np.log(2))


def test_rejection_has_no_elementary_symmetric_polynomial_overflow():
    # Binomial(1100,550) overflows a float64 ESP table. Identity has uniform
    # exact k-DPP weights, so every rejection branch accepts on its first try.
    result = exact_rejection_sweep(np.eye(1100), [0, 10, 550, 1090, 1100], np.random.default_rng(3))
    assert result["diagnostics"]["rounds"] == {"direct": 1, "complement": 1}
    assert all(row["logdet"] == 0 for row in result["records"])


def test_empty_and_singleton_pools_and_ordinary_zero_kernel():
    for matrix in (np.zeros((3, 3)), np.empty((0, 0))):
        ids, score, _ = OrdinaryDPPSampler(matrix).sample(np.random.default_rng(1))
        assert len(ids) == 0 and score == 0
    empty = exact_rejection_sweep(np.empty((0, 0)), [0], np.random.default_rng(1))
    assert selected_indices(empty["records"][0], empty["permutations"]).size == 0
    singleton = exact_rejection_sweep(np.ones((1, 1)), [0, 1], np.random.default_rng(1))
    assert [row["logdet"] for row in singleton["records"]] == [0, 0]
    # A valid singular ordinary DPP may return one but never both duplicates.
    sampler = OrdinaryDPPSampler(np.ones((2, 2)))
    for seed in range(20):
        assert len(sampler.sample(np.random.default_rng(seed))[0]) <= 1


@pytest.mark.parametrize("matrix,sizes", [
    (np.diag([1., 2.]), [1]), (np.diag([1., 1. + 1e-14]), [1]),
    (np.ones((2, 2)), [1]), (np.array([[1., 2.], [2., 1.]]), [1]),
    (np.array([[1., .1], [.2, 1.]]), [1]), (np.array([[np.nan]]), [1]),
    (np.eye(2), [-1]), (np.eye(2), [3]), (np.eye(2), []),
    (np.eye(2), [1, 1]), (np.eye(2), [1.0]),
])
def test_rejection_invalid_inputs_and_rank_deficiency_fail_explicitly(matrix, sizes):
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        exact_rejection_sweep(matrix, sizes, np.random.default_rng(1))


def test_prefix_logdet_and_nonfinite_or_negative_spectrum():
    matrix = kernel()
    order = np.array([3, 1, 0, 2])
    scores = prefix_logdets(matrix, order, 4)
    for size, score in enumerate(scores):
        assert score == pytest.approx(stable_logdet(matrix[np.ix_(order[:size], order[:size])]))
    with pytest.raises(ValueError):
        prefix_logdets(matrix, order, 5)
    with pytest.raises(ValueError, match="nonnegative"):
        OrdinaryDPPSampler(np.diag([1., -1e-12]))
    with pytest.raises(ValueError):
        OrdinaryDPPSampler(np.diag([1., np.inf]))


@pytest.mark.parametrize("pool,expected", [(20, [10, 15, 20]), (22, [10, 15, 20, 22]),
                                           (10, [10]), (7, [7]), (0, [0])])
def test_sweep_grid_includes_full_endpoint_without_redrawing(pool, expected):
    assert sweep_sizes(pool) == expected


def test_rng_reproducibility_and_global_rng_not_modified():
    state = np.random.get_state()
    one = exact_rejection_sweep(kernel(), [1, 2, 3, 4], np.random.default_rng(5))
    two = exact_rejection_sweep(kernel(), [1, 2, 3, 4], np.random.default_rng(5))
    OrdinaryDPPSampler(kernel()).sample(np.random.default_rng(5))
    assert one["records"] == two["records"] and one["permutations"] == two["permutations"]
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


def small_config():
    return VariablePoolConfig(data=DataConfig(kmer_length=3, global_collection_size=24),
                              partition=PartitionConfig(num_clients=3), kernel=KernelConfig(gamma=.6),
                              sweep=PoolSweepConfig(minimum_selection_size=2, selection_size_interval=3),
                              experiment=TrialConfig(seed=42, repetitions=2))


def test_full_baseline_reproducibility_ownership_and_every_client_logdet():
    config = small_config()
    result = run_variable_pool_experiment(config)
    again = run_variable_pool_experiment(config)
    assert result["trials"][0]["collection_sha256"] != result["trials"][1]["collection_sha256"]
    for trial, repeated in zip(result["trials"], again["trials"]):
        assert trial["global_collection"] == repeated["global_collection"]
        assert trial["client_partitions"] == repeated["client_partitions"]
        assert trial["pool_ids"] == repeated["pool_ids"]
        assert trial["sweep"]["records"] == repeated["sweep"]["records"]
        pool = np.array(trial["pool_ids"])
        encoded = np.array([[config.data.alphabet.index(x) for x in word] for word in trial["global_collection"]])
        for client, partition in zip(trial["clients"], trial["client_partitions"]):
            ids = np.array(client["selected_ids"], dtype=int)
            assert set(ids) <= set(partition)
            assert client["logdet"] == pytest.approx(stable_logdet(hamming_rbf(encoded[ids], .6)))
        assert trial["pool_size"] == sum(client["selection_size"] for client in trial["clients"])
        assert trial["communication"]["upload_id_payload_bytes"] == 8 * len(pool)
        for row in trial["sweep"]["records"]:
            selected = pool[selected_indices(row, trial["sweep"]["permutations"])]
            assert len(selected) == len(set(selected)) == row["selection_size"]
            assert row["logdet"] == pytest.approx(stable_logdet(hamming_rbf(encoded[selected], .6)), abs=1e-12)
            for index, partition in enumerate(trial["client_partitions"]):
                contribution = np.intersect1d(selected, partition)
                assert row["client_selection_sizes"][index] == len(contribution)
                assert row["client_logdets"][index] == pytest.approx(
                    stable_logdet(hamming_rbf(encoded[contribution], .6)), abs=1e-12)


def test_empty_pool_diagnostics_and_invalid_owner():
    sweep = exact_rejection_sweep(np.empty((0, 0)), [0], np.random.default_rng(1))
    attach_client_diagnostics(np.empty((0, 0)), np.empty(0, dtype=np.int64), 3, sweep)
    assert sweep["records"][0]["client_logdets"] == [0, 0, 0]
    with pytest.raises(ValueError):
        attach_client_diagnostics(np.eye(2), np.array([0, 3]), 3, sweep)


def test_config_defaults_strictness_and_validation(tmp_path):
    path = tmp_path / "config.yml"
    path.write_text("kernel:\n  gamma: 0.6\n", encoding="utf-8")
    config = load_variable_pool_config(path)
    assert config.data.global_collection_size == 10000 and config.data.kmer_length == 20
    assert config.partition.num_clients == 10 and config.kernel.gamma == .6
    assert config.experiment.repetitions == 3 and not config.output.save_history
    actual = load_variable_pool_config("configs/variable_pool_baseline.yml")
    assert actual == config
    path.write_text("sweep:\n  selection_sze: 10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_variable_pool_config(path)
    path.write_text("clients: 10\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_variable_pool_config(path)
    with pytest.raises(ValueError):
        PoolSweepConfig(selection_size_interval=0)
    with pytest.raises(ValueError):
        replace(config, partition=PartitionConfig(num_clients=3))
    with pytest.raises(ValueError):
        replace(config, output=replace(config.output, save_history=True))


def test_artifacts_figures_and_reconstruction(tmp_path):
    pytest.importorskip("matplotlib")
    result = run_variable_pool_experiment(small_config())
    write_variable_pool_results(result, tmp_path)
    paths = plot_variable_pool(result, tmp_path)
    report = write_variable_pool_report(result, tmp_path)
    assert len(paths) == 8 and all(path.stat().st_size > 100 for path in paths)
    assert (tmp_path / "server_logdet.csv").exists() and (tmp_path / "client_logdet_by_size.csv").exists()
    assert yaml.safe_load((tmp_path / "config.yml").read_text()) == result["config"]
    persisted = json.loads((tmp_path / "results.json").read_text())
    assert persisted["method"]["server_sampler"].startswith("exact coupled rejection")
    text = report.read_text()
    assert "not DPPy's" in text and "correlated" in text and "non-private" in text
    with pytest.raises(FileExistsError):
        write_variable_pool_results(result, tmp_path)
