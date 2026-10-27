from dataclasses import replace
from itertools import combinations
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.config import (
    DataConfig, ExperimentConfig, NumericsConfig, OutputConfig, PartitionConfig,
    SearchConfig, SelectionConfig, TrialConfig, load_config,
)
from fed_skdpp.experiment import run_trial
from fed_skdpp.kernels import stable_logdet
from fed_skdpp.mcmc import conditional_l_kernel, sample_conditional_k_dpp
from fed_skdpp.pool_retention import analyze_pool_retention
from fed_skdpp.sweep import ClientSelectionSweep, run_client_selection_sweep, write_sweep_results
from fed_skdpp.sweep_cli import main


pytest.importorskip("dppy")


def settings(steps=200):
    return SearchConfig(algorithm="dppy_mcmc", chain_steps=steps, burn_in=0)


def fixture_kernel():
    return np.diag([1., 2., 4., 8.]) + 0.2 * np.ones((4, 4))


def test_schur_identity_and_exact_forced_set_normalization():
    matrix = fixture_kernel()
    before = matrix.copy()
    context = np.array([1])
    conditional, eligible, base = conditional_l_kernel(matrix, context)
    assert np.array_equal(eligible, [0, 2, 3])
    assert np.array_equal(matrix, before) and np.array_equal(context, [1])
    joint_weights, conditional_weights = [], []
    for subset in combinations(range(len(eligible)), 2):
        ids = np.array(subset)
        joint = np.r_[context, eligible[ids]]
        joint_score = stable_logdet(matrix[np.ix_(joint, joint)])
        conditional_score = stable_logdet(conditional[np.ix_(ids, ids)])
        assert joint_score == pytest.approx(base + conditional_score, abs=1e-12)
        joint_weights.append(np.exp(joint_score))
        conditional_weights.append(np.exp(conditional_score))
    joint_prob = np.asarray(joint_weights) / sum(joint_weights)
    conditional_prob = np.asarray(conditional_weights) / sum(conditional_weights)
    assert np.allclose(joint_prob, conditional_prob)
    assert conditional_prob.sum() == pytest.approx(1)


def test_native_dppy_path_matches_on_well_conditioned_fixture():
    from dppy.finite_dpps import FiniteDPP
    import dppy.mcmc_sampling as native_module

    original_helper = native_module.det_ST
    matrix = fixture_kernel()
    initial = np.array([0, 2])
    result, chain, scores, diagnostics = sample_conditional_k_dpp(
        matrix, np.array([], dtype=np.int64), 2, np.random.default_rng(43), settings(), initial,
    )
    dpp = FiniteDPP("likelihood", L=matrix)
    dpp.sample_mcmc_k_dpp(
        size=2, mode="E", s_init=initial.tolist(), nb_iter=201,
        random_state=np.random.RandomState(diagnostics["random_state_seed"]),
    )
    assert np.array_equal(chain, np.sort(np.asarray(dpp.list_of_samples[-1]), axis=1))
    assert native_module.det_ST is original_helper
    assert result.selected == tuple(chain[-1])
    assert result.logdet == scores[-1]
    assert diagnostics["sampler"] == "dppy.mcmc_sampling.basis_exchange_sampler"


def test_empirical_frequencies_match_exhaustive_conditional_k_dpp():
    matrix = fixture_kernel()
    context = np.array([0])
    conditional, eligible, _ = conditional_l_kernel(matrix, context)
    subsets = list(combinations(range(len(eligible)), 2))
    weights = np.array([np.exp(stable_logdet(conditional[np.ix_(s, s)])) for s in subsets])
    expected = weights / weights.sum()
    _, chain, _, _ = sample_conditional_k_dpp(
        matrix, context, 2, np.random.default_rng(123), settings(40000), save_history=False,
    )
    unique, counts = np.unique(chain[2001:], axis=0, return_counts=True)
    frequencies = {tuple(row): count / counts.sum() for row, count in zip(unique, counts)}
    assert np.allclose([frequencies[s] for s in subsets], expected, atol=0.03, rtol=0)


def test_terminal_state_not_map_and_downhill_moves_are_allowed():
    result, chain, scores, diagnostics = sample_conditional_k_dpp(
        fixture_kernel(), np.array([0]), 2, np.random.default_rng(123), settings(1000),
    )
    assert diagnostics["downhill_swaps"] > 0
    assert np.any(np.diff(scores) < 0)
    assert result.selected == tuple(chain[-1])
    assert result.logdet == scores[-1]
    assert not diagnostics["mixing_guarantee"]
    assert result.iterations == 1000 and len(chain) == len(scores) == len(result.history) == 1001
    assert np.all(chain[:, 1:] > chain[:, :-1])
    assert np.isfinite(scores).all()
    assert result.accepted_swaps <= diagnostics["proposed_swaps"] <= 1000


def test_logdet_adapter_handles_raw_determinant_underflow_without_repairs():
    # Every size-two raw determinant underflows, but log probabilities are valid.
    matrix = 1e-200 * np.diag([1., 2., 3., 4.])
    _, chain, scores, diagnostics = sample_conditional_k_dpp(
        matrix, np.array([], dtype=np.int64), 2, np.random.default_rng(7), settings(500),
    )
    assert np.isfinite(scores).all()
    assert len(np.unique(chain, axis=0)) > 1
    assert diagnostics["downhill_swaps"] > 0
    assert np.array_equal(matrix, 1e-200 * np.diag([1., 2., 3., 4.]))


@pytest.mark.parametrize("size", [0, 3])
def test_single_state_chain_has_full_length(size):
    result, chain, scores, diagnostics = sample_conditional_k_dpp(
        fixture_kernel(), np.array([0]), size, np.random.default_rng(8), settings(10),
    )
    assert chain.shape == (11, size) and len(scores) == 11
    assert np.array_equal(chain, np.repeat(chain[:1], 11, axis=0))
    assert result.stop_reason == "single_state_chain"
    assert result.accepted_swaps == diagnostics["proposed_swaps"] == 0


def test_rank_deficient_proposals_rejected_and_singular_context_or_init_fail():
    matrix = np.diag([1., 2., 0., 0.])
    result, chain, scores, _ = sample_conditional_k_dpp(
        matrix, np.array([], dtype=np.int64), 1, np.random.default_rng(4), settings(), np.array([0]),
    )
    assert set(chain.ravel()) <= {0, 1} and np.isfinite(scores).all()
    with pytest.raises(ValueError, match="Forced-set"):
        conditional_l_kernel(matrix, np.array([2]))
    with pytest.raises(ValueError, match="Initial selection"):
        sample_conditional_k_dpp(matrix, np.array([], dtype=np.int64), 1,
                                 np.random.default_rng(4), settings(), np.array([2]))


@pytest.mark.parametrize("context", [np.array([0, 0]), np.array([4]), np.array([0.0])])
def test_invalid_contexts_rejected(context):
    with pytest.raises(ValueError):
        conditional_l_kernel(fixture_kernel(), context)


def test_invalid_kernel_and_initial_indices_rejected():
    with pytest.raises(ValueError):
        conditional_l_kernel(np.array([[1., 2.], [2., 1.]]), np.array([], dtype=np.int64))
    for initial in (np.array([0, 0]), np.array([0, 4]), np.array([0., 1.])):
        with pytest.raises(ValueError):
            sample_conditional_k_dpp(fixture_kernel(), np.array([0]), 2,
                                     np.random.default_rng(1), settings(), initial)


def test_empty_context_is_regular_and_seed_is_reproducible():
    matrix = fixture_kernel()
    conditional, eligible, base = conditional_l_kernel(matrix, np.array([], dtype=np.int64))
    assert np.array_equal(conditional, matrix) and base == 0
    assert np.array_equal(eligible, np.arange(4))
    first = sample_conditional_k_dpp(matrix, np.array([], dtype=np.int64), 2, np.random.default_rng(2), settings())
    second = sample_conditional_k_dpp(matrix, np.array([], dtype=np.int64), 2, np.random.default_rng(2), settings())
    assert np.array_equal(first[1], second[1]) and np.array_equal(first[2], second[2])
    assert first[3] == second[3]


@pytest.fixture
def mcmc_config():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=5, final_selection_size=8,
                                  forced_set_size=2, final_objective="pooled_context"),
        search=settings(100), experiment=TrialConfig(seed=43, repetitions=2),
        output=OutputConfig(save_history=False, save_chain=True),
    )


def test_mcmc_sweep_pairs_data_and_saves_entire_global_index_chains(tmp_path, mcmc_config):
    result = run_client_selection_sweep(mcmc_config, ClientSelectionSweep((2, 5, 8)), "dppy",
                                         artifact_directory=tmp_path / "chains")
    retention = analyze_pool_retention(result)
    reference = result["points"][0]["experiment"]["trials"]
    for point in result["points"]:
        for a, trial in zip(reference, point["experiment"]["trials"]):
            assert a["collection_sha256"] == trial["collection_sha256"]
            assert a["client_partitions"] == trial["client_partitions"]
            assert a["client_disclosed_context_ids"] == trial["client_disclosed_context_ids"]
            assert trial["centralized"]["initial_ids"] == trial["server"]["initial_ids"]
            contexts = set(trial["server_shared_context_ids"])
            for stage in [trial["centralized"], trial["server"], *trial["clients"]]:
                with np.load(stage["chain_file"], allow_pickle=False) as saved:
                    chain, scores = saved["selected_ids"], saved["joint_logdet"]
                    assert chain.shape == (101, stage["selection_size"])
                    assert chain[0].tolist() == stage["initial_ids"]
                    assert chain[-1].tolist() == stage["selected_ids"]
                    assert not set(chain.ravel()) & contexts
                    assert np.all(chain[:, 1:] > chain[:, :-1])
                    assert scores[-1] == stage["logdet"]
            assert set(trial["server"]["selected_ids"]) <= set(trial["server_pool_ids"])
    assert retention["points"][-1]["summary"]["final_overlap"]["mean"] == 8
    assert retention["points"][-1]["summary"]["pool_centralized_overlap"]["mean"] == 8
    for trial in result["points"][-1]["experiment"]["trials"]:
        assert trial["metrics"]["logdet_gap_fed_minus_central"] == 0
    assert retention["points"][0]["summary"]["pool_centralized_overlap"] == retention["points"][0]["summary"]["final_overlap"]


def test_backend_and_missing_artifact_directory_rejected_before_run(mcmc_config):
    with pytest.raises(ValueError, match="backend=dppy"):
        run_trial(mcmc_config, "numpy", 0)
    with pytest.raises(ValueError, match="artifact_directory"):
        run_trial(mcmc_config, "dppy", 0)


def test_one_client_without_compression_matches_centralized_reference():
    config = ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=20),
        partition=PartitionConfig(num_clients=1),
        selection=SelectionConfig(client_selection_size=18, final_selection_size=4,
                                  forced_set_size=2, final_objective="pooled_context"),
        search=settings(), output=OutputConfig(save_history=False),
    )
    trial = run_trial(config, "dppy", 0)
    assert trial["centralized"]["selected_ids"] == trial["server"]["selected_ids"]
    assert trial["metrics"]["logdet_gap_fed_minus_central"] == 0
    assert trial["client_disclosed_context_ids"][0] == trial["server_shared_context_ids"]


def test_sampler_does_not_consume_numpy_global_rng():
    before = np.random.get_state()
    sample_conditional_k_dpp(fixture_kernel(), np.array([0]), 2, np.random.default_rng(3), settings())
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_chain_files_are_never_overwritten(tmp_path, mcmc_config):
    run_trial(mcmc_config, "dppy", 0, tmp_path)
    with pytest.raises(FileExistsError):
        run_trial(mcmc_config, "dppy", 0, tmp_path)


def test_configuration_records_requested_budget_and_preserves_scientific_setup():
    previous = load_config("configs/client_selection_repeat.yml")
    current = load_config("configs/dppy_mcmc.yml")
    assert current == replace(previous, search=current.search, output=current.output)
    assert current.search.chain_steps == 10000 and current.search.burn_in == 2000
    assert current.selection.final_selection_size == 100
    assert current.output.save_chain and not current.output.save_history
    for bad in ({"chain_steps": 0}, {"chain_steps": True}, {"burn_in": -1}, {"burn_in": True}, {"burn_in": 10000}):
        with pytest.raises(ValueError):
            SearchConfig(algorithm="dppy_mcmc", **bad)


def test_mcmc_cli_outputs_all_plots_and_preserves_result_directory(tmp_path, mcmc_config):
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump(replace(mcmc_config, experiment=TrialConfig(repetitions=1)).as_dict()))
    sweep = tmp_path / "sweep.yml"
    sweep.write_text("client_selection_sizes: [2, 5, 8]\n")
    assert main(["--config", str(config), "--sweep-config", str(sweep), "--backend", "dppy"]) == 0
    directory = next((tmp_path / "results").iterdir())
    parsed = json.loads((directory / "results.json").read_text())
    assert parsed["metadata"]["client_contexts_disclosed"]
    assert (directory / "client_selection_size_mcmc_metrics.png").exists()
    assert (directory / "mcmc_terminal_chain_trace.png").exists()
    assert list(directory.glob("dppy_pool_retention_*/client_selection_size_pool_retention.png"))
    with pytest.raises(FileExistsError):
        write_sweep_results(parsed, directory.parent, directory)
