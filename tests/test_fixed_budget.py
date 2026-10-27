from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.cli import main
from fed_skdpp.config import (
    DataConfig, ExperimentConfig, OutputConfig, PartitionConfig, SearchConfig,
    SelectionConfig, TrialConfig, load_config,
)
from fed_skdpp.experiment import run_trial
from fed_skdpp.kernels import hamming_rbf, stable_logdet
from fed_skdpp.search import semi_supervised_local_search, stochastic_local_search


def test_fixed_budget_ignores_patience_counts_attempts_and_accepts_ties():
    result = stochastic_local_search(
        10, 3, lambda _: 0., np.random.default_rng(9), SearchConfig(patience=2),
        proposal_budget=137,
    )
    assert result.iterations == result.accepted_swaps == result.stalled_iterations == 137
    assert result.stop_reason == "proposal_budget" and len(result.history) == 138
    assert result.strict_improvements == 0 and len(set(result.selected)) == 3


def test_fixed_budget_preserves_original_proposals_rng_and_acceptance():
    encoded = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3]])
    matrix = hamming_rbf(encoded, 0.5)
    objective = lambda ids: stable_logdet(matrix[np.ix_(ids, ids)])
    settings = SearchConfig(patience=7)
    fixed = stochastic_local_search(6, 3, objective, np.random.default_rng(3), settings, proposal_budget=200)
    original = stochastic_local_search(
        6, 3, objective, np.random.default_rng(3), replace(settings, patience=201, max_iterations=200),
    )
    assert fixed.history == original.history and fixed.selected == original.selected
    assert fixed.iterations == original.iterations == 200
    assert np.all(np.diff([step.logdet for step in fixed.history]) >= 0)
    assert fixed.stop_reason == "proposal_budget" and original.stop_reason == "max_iterations"


@pytest.mark.parametrize("size", [0, 3])
def test_no_possible_swap_reports_zero_attempts(size):
    result = stochastic_local_search(3, size, lambda _: 0., np.random.default_rng(1),
                                     SearchConfig(), proposal_budget=100)
    assert result.iterations == 0 and result.stop_reason == "no_possible_swap"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_fixed_budgets(value):
    with pytest.raises(ValueError, match="positive integer"):
        SearchConfig(local_proposal_budget=value)
    with pytest.raises(ValueError, match="positive integer"):
        stochastic_local_search(5, 2, lambda _: 0., np.random.default_rng(1),
                                 SearchConfig(), proposal_budget=value)


def test_incompatible_safety_cap_and_sampling_algorithms_rejected():
    with pytest.raises(ValueError, match="safety cap"):
        SearchConfig(max_iterations=10, global_proposal_budget=11)
    with pytest.raises(ValueError, match="safety cap"):
        stochastic_local_search(5, 2, lambda _: 0., np.random.default_rng(1),
                                 SearchConfig(max_iterations=10), proposal_budget=11)
    for algorithm in ("dppy_mcmc", "dppy_exact"):
        with pytest.raises(ValueError, match="only to stochastic_local_search"):
            SearchConfig(algorithm=algorithm, local_proposal_budget=5)


def test_direct_forced_set_wrapper_keeps_context_immutable():
    matrix = np.diag(np.arange(1., 7.)) + 0.2
    context = np.array([1, 4])
    before = matrix.copy()
    result = semi_supervised_local_search(
        matrix, context, 2, np.random.default_rng(8), SearchConfig(patience=1), proposal_budget=43,
    )
    assert result.iterations == 43 and not set(result.selected) & set(context)
    assert np.array_equal(matrix, before) and np.array_equal(context, [1, 4])
    assert result.logdet >= result.initial_logdet


@pytest.fixture
def fixed_config():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=5, final_selection_size=8,
                                  forced_set_size=2, final_objective="pooled_context"),
        search=SearchConfig(patience=2, local_proposal_budget=11, global_proposal_budget=31),
        experiment=TrialConfig(seed=42, repetitions=2), output=OutputConfig(save_history=False),
    )


def test_stages_use_correct_distinct_budgets_and_preserve_pairing(fixed_config):
    trial = run_trial(fixed_config, "numpy", 0)
    prior = run_trial(replace(fixed_config, search=SearchConfig(patience=2)), "numpy", 0)
    for name in ("collection_sha256", "client_partitions", "client_disclosed_context_ids"):
        assert trial[name] == prior[name]
    for stage in trial["clients"]:
        assert stage["iterations"] == stage["configured_proposal_budget"] == 11
        assert stage["stopping_rule"] == "fixed_proposal_budget" and stage["stop_reason"] == "proposal_budget"
        assert len(set(stage["selected_ids"])) == 5
    for name in ("centralized", "server"):
        stage = trial[name]
        assert stage["iterations"] == stage["configured_proposal_budget"] == 31
        assert stage["stop_reason"] == "proposal_budget"
        assert not set(stage["selected_ids"]) & set(trial["server_shared_context_ids"])
    assert trial["centralized"]["initial_ids"] == trial["server"]["initial_ids"]
    assert trial["metrics"]["client_proposals"] == 44
    assert trial["metrics"]["centralized_proposals"] == trial["metrics"]["server_proposals"] == 31
    assert trial["metrics"]["federated_total_proposals"] == 75
    repeated = run_trial(fixed_config, "numpy", 0)
    assert trial["centralized"]["selected_ids"] == repeated["centralized"]["selected_ids"]
    assert trial["server"]["selected_ids"] == repeated["server"]["selected_ids"]


def test_fixed_one_client_full_pool_matches_central_reference(fixed_config):
    config = replace(fixed_config, partition=PartitionConfig(num_clients=1),
                     selection=SelectionConfig(client_selection_size=38, final_selection_size=8,
                                               forced_set_size=2, final_objective="pooled_context"))
    trial = run_trial(config, "numpy", 0)
    assert trial["clients"][0]["iterations"] == 0
    assert trial["centralized"]["selected_ids"] == trial["server"]["selected_ids"]
    assert trial["metrics"]["jaccard"] == 1


def test_config_changes_only_budget_and_history_output():
    current, prior = load_config("configs/sls_fixed_budget.yml"), load_config("configs/kmer_sls.yml")
    assert current == replace(prior, search=replace(prior.search, local_proposal_budget=10000,
                                                   global_proposal_budget=100000),
                              output=replace(prior.output, save_history=False))


def test_cli_saves_distinct_budget_metadata_and_metrics(tmp_path, fixed_config):
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(fixed_config.as_dict()))
    assert main("numpy", ["--config", str(config_path), "--output-dir", str(tmp_path / "runs")]) == 0
    path = next((tmp_path / "runs").glob("numpy_sls_fixed_budget_*/results.json"))
    saved = json.loads(path.read_text())
    assert saved["config"]["search"]["local_proposal_budget"] == 11
    assert saved["config"]["search"]["global_proposal_budget"] == 31
    assert saved["trials"][0]["centralized"]["iterations"] == 31
    assert saved["summary"]["intersection_size"]["mean"] >= 0
    assert (path.parent / "sls_fixed_budget_comparison.png").stat().st_size > 1000
    text = (path.parent / "sls_fixed_budget_comparison.svg").read_text()
    assert "fixed local and global proposal budgets" in text
    assert "Local: 11 proposals" in text and "central/server: 31 proposals" in text
    assert "exact k-DPP sampling" not in text
