from dataclasses import replace
from itertools import combinations
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.budget import (
    BudgetConfig, audited_stochastic_local_search, load_budget_config,
    run_budget_experiment, write_budget_results,
)
from fed_skdpp.budget_cli import main
from fed_skdpp.config import (
    DataConfig, ExperimentConfig, KernelConfig, PartitionConfig, SearchConfig,
    SelectionConfig, TrialConfig, load_config,
)
from fed_skdpp.kernels import KernelError, hamming_rbf, stable_logdet
from fed_skdpp.objectives import ForcedSetObjective
from fed_skdpp.search import stochastic_local_search
from fed_skdpp.swap_audit import audit_one_swap_neighborhood


@pytest.fixture
def base():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=3, final_selection_size=6,
                                  forced_set_size=2, final_objective="pooled_context"),
        kernel=KernelConfig(gamma=0.25), search=SearchConfig(patience=5),
        experiment=TrialConfig(seed=42, repetitions=2),
    )


def test_requested_defaults():
    budget = load_budget_config("configs/search_budget.yml")
    assert budget == BudgetConfig(100000, 1000, 50)
    config = load_config("configs/kmer_sls.yml")
    assert config.data.kmer_length == 6 and config.data.global_collection_size == 1000
    assert config.selection.forced_set_size * config.partition.num_clients == 100
    assert config.selection.client_selection_size == 20 and config.selection.final_selection_size == 100
    assert config.experiment.repetitions == 3 and config.kernel.gamma == 0.25


@pytest.mark.parametrize("settings", [
    {"max_proposals": 0}, {"max_proposals": True}, {"checkpoint_interval": 0},
    {"checkpoint_interval": 100001}, {"reference_patience": -1},
])
def test_invalid_budget_settings(settings):
    with pytest.raises(ValueError):
        BudgetConfig(**settings)


def test_strict_budget_loader(tmp_path):
    path = tmp_path / "budget.yml"
    for text in ("[10, 100]\n", "max_steps: 100\n"):
        path.write_text(text)
        with pytest.raises(ValueError):
            load_budget_config(path)


@pytest.mark.parametrize("forced", [(), (0,), (0, 1)])
def test_audit_all_swaps_agree_with_exhaustive_direct_scores(forced):
    rng = np.random.default_rng(19)
    features = rng.normal(size=(7, 7))
    matrix = np.asarray(features @ features.T + np.eye(7) * 0.1, dtype=np.float64)
    forced = np.asarray(forced, dtype=np.int64)
    before = forced.copy()
    eligible = sorted(set(range(7)) - set(forced))
    for subset in combinations(eligible, 2):
        selected = np.asarray(subset, dtype=np.int64)
        result = audit_one_swap_neighborhood(matrix, forced, selected)
        current = stable_logdet(matrix[np.ix_(np.r_[forced, selected], np.r_[forced, selected])])
        gains = []
        for outgoing in subset:
            for incoming in set(eligible) - set(subset):
                new = sorted((set(forced) | set(subset)) - {outgoing} | {incoming})
                gains.append(stable_logdet(matrix[np.ix_(new, new)]) - current)
        assert result.total_swaps == len(gains)
        assert result.max_logdet_gain == pytest.approx(max(gains), abs=1e-10)
        assert result.directly_verified_best_gain == pytest.approx(max(gains), abs=1e-10)
        assert result.improving_swaps == sum(gain > 1e-10 for gain in gains)
        assert result.certified == (max(gains) <= 1e-10)
    assert np.array_equal(forced, before)


def test_audit_ties_duplicates_zero_quality_empty_and_full_selections():
    empty = np.empty(0, dtype=np.int64)
    assert audit_one_swap_neighborhood(np.eye(4), empty, np.array([0, 1])).certified
    duplicate = np.ones((2, 2), dtype=np.float64)
    result = audit_one_swap_neighborhood(duplicate, empty, np.array([0]))
    assert result.certified and result.max_logdet_gain == pytest.approx(0)
    zero_quality = np.diag(np.array([1., 0., 0.]))
    result = audit_one_swap_neighborhood(zero_quality, empty, np.array([0]))
    assert result.certified and result.max_logdet_gain is None
    assert result.total_swaps == 2
    for selected in (empty, np.arange(4, dtype=np.int64)):
        result = audit_one_swap_neighborhood(np.eye(4), empty, selected)
        assert result.certified and result.total_swaps == 0
    with pytest.raises(KernelError, match="nonsingular"):
        audit_one_swap_neighborhood(duplicate, np.array([0, 1]), empty)
    with pytest.raises(ValueError, match="Candidates"):
        audit_one_swap_neighborhood(np.eye(4), np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match="Candidates"):
        audit_one_swap_neighborhood(np.eye(4), empty, np.array([0]), candidate_indices=np.array([1, 2]))


def test_small_certified_run_is_exactly_bounded_by_tiny_map():
    encoded = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3]])
    matrix = hamming_rbf(encoded, 0.25)
    objective = ForcedSetObjective(matrix, np.array([0]))
    settings = SearchConfig(patience=2)
    result, audits = audited_stochastic_local_search(
        objective, 2, np.random.default_rng(7), settings, BudgetConfig(2000, 10, 2),
    )
    exact = max(objective(np.asarray(subset, dtype=np.int64)) for subset in combinations(range(5), 2))
    assert result.logdet <= exact + 1e-10
    assert audits[-1]["certified"] and result.stop_reason == "one_swap_local_optimum"
    assert result.iterations % 10 == 0
    assert np.all(np.diff([step.logdet for step in result.history]) >= 0)
    assert all(len(set(row["selected_ids"])) == 2 for row in audits)
    assert objective.forced_indices.tolist() == [0]


def test_audits_preserve_unmodified_sls_rng_acceptance_and_history():
    encoded = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [3, 3]])
    objective = ForcedSetObjective(hamming_rbf(encoded, 0.25), np.array([0]))
    settings, budget = SearchConfig(patience=2), BudgetConfig(100, 10, 2)
    actual, _ = audited_stochastic_local_search(objective, 2, np.random.default_rng(31), settings, budget)
    expected = stochastic_local_search(
        5, 2, objective, np.random.default_rng(31),
        replace(settings, patience=budget.max_proposals + 1, max_iterations=actual.iterations),
    )
    assert actual.history == expected.history and actual.selected == expected.selected
    assert actual.initial_selection == expected.initial_selection


def test_cap_is_audited_without_false_convergence_claim():
    matrix = np.diag(np.array([1., 2., 3., 4., 5., 6., 7., 8.]))
    objective = ForcedSetObjective(matrix, np.array([], dtype=np.int64))
    # Initial {0,1} cannot become the best {6,7} in one swap.
    search, audits = audited_stochastic_local_search(
        objective, 2, np.random.default_rng(1), SearchConfig(patience=1), BudgetConfig(1, 1, 1),
        initial_selection=np.array([0, 1]),
    )
    assert search.iterations == 1 and search.stop_reason == "max_iterations"
    assert not audits[-1]["certified"] and audits[-1]["improving_swaps"] > 0


def test_nonmultiple_cap_and_no_possible_swap_get_final_audit():
    matrix = np.diag(np.arange(1, 9, dtype=np.float64))
    objective = ForcedSetObjective(matrix, np.array([], dtype=np.int64))
    search, audits = audited_stochastic_local_search(
        objective, 2, np.random.default_rng(1), SearchConfig(), BudgetConfig(5, 3),
        initial_selection=np.array([0, 1]),
    )
    assert audits[-1]["iteration"] == search.iterations
    full, audits = audited_stochastic_local_search(
        objective, 8, np.random.default_rng(1), SearchConfig(), BudgetConfig(5, 3),
    )
    assert full.iterations == 0 and audits[-1]["certified"]


def test_hook_receives_copy_and_default_behavior_is_unchanged():
    seen = []
    def hook(step, selected):
        seen.append(step.iteration)
        selected[:] = -100
        return "test_stop" if step.iteration == 3 else None
    result = stochastic_local_search(6, 2, lambda _: 0., np.random.default_rng(1), SearchConfig(), on_step=hook)
    assert result.iterations == 3 and result.stop_reason == "test_stop"
    assert min(result.selected) >= 0 and seen == [1, 2, 3]
    with pytest.raises(ValueError, match="stop reason"):
        stochastic_local_search(6, 2, lambda _: 0., np.random.default_rng(1), SearchConfig(), on_step=lambda *args: False)


def test_experiment_central_runs_precede_all_clients_and_preserve_invariants(base, tmp_path):
    before, order = base.as_dict(), []
    result = run_budget_experiment(base, BudgetConfig(300, 10, 5), "numpy", tmp_path,
                                  on_stage=lambda kind, trial, client, stage: order.append((kind, trial, client)))
    assert base.as_dict() == before
    assert order[:2] == [("centralized", 0, None), ("centralized", 1, None)]
    assert all(row[0] == "client" for row in order[2:]) and len(order) == 10
    assert len({trial["collection_sha256"] for trial in result["trials"]}) == 2
    assert result["communication_rounds"] == result["transmitted_bytes"] == 0
    assert result["effective_search_config"]["patience"] == 301
    for trial in result["trials"]:
        alphabet = {char: code for code, char in enumerate("ACGT")}
        encoded = np.asarray([[alphabet[c] for c in word] for word in trial["global_collection"]])
        stages = [(trial["centralized"], trial["pooled_context_ids"])]
        stages.extend(zip(trial["clients"], trial["client_disclosed_context_ids"]))
        for stage, context in stages:
            selected = set(stage["selected_ids"])
            assert len(selected) == stage["selection_size"] and not selected & set(context)
            joint = sorted(selected | set(context))
            score = stable_logdet(hamming_rbf(encoded[joint], 0.25))
            assert stage["logdet"] == pytest.approx(score, abs=1e-10)
            with np.load(stage["trace_file"], allow_pickle=False) as trace:
                assert len(trace["logdet"]) == stage["iterations"] + 1
                assert np.all(np.diff(trace["logdet"]) >= 0)
                stalled = trace["stalled_iterations"]
                hits = np.flatnonzero(stalled >= 5)
                assert stage["reference_patience_stop_iteration"] == (int(hits[0]) if len(hits) else None)
                if len(hits):
                    assert stage["gain_after_reference_patience"] == pytest.approx(score - trace["logdet"][hits[0]])
            assert stage["certified"] == (stage["first_certified_checkpoint"] is not None)
            if not stage["certified"]:
                assert stage["iterations"] == 300 and stage["final_improving_swaps"] > 0
        for client, stage in enumerate(trial["clients"]):
            assert set(stage["selected_ids"]) <= set(trial["client_partitions"][client])
    if result["summary"]["centralized"]["capped_states"]:
        assert result["transfer"]["recommended_empirical_centralized_budget"] is None
    write_budget_results(result, tmp_path)
    assert json.loads((tmp_path / "results.json").read_text())["budget_config"]["max_proposals"] == 300
    with pytest.raises(FileExistsError):
        write_budget_results(result, tmp_path)


def test_budget_experiment_reproducibility_and_invalid_modes(base, tmp_path):
    a = run_budget_experiment(base, BudgetConfig(30, 10, 5), "numpy", tmp_path / "a")
    b = run_budget_experiment(base, BudgetConfig(30, 10, 5), "numpy", tmp_path / "b")
    for left, right in zip(a["trials"], b["trials"]):
        assert left["collection_sha256"] == right["collection_sha256"]
        for sa, sb in zip([left["centralized"], *left["clients"]], [right["centralized"], *right["clients"]]):
            assert sa["selected_ids"] == sb["selected_ids"] and sa["iterations"] == sb["iterations"]
            assert sa["logdet"] == sb["logdet"]
    for invalid in (replace(base, search=SearchConfig(algorithm="dppy_mcmc")),
                    replace(base, selection=replace(base.selection, final_objective="regular"))):
        with pytest.raises(ValueError):
            run_budget_experiment(invalid, BudgetConfig(), "numpy", tmp_path)


def test_cli_plots_and_traces(base, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("dppy")
    config, budget = tmp_path / "config.yml", tmp_path / "budget.yml"
    config.write_text(yaml.safe_dump(replace(base, experiment=TrialConfig(repetitions=1)).as_dict()))
    budget.write_text("max_proposals: 30\ncheckpoint_interval: 10\nreference_patience: 5\n")
    assert main(["--config", str(config), "--budget-config", str(budget)]) == 0
    directory = next((tmp_path / "results").iterdir())
    result = json.loads((directory / "results.json").read_text())
    assert result["backend"] == "dppy" and not result["privacy_guarantee"]
    for name in ("search_budget_logdet_curves", "search_budget_remaining_swap_gains"):
        assert (directory / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "<svg" in (directory / f"{name}.svg").read_text()
    assert (directory / "stages.csv").exists() and (directory / "checkpoints.csv").exists()
