from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.backends import KernelObjective
from fed_skdpp.cli import main
from fed_skdpp.config import (
    DataConfig, ExperimentConfig, PartitionConfig, SearchConfig, SelectionConfig, TrialConfig,
)
from fed_skdpp.experiment import _rng, run_experiment, run_trial
from fed_skdpp.kernels import hamming_rbf, stable_logdet
from fed_skdpp.search import semi_supervised_local_search


@pytest.fixture
def pooled_config():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(
            client_selection_size=5, final_selection_size=5, forced_set_size=2,
            final_objective="pooled_context",
        ),
        search=SearchConfig(patience=7), experiment=TrialConfig(seed=17, repetitions=2),
    )


def _encode(trial, alphabet):
    codes = {letter: index for index, letter in enumerate(alphabet)}
    return np.array([[codes[letter] for letter in kmer] for kmer in trial["global_collection"]])


def test_both_final_searches_use_same_fixed_pooled_context(pooled_config):
    trial = run_trial(pooled_config, "numpy", 0)
    context = np.array(trial["server_shared_context_ids"], dtype=np.int64)
    assert len(context) == len(set(context)) == 8
    assert np.array_equal(context, np.sort(np.concatenate(trial["client_disclosed_context_ids"])))
    for part, local_context, client in zip(
        trial["client_partitions"], trial["client_disclosed_context_ids"], trial["clients"],
    ):
        assert len(local_context) == 2 and set(local_context) <= set(part)
        assert not set(local_context) & set(client["selected_ids"])
    assert not set(context) & set(trial["server_pool_ids"])
    assert trial["centralized"]["initial_ids"] == trial["server"]["initial_ids"]
    assert trial["centralized"]["initial_logdet"] == pytest.approx(trial["server"]["initial_logdet"], abs=1e-12)
    encoded = _encode(trial, pooled_config.data.alphabet)
    base = stable_logdet(hamming_rbf(encoded[context], pooled_config.kernel.gamma))
    for stage_name, metric_prefix in (("centralized", "centralized"), ("server", "federated")):
        stage = trial[stage_name]
        selected = stage["selected_ids"]
        assert stage["objective"] == "joint_logdet"
        assert stage["forced_set_size"] == 8 and stage["joint_selection_size"] == 13
        assert len(selected) == len(set(selected)) == 5
        assert not set(selected) & set(context)
        joint = np.union1d(context, selected)
        expected = stable_logdet(hamming_rbf(encoded[joint], pooled_config.kernel.gamma))
        assert stage["logdet"] == pytest.approx(expected, abs=1e-12)
        assert trial["metrics"][f"{metric_prefix}_logdet"] == pytest.approx(expected)
        assert trial["metrics"][f"{metric_prefix}_conditional_gain"] == pytest.approx(expected - base)
        assert stage["stalled_iterations"] == 7
        assert np.all(np.diff([step["logdet"] for step in stage["history"]]) >= 0)
    assert trial["centralized"]["kernel_item_count"] == 40
    assert trial["server"]["kernel_item_count"] == 28
    metrics = trial["metrics"]
    assert metrics["shared_context_size"] == 8
    assert metrics["centralized_joint_selection_size"] == metrics["federated_joint_selection_size"] == 13
    assert metrics["logdet_gap_fed_minus_central"] == pytest.approx(
        metrics["federated_conditional_gain"] - metrics["centralized_conditional_gain"],
    )
    assert metrics["estimated_suggestion_upload_bytes"] == 20 * 8
    assert metrics["estimated_context_upload_bytes"] == 8 * 8
    assert metrics["estimated_client_upload_bytes"] == (20 + 8) * 8
    assert metrics["estimated_final_broadcast_bytes"] == 4 * 5 * 8
    assert metrics["estimated_total_communication_bytes"] == (20 + 8 + 20) * 8


def test_one_client_no_compression_matches_single_context_paper_search(pooled_config):
    config = replace(
        pooled_config, partition=PartitionConfig(num_clients=1),
        selection=SelectionConfig(
            client_selection_size=37, final_selection_size=5, forced_set_size=3,
            final_objective="pooled_context",
        ),
    )
    trial = run_trial(config, "numpy", 0)
    assert trial["centralized"]["selected_ids"] == trial["server"]["selected_ids"]
    assert trial["centralized"]["history"] == trial["server"]["history"]
    assert trial["metrics"]["jaccard"] == 1
    assert trial["metrics"]["logdet_gap_fed_minus_central"] == 0
    # Match the paper baseline with exactly the same index order, G0, initial
    # G1, kernel, and RNG stream, so floating-point tie decisions are paired.
    encoded = _encode(trial, config.data.alphabet)
    context = np.array(trial["server_shared_context_ids"], dtype=np.int64)
    eligible = np.setdiff1d(np.arange(len(encoded)), context)
    matrix_ids = np.concatenate((context, eligible))
    kernel = KernelObjective(encoded[matrix_ids], config.kernel, config.numerics, "numpy")
    initial = 3 + np.searchsorted(eligible, trial["centralized"]["initial_ids"])
    reference = semi_supervised_local_search(
        kernel.matrix, np.arange(3), 5, _rng(config.experiment.seed, 0, 3), config.search,
        initial_selection=initial,
    )
    assert matrix_ids[list(reference.selected)].tolist() == trial["centralized"]["selected_ids"]
    assert reference.logdet == trial["centralized"]["logdet"]


def test_empty_pooled_context_reduces_to_regular_final_objective(pooled_config):
    pooled = replace(pooled_config, selection=replace(pooled_config.selection, forced_set_size=0))
    regular = replace(pooled, selection=replace(pooled.selection, final_objective="regular"))
    a, b = run_trial(pooled, "numpy", 0), run_trial(regular, "numpy", 0)
    for stage in ("centralized", "server"):
        assert a[stage]["selected_ids"] == b[stage]["selected_ids"]
        assert a[stage]["logdet"] == b[stage]["logdet"]
    assert a["server_shared_context_ids"] == []
    assert "server_shared_context_ids" not in b


def test_pooled_backends_reproduce_selections_scores_and_contexts(pooled_config):
    pytest.importorskip("dppy")
    a, b = run_experiment(pooled_config, "numpy"), run_experiment(pooled_config, "dppy")

    def without_runtime(value):
        if isinstance(value, dict):
            return {key: without_runtime(item) for key, item in value.items() if "seconds" not in key and key != "backend"}
        if isinstance(value, (list, tuple)):
            return [without_runtime(item) for item in value]
        return value

    assert without_runtime(a) == without_runtime(b)


def test_pooled_mode_explicitly_reports_context_disclosure(pooled_config, tmp_path, capsys):
    config = replace(pooled_config, experiment=replace(pooled_config.experiment, repetitions=1))
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(config.as_dict()), encoding="utf-8")
    assert main("numpy", ["--config", str(config_path)]) == 0
    output = capsys.readouterr().out
    assert "NON-PRIVATE" in output and "all 8 context items disclosed" in output
    result = json.loads(next((tmp_path / "results").glob("*/results.json")).read_text())
    assert "NON-PRIVATE" in result["scope"]
    assert result["metadata"]["client_contexts_disclosed"] is True
    assert result["metadata"]["private_forced_sets"] is False
    assert result["metadata"]["privacy_guarantee"] is False
    assert result["metadata"]["disclosed_context_membership_logged"] is True
    assert result["forced_set_construction"]["disclosed_to_server"] is True
    assert result["forced_set_construction"]["pooled_size"] == 8
    assert result["final_objective"] == "pooled_context"


@pytest.mark.parametrize("objective", ["mean_gain", "unknown", None, ["regular"]])
def test_unknown_final_objective_is_rejected(objective):
    with pytest.raises(ValueError, match="final_objective"):
        SelectionConfig(final_objective=objective)
