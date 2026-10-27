from copy import deepcopy
import csv
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from fed_skdpp.config import (
    DataConfig, ExperimentConfig, PartitionConfig, SearchConfig,
    SelectionConfig, TrialConfig, load_config,
)
from fed_skdpp.plotting import plot_pool_retention
from fed_skdpp.pool_retention import analyze_pool_retention, write_pool_retention
from fed_skdpp.pool_retention_cli import main
from fed_skdpp.sweep import ClientSelectionSweep, run_client_selection_sweep


@pytest.fixture
def saved_sweep():
    config = ExperimentConfig(
        data=DataConfig(kmer_length=2, global_collection_size=12),
        partition=PartitionConfig(num_clients=2),
        selection=SelectionConfig(
            forced_set_size=1, client_selection_size=2, final_selection_size=3,
            final_objective="pooled_context",
        ),
    )
    first = {
        "trial": 0, "server_pool_ids": [0, 1, 2, 3],
        "clients": [{"selected_ids": [0, 1]}, {"selected_ids": [2, 3]}],
        "centralized": {"selected_ids": [0, 1, 4]},
        "server": {"selected_ids": [0, 2, 3]},
        "server_shared_context_ids": [10, 11],
    }
    second = deepcopy(first)
    second["trial"] = 1
    second["centralized"]["selected_ids"] = [0, 1, 2]
    second["server"]["selected_ids"] = [0, 1, 2]
    return {
        "backend": "numpy", "base_config": config.as_dict(),
        "sweep_config": {"client_selection_sizes": [2]},
        "metadata": {"source_sha256": "original-experiment-source-hash"},
        "points": [{"client_selection_size": 2, "experiment": {"trials": [first, second]}}],
    }


def test_exact_counts_loss_decomposition_and_input_unchanged(saved_sweep):
    before = deepcopy(saved_sweep)
    result = analyze_pool_retention(saved_sweep)
    assert saved_sweep == before
    rows = result["points"][0]["trials"]
    assert rows[0]["pool_centralized_overlap"] == 2
    assert rows[0]["excluded_locally"] == 1
    assert rows[0]["final_overlap"] == 1
    assert rows[0]["available_not_selected_by_server"] == 1
    assert rows[1]["pool_centralized_overlap"] == 3
    for row in rows:
        assert row["final_overlap"] <= row["pool_centralized_overlap"] <= 3
        assert row["excluded_locally"] + row["pool_centralized_overlap"] == 3
        assert row["excluded_locally"] + row["available_not_selected_by_server"] == 3 - row["final_overlap"]
    stats = result["points"][0]["summary"]["pool_centralized_overlap"]
    assert stats["mean"] == 2.5
    assert stats["std"] == pytest.approx(np.std([2, 3], ddof=1))
    assert stats["min"] == 2 and stats["max"] == 3
    assert result["original_experiment_metadata"]["source_sha256"] == "original-experiment-source-hash"
    assert "selected_ids" not in json.dumps(result)
    assert "server_shared_context_ids" not in json.dumps(result)


@pytest.mark.parametrize("stage, ids, message", [
    ("centralized", [0, 0, 4], "unique"),
    ("centralized", [0, 1], "cardinality"),
    ("centralized", [0, 1, 12], "universe"),
    ("centralized", [0, 1, True], "integer"),
    ("server", [0, 1, 4], "subset"),
    ("centralized", [0, 1, 10], "fixed shared context"),
])
def test_invalid_selections_rejected(saved_sweep, stage, ids, message):
    saved_sweep["points"][0]["experiment"]["trials"][0][stage]["selected_ids"] = ids
    with pytest.raises(ValueError, match=message):
        analyze_pool_retention(saved_sweep)


def test_pool_must_equal_union_of_client_suggestions(saved_sweep):
    saved_sweep["points"][0]["experiment"]["trials"][0]["server_pool_ids"].append(5)
    with pytest.raises(ValueError, match="union of client suggestions"):
        analyze_pool_retention(saved_sweep)


def test_empty_input_rejected(saved_sweep):
    saved_sweep["points"][0]["experiment"]["trials"] = []
    with pytest.raises(ValueError, match="at least one trial"):
        analyze_pool_retention(saved_sweep)
    saved_sweep["points"] = []
    with pytest.raises(ValueError, match="at least one evaluated size"):
        analyze_pool_retention(saved_sweep)


def test_one_trial_has_zero_dispersion_and_optional_metadata(saved_sweep):
    saved_sweep["points"][0]["experiment"]["trials"].pop()
    saved_sweep.pop("metadata")
    result = analyze_pool_retention(saved_sweep)
    assert result["original_experiment_metadata"] == {}
    assert all(stats["std"] == 0 for stats in result["points"][0]["summary"].values())


def test_repeat_config_changes_seed_not_experimental_parameters():
    base = load_config("config.yml")
    repeat = load_config("configs/client_selection_repeat.yml")
    assert repeat.experiment.seed == 43 and repeat.experiment.repetitions == 5
    expected = replace(base, experiment=repeat.experiment, output=repeat.output)
    assert repeat == expected


def test_fresh_trials_are_paired_across_sizes_and_full_pool_keeps_every_item():
    config = ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(
            forced_set_size=2, client_selection_size=5, final_selection_size=8,
            final_objective="pooled_context",
        ),
        search=SearchConfig(patience=5), experiment=TrialConfig(seed=43, repetitions=2),
    )
    sweep = run_client_selection_sweep(config, ClientSelectionSweep((2, 5, 8)), "numpy")
    result = analyze_pool_retention(sweep)
    for row in result["points"][0]["trials"]:
        assert row["final_overlap"] == row["pool_centralized_overlap"]
    for row in result["points"][-1]["trials"]:
        assert row["pool_centralized_overlap"] == row["final_overlap"] == 8
        assert row["excluded_locally"] == row["available_not_selected_by_server"] == 0
    reference = sweep["points"][0]["experiment"]["trials"]
    assert reference[0]["collection_sha256"] != reference[1]["collection_sha256"]
    assert reference[0]["client_partitions"] != reference[1]["client_partitions"]
    assert reference[0]["client_disclosed_context_ids"] != reference[1]["client_disclosed_context_ids"]
    for point in sweep["points"]:
        for a, b in zip(reference, point["experiment"]["trials"]):
            assert a["collection_sha256"] == b["collection_sha256"]
            assert a["client_partitions"] == b["client_partitions"]
            assert a["client_disclosed_context_ids"] == b["client_disclosed_context_ids"]


def test_analysis_files_never_overwrite_and_plot_valid(tmp_path, saved_sweep):
    pytest.importorskip("matplotlib")
    result = analyze_pool_retention(saved_sweep)
    first = write_pool_retention(result, tmp_path)
    second = write_pool_retention(result, tmp_path)
    assert first != second
    assert json.loads((first / "analysis.json").read_text()) == result
    with (first / "pool_retention_trials.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and int(rows[0]["excluded_locally"]) == 1
    with (first / "pool_retention_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["pool_centralized_overlap_mean"]) == 2.5
    png, svg = plot_pool_retention(result, first)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in svg.read_text()


def test_cli_preserves_saved_results_and_original_provenance(tmp_path, saved_sweep, capsys):
    pytest.importorskip("matplotlib")
    source = tmp_path / "results.json"
    source.write_text(json.dumps(saved_sweep), encoding="utf-8")
    before = source.read_bytes()
    assert main(["--results", str(source), "--output-dir", str(tmp_path / "analysis")]) == 0
    assert source.read_bytes() == before
    assert "excluded locally=0.5" in capsys.readouterr().out
    target = next((tmp_path / "analysis").iterdir())
    result = json.loads((target / "analysis.json").read_text())
    assert result["source_results"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert result["source_results"]["path"] == str(source)
    assert result["original_experiment_metadata"]["source_sha256"] == "original-experiment-source-hash"
    assert len(result["analysis_metadata"]["source_sha256"]) == 64
    assert (target / "client_selection_size_pool_retention.png").exists()
