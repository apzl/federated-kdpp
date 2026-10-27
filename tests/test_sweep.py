from dataclasses import replace
import csv
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.config import (
    DataConfig, ExperimentConfig, OutputConfig, PartitionConfig,
    SearchConfig, SelectionConfig, TrialConfig, load_config,
)
from fed_skdpp.plotting import plot_client_selection_overlap
from fed_skdpp.sweep import (
    ClientSelectionSweep, load_sweep_config, run_client_selection_sweep, write_sweep_results,
)
from fed_skdpp.sweep_cli import main


@pytest.fixture
def sweep_base():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(
            client_selection_size=5, final_selection_size=5, forced_set_size=2,
            final_objective="pooled_context",
        ),
        search=SearchConfig(patience=5), experiment=TrialConfig(seed=17, repetitions=2),
    )


def test_requested_sweep_config_preserves_global_size_and_feasibility():
    base = load_config("config.yml")
    sweep = load_sweep_config("configs/client_selection_sweep.yml")
    assert sweep.client_selection_sizes == tuple(range(10, 91, 10))
    assert base.data.global_collection_size == 1000
    assert base.client_collection_size - base.selection.forced_set_size == 90
    for size in sweep.client_selection_sizes:
        replace(base, selection=replace(base.selection, client_selection_size=size))


@pytest.mark.parametrize("values", [(), (0,), (-1,), (True,), (1.5,), (5, 5), (5, 2)])
def test_invalid_sweep_values(values):
    with pytest.raises(ValueError):
        ClientSelectionSweep(values)


def test_sweep_yaml_rejects_unknown_keys(tmp_path):
    path = tmp_path / "sweep.yml"
    path.write_text("client_selection_size: [2, 5, 8]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only client_selection_sizes"):
        load_sweep_config(path)


def test_all_values_are_validated_before_any_trials(sweep_base):
    progress = []
    with pytest.raises(ValueError, match="Infeasible client_selection_size=9"):
        run_client_selection_sweep(
            sweep_base, ClientSelectionSweep((2, 5, 9)), "numpy",
            on_trial=lambda size, row: progress.append(row),
        )
    assert not progress
    with pytest.raises(ValueError, match="server candidate pool"):
        run_client_selection_sweep(sweep_base, ClientSelectionSweep((1,)), "numpy")
    with pytest.raises(ValueError, match="backend"):
        run_client_selection_sweep(sweep_base, ClientSelectionSweep((2,)), "unknown")


def test_sweep_varies_only_client_size_and_reuses_paired_data(sweep_base):
    before = sweep_base.as_dict()
    progress, finished = [], []
    result = run_client_selection_sweep(
        sweep_base, ClientSelectionSweep((2, 5, 8)), "numpy",
        on_trial=lambda size, row: progress.append((size, row["trial"])),
        on_point=lambda point: finished.append(point["client_selection_size"]),
    )
    assert sweep_base.as_dict() == before
    assert finished == [2, 5, 8] and len(progress) == 6
    first_trials = result["points"][0]["experiment"]["trials"]
    for point in result["points"]:
        config = point["experiment"]["config"]
        assert config["selection"]["client_selection_size"] == point["client_selection_size"]
        expected = replace(sweep_base, selection=replace(sweep_base.selection, client_selection_size=point["client_selection_size"]))
        assert config == expected.as_dict()
        values = []
        for reference, trial in zip(first_trials, point["experiment"]["trials"]):
            assert reference["collection_sha256"] == trial["collection_sha256"]
            assert reference["client_partitions"] == trial["client_partitions"]
            assert reference["server_shared_context_ids"] == trial["server_shared_context_ids"]
            assert trial["centralized"]["initial_ids"] == trial["server"]["initial_ids"]
            a, b = set(trial["centralized"]["selected_ids"]), set(trial["server"]["selected_ids"])
            values.append(len(a & b))
        assert point["overlap"]["mean"] == np.mean(values)
        assert point["overlap"]["std"] == np.std(values, ddof=1)
    # With no candidate compression the final paired searches are identical.
    maximum = result["points"][-1]
    assert maximum["overlap"]["mean"] == 5 and maximum["overlap"]["std"] == 0


def test_sweep_output_is_reproducible_and_never_overwrites(tmp_path, sweep_base):
    result = run_client_selection_sweep(sweep_base, ClientSelectionSweep((2, 8)), "numpy")
    first = write_sweep_results(result, tmp_path)
    second = write_sweep_results(result, tmp_path)
    assert first != second
    parsed = json.loads((first / "results.json").read_text())
    assert parsed["sweep_config"]["client_selection_sizes"] == [2, 8]
    assert parsed["base_config"]["data"]["global_collection_size"] == 40
    with (first / "overlap_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert float(rows[-1]["mean"]) == 5
    with (first / "metrics.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 4
    assert (first / "config.yml").exists() and (first / "sweep.yml").exists()
    assert (first / "search_history.csv").exists()


def test_one_trial_has_zero_sample_dispersion_and_optional_history(sweep_base, tmp_path):
    base = replace(sweep_base, experiment=TrialConfig(repetitions=1), output=OutputConfig(save_history=False))
    result = run_client_selection_sweep(base, ClientSelectionSweep((8,)), "numpy")
    assert result["points"][0]["overlap"]["std"] == 0
    directory = write_sweep_results(result, tmp_path)
    assert not (directory / "search_history.csv").exists()


def test_plot_outputs_png_and_svg(tmp_path, sweep_base):
    pytest.importorskip("matplotlib")
    result = run_client_selection_sweep(sweep_base, ClientSelectionSweep((2, 5, 8)), "numpy")
    png, svg = plot_client_selection_overlap(result, tmp_path)
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in svg.read_text()
    assert png.stat().st_size > 1000 and svg.stat().st_size > 1000


def test_cli_runs_sweep_and_saves_plot(tmp_path, sweep_base, capsys):
    pytest.importorskip("matplotlib")
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump(sweep_base.as_dict()), encoding="utf-8")
    sweep = tmp_path / "sweep.yml"
    sweep.write_text("client_selection_sizes: [2, 8]\n", encoding="utf-8")
    assert main(["--config", str(config), "--sweep-config", str(sweep)]) == 0
    output = capsys.readouterr().out
    assert "mean nearest Hamming" in output and "Plot:" in output
    directory = next((tmp_path / "results").iterdir())
    parsed = json.loads((directory / "results.json").read_text())
    assert parsed["metadata"]["client_contexts_disclosed"] is True
    assert parsed["metadata"]["dependency_versions"]["matplotlib"]
    assert len(parsed["metadata"]["source_sha256"]) == 64
    assert (directory / "client_selection_size_hamming_distance.png").exists()
    assert (directory / "hamming_distance_summary.csv").exists()
