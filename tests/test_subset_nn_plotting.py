from copy import deepcopy
import csv
import hashlib
import json

import numpy as np
import pytest

from fed_skdpp.subset_nn_plotting import main, per_trial_subset_distance_stats, write_subset_distance_plots


def fixture_analysis():
    trials = []
    for trial in range(3):
        row = {"trial": trial}
        for method, distances in (("centralized", [1, 1, 2, 3, 3]), ("federated", [0, 0, 2, 4, 4])):
            # Different global IDs, with no global-population distance vector.
            row[method] = {"selected_ids": [2, 4, 5, 7, 9], "nearest_hamming": distances}
        trials.append(row)
    return {"metric": "within_subset_mean_nearest_hamming", "selection_size": 5,
            "data_config": {"global_collection_size": 10, "kmer_length": 4},
            "kernel_config": {"gamma": .6}, "trials": trials}


def test_per_selected_item_statistics_and_linear_percentiles_without_mutation():
    result = fixture_analysis()
    before = deepcopy(result)
    rows = per_trial_subset_distance_stats(result)
    assert len(rows) == 6 and result == before
    for row in rows:
        values = result["trials"][row["trial"]][row["method"]]["nearest_hamming"]
        assert row["count"] == 5
        assert row["mean"] == np.mean(values) and row["median"] == np.median(values)
        for key, percentile in (("p05", 5), ("p10", 10), ("q1", 25), ("q3", 75)):
            assert row[key] == np.percentile(values, percentile, method="linear")
        assert row["min"] == min(values) and row["max"] == max(values)


def test_interpolated_percentiles_are_not_rounded():
    result = fixture_analysis()
    result["trials"][0]["centralized"]["nearest_hamming"] = [0, 2, 2, 4, 4]
    row = per_trial_subset_distance_stats(result)[0]
    assert row["p05"] == .4 and row["p10"] == .8


def test_optional_modern_query_metadata_is_validated():
    result = fixture_analysis()
    before = per_trial_subset_distance_stats(result)
    for trial in result["trials"]:
        for method in ("centralized", "federated"):
            entry = trial[method]
            entry["query_population_size"] = result["selection_size"]
            entry["query_ids"] = entry["selected_ids"].copy()
    assert per_trial_subset_distance_stats(result) == before


@pytest.mark.parametrize("change", ["coverage", "size", "short", "nan", "fractional", "negative",
                                    "large", "duplicate", "id_type", "id_bounds", "queries", "count", "trials"])
def test_invalid_or_wrong_metric_analysis_is_rejected(change):
    result = fixture_analysis()
    entry = result["trials"][0]["centralized"]
    if change == "coverage":
        result["metric"] = "global_to_subset_mean_nearest_hamming"
    elif change == "size":
        result["selection_size"] = 1
    elif change == "short":
        entry["nearest_hamming"].pop()
    elif change == "nan":
        entry["nearest_hamming"][0] = float("nan")
    elif change == "fractional":
        entry["nearest_hamming"][0] = .5
    elif change == "negative":
        entry["nearest_hamming"][0] = -1
    elif change == "large":
        entry["nearest_hamming"][0] = 5
    elif change == "duplicate":
        entry["selected_ids"][0] = entry["selected_ids"][1]
    elif change == "id_type":
        entry["selected_ids"][0] = 2.5
    elif change == "id_bounds":
        entry["selected_ids"][0] = 10
    elif change == "queries":
        entry["query_ids"] = list(range(5))
    elif change == "count":
        entry["query_population_size"] = 10
    else:
        result["trials"][1]["trial"] = 0
    with pytest.raises(ValueError):
        per_trial_subset_distance_stats(result)


def test_boxes_and_percentile_markers_match_cached_distances(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    boxes, markers = [], []
    original_box, original_plot = Axes.bxp, Axes.plot
    def record_box(self, statistics, *args, **kwargs):
        boxes.append((statistics, kwargs))
        return original_box(self, statistics, *args, **kwargs)
    def record_plot(self, *args, **kwargs):
        if kwargs.get("marker") in ("v", "^"):
            markers.append((args, kwargs))
        return original_plot(self, *args, **kwargs)
    monkeypatch.setattr(Axes, "bxp", record_box)
    monkeypatch.setattr(Axes, "plot", record_plot)
    paths, rows = write_subset_distance_plots(fixture_analysis(), tmp_path)
    assert len(paths) == 8 and all(path.stat().st_size > 100 for path in paths)
    assert len(rows) == 6 and len(boxes) == 6 and len(markers) == 24
    for statistics, kwargs in boxes:
        assert kwargs["showmeans"] and not kwargs["showfliers"]
        assert [s["whislo"] for s in statistics] == [1, 0]
        assert [s["whishi"] for s in statistics] == [3, 4]
        assert [s["med"] for s in statistics] == [2, 2]
        assert [s["mean"] for s in statistics] == [2, 2]
    for offset in range(0, len(markers), 4):
        assert [call[0] for call in markers[offset:offset + 4]] == [(.9, 1), (1.1, 1), (1.9, 0), (2.1, 0)]
    with (tmp_path / "distance_stats.csv").open() as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 6 and {row["count"] for row in saved} == {"5"}
    with (tmp_path / "paired_comparison.csv").open() as handle:
        comparisons = list(csv.DictReader(handle))
    assert len(comparisons) == 3
    assert all(float(row["p05_centralized_minus_federated"]) == 1 for row in comparisons)
    assert "Self-matches are excluded" in (tmp_path / "report.md").read_text()
    with pytest.raises(FileExistsError):
        write_subset_distance_plots(fixture_analysis(), tmp_path)


@pytest.mark.parametrize("tamper", [False, True])
def test_cli_verifies_inputs_and_preserves_original_analysis(tmp_path, capsys, tamper):
    pytest.importorskip("matplotlib")
    result = fixture_analysis()
    inputs = {}
    for name in ("centralized", "federated"):
        source = tmp_path / f"{name}.json"
        source.write_text("{}", encoding="utf-8")
        inputs[f"{name}_results"] = str(source)
        inputs[f"{name}_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["inputs"] = inputs
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    before = path.read_bytes()
    if tamper:
        source.write_text("changed", encoding="utf-8")
        with pytest.raises(ValueError, match="source no longer matches"):
            main(["--analysis-results", str(path), "--output-dir", str(tmp_path / "runs")])
        assert not (tmp_path / "runs").exists()
        return
    assert main(["--analysis-results", str(path), "--output-dir", str(tmp_path / "runs")]) == 0
    directory = next((tmp_path / "runs").glob("within_subset_nn_percentiles_*"))
    metadata = json.loads((directory / "plot_metadata.json").read_text())
    assert not metadata["sampling_rerun"] and not metadata["distances_recomputed"]
    assert metadata["percentile_method"] == "linear" and metadata["percentiles"] == [5, 10]
    assert metadata["analysis_sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before and metadata["inputs"] == inputs
    assert "Combined plot:" in capsys.readouterr().out
