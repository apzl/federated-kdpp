from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from fed_skdpp.nn_trial_plotting import main, per_trial_distance_stats, write_per_trial_distance_plots


def fixture_analysis():
    trials = []
    for trial in range(3):
        row = {"trial": trial}
        for method, distances in (("centralized", [0, 0, 1, 2, 3]), ("federated", [0, 0, 2, 2, 4])):
            row[method] = {"query_population_size": 5, "query_ids": list(range(5)),
                           "selected_ids": [0, 1], "nearest_hamming": distances}
        trials.append(row)
    return {"metric": "global_to_subset_mean_nearest_hamming", "selection_size": 2,
            "data_config": {"global_collection_size": 5, "kmer_length": 4},
            "kernel_config": {"gamma": .6}, "trials": trials}


def test_statistics_are_over_each_trials_global_items_not_means():
    result = fixture_analysis()
    before = deepcopy(result)
    rows = per_trial_distance_stats(result)
    assert len(rows) == 6 and result == before
    for row in rows:
        values = result["trials"][row["trial"]][row["method"]]["nearest_hamming"]
        assert row["count"] == 5 and row["min"] == 0
        assert row["mean"] == np.mean(values) and row["median"] == np.median(values)
        assert row["max"] == max(values)
        assert row["q1"] == np.percentile(values, 25) and row["q3"] == np.percentile(values, 75)


@pytest.mark.parametrize("change", ["metric", "short", "nan", "fractional", "self", "order"])
def test_invalid_or_wrong_metric_analysis_is_rejected(change):
    result = fixture_analysis()
    entry = result["trials"][0]["centralized"]
    if change == "metric":
        result["metric"] = "within_subset_mean_nearest_hamming"
    elif change == "short":
        entry["nearest_hamming"].pop()
    elif change == "nan":
        entry["nearest_hamming"][-1] = float("nan")
    elif change == "fractional":
        entry["nearest_hamming"][-1] = .5
    elif change == "self":
        entry["nearest_hamming"][0] = 1
    else:
        entry["query_ids"] = entry["query_ids"][::-1]
    with pytest.raises(ValueError):
        per_trial_distance_stats(result)


def test_plots_have_actual_min_max_whiskers_and_numeric_statistics(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.axes import Axes

    calls = []
    original = Axes.bxp
    def record(self, statistics, *args, **kwargs):
        calls.append((statistics, kwargs))
        return original(self, statistics, *args, **kwargs)
    monkeypatch.setattr(Axes, "bxp", record)
    paths, rows = write_per_trial_distance_plots(fixture_analysis(), tmp_path)
    assert len(paths) == 8 and all(path.stat().st_size > 100 for path in paths)
    assert len(rows) == 6 and len(calls) == 6  # Three combined panels plus three separate images.
    for statistics, kwargs in calls:
        assert kwargs["showmeans"] and not kwargs["showfliers"]
        assert [s["whislo"] for s in statistics] == [0, 0]
        assert [s["whishi"] for s in statistics] == [3, 4]
        assert [s["med"] for s in statistics] == [1, 2]
        assert [s["mean"] for s in statistics] == [1.2, 1.6]
    assert len((tmp_path / "distance_stats.csv").read_text().splitlines()) == 7
    assert "actual minimum and maximum" in (tmp_path / "report.md").read_text()
    with pytest.raises(FileExistsError):
        write_per_trial_distance_plots(fixture_analysis(), tmp_path)


def test_cli_reads_saved_analysis_without_changing_source(tmp_path, capsys):
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
    assert main(["--analysis-results", str(path), "--output-dir", str(tmp_path / "runs")]) == 0
    directory = next((tmp_path / "runs").glob("global_to_subset_nn_trial_boxes_*"))
    metadata = json.loads((directory / "plot_metadata.json").read_text())
    assert not metadata["sampling_rerun"] and not metadata["distances_recomputed"]
    assert path.read_bytes() == before
    assert "Combined plot:" in capsys.readouterr().out
