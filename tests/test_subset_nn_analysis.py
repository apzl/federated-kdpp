import hashlib
import json

import numpy as np
import pytest

from fed_skdpp.centralized_pool import (
    CentralizedPoolConfig, CentralizedSizeSweep, SavedPoolSource, run_centralized_pool_experiment,
)
from fed_skdpp.config import DataConfig, PartitionConfig, TrialConfig
from fed_skdpp.subset_nn_analysis import (
    analyze_saved_subsets, global_to_subset_nearest_hamming,
    within_subset_nearest_hamming, write_subset_nn_analysis,
)
from fed_skdpp.variable_pool import PoolSweepConfig, VariablePoolConfig, run_variable_pool_experiment


def test_within_selection_excludes_self_and_keeps_distinct_duplicate_rows():
    assert np.array_equal(within_subset_nearest_hamming([[0, 0, 0], [0, 0, 1], [1, 1, 1]]), [1, 1, 2])
    assert np.array_equal(within_subset_nearest_hamming([[0, 0], [0, 0], [1, 1]]), [0, 0, 2])
    assert np.array_equal(within_subset_nearest_hamming([["A", "C"], ["G", "T"]]), [2, 2])


@pytest.mark.parametrize("values", [[], [[0, 1]], [[], []], [0, 1],
                                     [[0., np.nan], [0., 1.]], [[1j], [2j]]])
def test_invalid_nearest_neighbor_inputs_fail(values):
    with pytest.raises(ValueError):
        within_subset_nearest_hamming(values)


def test_distance_permutation_invariance_and_independent_reference():
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(517)
    words = rng.integers(4, size=(50, 20))
    distances = np.rint(cdist(words, words, metric="hamming") * 20)
    np.fill_diagonal(distances, np.inf)
    nearest = within_subset_nearest_hamming(words)
    assert np.array_equal(nearest, distances.min(axis=1))
    perm = rng.permutation(len(words))
    assert np.array_equal(within_subset_nearest_hamming(words[perm]), nearest[perm])


def test_global_to_selected_coverage_direction_includes_exact_matches():
    full = np.array([[0, 0, 0], [0, 0, 1], [1, 1, 1]])
    selected = full[[0]]
    assert np.array_equal(global_to_subset_nearest_hamming(full, selected, batch_size=1), [0, 1, 3])
    assert np.array_equal(global_to_subset_nearest_hamming(selected, full), [0])
    # The same input on both sides means coverage zero, not an exclusion rule.
    assert np.array_equal(global_to_subset_nearest_hamming(full, full), [0, 0, 0])
    assert np.array_equal(global_to_subset_nearest_hamming(full, full[[0, 2]], batch_size=2), [0, 1, 0])


@pytest.mark.parametrize("full,selected,batch", [
    ([], [[0]], 1), ([[0]], [], 1), ([[0]], [[0, 1]], 1),
    ([[np.nan]], [[0]], 1), ([[0]], [[0]], 0),
])
def test_invalid_global_to_selected_inputs_fail(full, selected, batch):
    with pytest.raises(ValueError):
        global_to_subset_nearest_hamming(full, selected, batch)


@pytest.fixture
def saved_results(tmp_path):
    pytest.importorskip("dppy")
    config = VariablePoolConfig(data=DataConfig(kmer_length=3, global_collection_size=24),
                                partition=PartitionConfig(num_clients=3),
                                sweep=PoolSweepConfig(minimum_selection_size=2, selection_size_interval=2),
                                experiment=TrialConfig(seed=42, repetitions=3))
    distributed = run_variable_pool_experiment(config)
    source = tmp_path / "distributed.json"
    source.write_text(json.dumps(distributed), encoding="utf-8")
    centralized = run_centralized_pool_experiment(CentralizedPoolConfig(
        source=SavedPoolSource(str(source)), sweep=CentralizedSizeSweep(2, 8, 2)))
    path = tmp_path / "centralized.json"
    path.write_text(json.dumps(centralized), encoding="utf-8")
    return path, source


def test_saved_analysis_preserves_source_data_and_reports_trial_means(saved_results):
    path, source = saved_results
    before = [p.read_bytes() for p in (path, source)]
    result = analyze_saved_subsets(path, 4)
    assert result["resampling"] is False
    assert result["selection_size"] == 4 and len(result["trials"]) == 3
    assert result["inputs"]["centralized_sha256"] == hashlib.sha256(before[0]).hexdigest()
    for method in ("centralized", "federated"):
        values = []
        for trial in result["trials"]:
            row = trial[method]
            assert len(row["selected_ids"]) == len(set(row["selected_ids"])) == 4
            assert len(row["nearest_hamming"]) == 4 and row["independent_distance_check_passed"]
            assert row["mean_nearest_hamming"] == np.mean(row["nearest_hamming"])
            values.append(row["mean_nearest_hamming"])
        assert result["summary"][method]["mean"] == np.mean(values)
        assert result["summary"][method]["std"] == np.std(values, ddof=1)
        assert result["summary"][method]["num_trials"] == 3
    assert before == [p.read_bytes() for p in (path, source)]


def test_no_silent_trial_dropping_and_no_modified_source(saved_results):
    path, source = saved_results
    with pytest.raises(ValueError, match="no unique"):
        analyze_saved_subsets(path, 1000)
    with pytest.raises(ValueError, match="at least two"):
        analyze_saved_subsets(path, 1)
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        analyze_saved_subsets(path, 4)


def test_boxplot_artifacts_contain_three_averages_not_pseudoreplicates(saved_results, tmp_path):
    pytest.importorskip("matplotlib")
    path, _ = saved_results
    result = analyze_saved_subsets(path, 4)
    directory = tmp_path / "artifacts"
    directory.mkdir()
    plots = write_subset_nn_analysis(result, directory)
    assert len(plots) == 2 and all(p.stat().st_size > 100 for p in plots)
    assert len((directory / "trial_means.csv").read_text().splitlines()) == 4
    assert len((directory / "per_item_distances.csv").read_text().splitlines()) == 25
    assert "one average per data trial" in (directory / "report.md").read_text()
    with pytest.raises(FileExistsError):
        write_subset_nn_analysis(result, directory)


def test_global_coverage_averages_over_all_global_items_and_writes_every_item(saved_results, tmp_path):
    pytest.importorskip("matplotlib")
    from scipy.spatial.distance import cdist

    path, _ = saved_results
    result = analyze_saved_subsets(path, 4, metric="global_to_subset")
    saved = json.loads(path.read_text())
    words_by_trial = {trial["trial"]: np.array([["ACGT".index(x) for x in word]
                                               for word in trial["global_collection"]]) for trial in saved["trials"]}
    assert result["metric"] == "global_to_subset_mean_nearest_hamming"
    assert result["direction"].startswith("lower")
    for trial in result["trials"]:
        words = words_by_trial[trial["trial"]]
        for method in ("centralized", "federated"):
            row = trial[method]
            ids = np.array(row["selected_ids"])
            nearest = np.array(row["nearest_hamming"])
            assert row["query_ids"] == list(range(24)) and len(nearest) == 24
            assert row["query_population_size"] == 24
            assert (nearest[ids] == 0).all()
            assert np.count_nonzero(nearest == 0) == 4
            expected = np.rint(cdist(words, words[ids], metric="hamming") * 3).min(axis=1)
            assert np.array_equal(nearest, expected)
            assert row["mean_nearest_hamming"] == expected.mean()
    directory = tmp_path / "coverage"
    directory.mkdir()
    paths = write_subset_nn_analysis(result, directory)
    assert all(path.name.startswith("global_to_subset_nn_boxplot") for path in paths)
    assert len((directory / "per_item_distances.csv").read_text().splitlines()) == 145
    assert "Selected global items contribute zero" in (directory / "report.md").read_text()
    assert "must not be used for this comparison" in (directory / "report.md").read_text()


def test_cli_defaults_to_requested_global_coverage(saved_results, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from fed_skdpp.subset_nn_analysis import main

    path, _ = saved_results
    assert main(["--centralized-results", str(path), "--selection-size", "4",
                 "--output-dir", str(tmp_path / "runs")]) == 0
    assert "global_to_subset_nn_boxplot.png" in capsys.readouterr().out
    directories = list((tmp_path / "runs").glob("global_to_subset_nn_*"))
    assert len(directories) == 1
    assert json.loads((directories[0] / "results.json").read_text())["metric"] == "global_to_subset_mean_nearest_hamming"
