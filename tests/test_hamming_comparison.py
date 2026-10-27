from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from fed_skdpp.config import (DataConfig, ExperimentConfig, OutputConfig,
                              PartitionConfig, SearchConfig, SelectionConfig, TrialConfig)
from fed_skdpp.experiment import run_experiment
from fed_skdpp.gamma_sweep import GammaSweep, rank_gammas, run_gamma_sweep
from fed_skdpp.hamming_analysis import (NotSLSComparison, main, reevaluate_file,
                                       reevaluate_sls_result)
from fed_skdpp.metrics import (HAMMING_METRIC, NORMALIZED_HAMMING_METRIC,
                              nearest_neighbor_hamming, scalar_summary)


def encoded(words):
    return np.array([list(word) for word in words])


def brute_distance(left, right):
    def directed(a, b):
        return sum(min(sum(x != y for x, y in zip(word, other)) for other in b) for word in a) / len(a)
    return (directed(left, right) + directed(right, left)) / 2


@pytest.fixture
def tiny():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=24),
        partition=PartitionConfig(num_clients=3),
        selection=SelectionConfig(client_selection_size=4, final_selection_size=5,
                                  forced_set_size=1, final_objective="pooled_context"),
        search=SearchConfig(patience=3, local_proposal_budget=7, global_proposal_budget=13),
        experiment=TrialConfig(seed=42, repetitions=2), output=OutputConfig(save_history=True),
    )


def test_symmetric_metric_uses_both_directions_exact_matches_and_many_to_one():
    result = nearest_neighbor_hamming(encoded(["AAAA", "AAAT"]), encoded(["AAAA", "TTTT"]))
    assert result["centralized_to_federated_mean_nearest_hamming"] == .5
    assert result["federated_to_centralized_mean_nearest_hamming"] == 1.5
    assert result[HAMMING_METRIC] == 1
    assert result[NORMALIZED_HAMMING_METRIC] == .25
    # Optimal one-to-one pairing would be (0 + 3)/2 = 1.5, a different metric.
    assert result[HAMMING_METRIC] != 1.5


def test_unequal_set_sizes_give_directed_means_equal_weight():
    result = nearest_neighbor_hamming(encoded(["AAAA", "AAAT", "AATT"]), encoded(["AAAA"]))
    assert result[HAMMING_METRIC] == .5  # (mean(0,1,2) + mean(0))/2, not 3/4
    assert result[NORMALIZED_HAMMING_METRIC] == .125


def test_identity_permutation_symmetry_and_raw_hamming_bounds():
    left = encoded(["ACGT", "AAAT", "TTTT"])
    right = encoded(["AAAA", "TGGT"])
    a, b = nearest_neighbor_hamming(left, right), nearest_neighbor_hamming(right, left)
    assert a[HAMMING_METRIC] == b[HAMMING_METRIC]
    assert a[HAMMING_METRIC] == brute_distance(["ACGT", "AAAT", "TTTT"], ["AAAA", "TGGT"])
    assert nearest_neighbor_hamming(left, left[::-1])[HAMMING_METRIC] == 0
    assert nearest_neighbor_hamming(encoded(["AAAA"]), encoded(["TTTT"]))[HAMMING_METRIC] == 4
    assert nearest_neighbor_hamming(encoded(["AAAA"]), encoded(["TTTT"]))[NORMALIZED_HAMMING_METRIC] == 1


def test_average_nearest_distance_is_not_claimed_to_obey_triangle_inequality():
    a, b, c = encoded(["AAAA"]), encoded(["AAAA", "TTTT"]), encoded(["TTTT"])
    assert nearest_neighbor_hamming(a, c)[HAMMING_METRIC] > (
        nearest_neighbor_hamming(a, b)[HAMMING_METRIC] + nearest_neighbor_hamming(b, c)[HAMMING_METRIC])


@pytest.mark.parametrize("left,right", [
    (np.empty((0, 4)), np.ones((1, 4))), (np.ones((1, 0)), np.ones((1, 0))),
    (np.array([0, 1]), np.ones((1, 2))), (np.ones((1, 3)), np.ones((1, 4))),
    (np.array([[float("nan")]]), np.ones((1, 1))),
    (np.array([[float("inf")]]), np.ones((1, 1))),
    (np.array([[object()]]), np.ones((1, 1))),
])
def test_bad_hamming_inputs_fail(left, right):
    with pytest.raises(ValueError):
        nearest_neighbor_hamming(left, right)


def test_summary_uses_sample_std_and_rejects_invalid_data():
    result = scalar_summary([1, 2, 3])
    assert result == {"mean": 2, "std": 1, "min": 1, "max": 3, "num_trials": 3}
    assert scalar_summary([1.25])["std"] == 0
    for values in ([], [float("nan")], [[1], [2]]):
        with pytest.raises(ValueError):
            scalar_summary(values)


def test_live_sls_metrics_match_independent_raw_sequence_calculation(tiny):
    result = run_experiment(tiny, "numpy")
    assert result["selection_comparison"]["primary_metric"] == HAMMING_METRIC
    for trial in result["trials"]:
        value = brute_distance(trial["centralized_kmers"], trial["federated_kmers"])
        assert trial["metrics"][HAMMING_METRIC] == pytest.approx(value)
        assert trial["metrics"][NORMALIZED_HAMMING_METRIC] == pytest.approx(value / 3)
        assert trial["centralized"]["iterations"] == trial["server"]["iterations"] == 13
    values = [trial["metrics"][HAMMING_METRIC] for trial in result["trials"]]
    assert result["summary"][HAMMING_METRIC]["mean"] == np.mean(values)
    assert result["summary"][HAMMING_METRIC]["std"] == np.std(values, ddof=1)


def test_gamma_ranking_minimizes_distance_not_overlap_or_dispersion():
    points = [{"gamma": gamma, "overlap": {"mean": overlap},
               "hamming_distance": {"mean": distance, "std": sd, "min": 0, "max": 3, "num_trials": 3}}
              for gamma, overlap, distance, sd in ((.1, 9, 1.5, 0), (.2, 1, .5, .2), (.3, 2, .5, 0))]
    ranked = rank_gammas(points)
    assert [row["gamma"] for row in ranked] == [.2, .3, .1]
    assert [row["rank"] for row in ranked] == [1, 1, 3]
    for invalid in ([], [{"gamma": .1}], [{"gamma": .1, "hamming_distance": {"mean": -1}}]):
        with pytest.raises(ValueError):
            rank_gammas(invalid)


def test_saved_reanalysis_preserves_all_selections_scores_config_and_source(tiny):
    source = run_experiment(tiny, "numpy")
    for trial in source["trials"]:
        for key in (HAMMING_METRIC, NORMALIZED_HAMMING_METRIC):
            trial["metrics"].pop(key)
    original = deepcopy(source)
    updated = reevaluate_sls_result(source)
    assert source == original
    assert updated["config"] == source["config"]
    for a, b in zip(source["trials"], updated["trials"]):
        assert a["global_collection"] == b["global_collection"]
        assert a["server_shared_context_ids"] == b["server_shared_context_ids"]
        for stage in ("centralized", "server"):
            for key in ("selected_ids", "initial_ids", "logdet", "iterations", "total_seconds"):
                assert a[stage][key] == b[stage][key]
            assert a[stage]["history"] and not b[stage]["history"]
        assert b["metrics"][HAMMING_METRIC] == pytest.approx(brute_distance(b["centralized_kmers"], b["federated_kmers"]))


def test_saved_gamma_sweep_is_reranked_with_all_points_and_trials(tiny):
    source = run_gamma_sweep(tiny, GammaSweep((.1, .25, 1.)), "numpy")
    source["ranking"] = [{"rank": 1, "gamma": 1., "mean_overlap": 5}]
    source["best_mean_overlap"] = 5
    source["ranking_criterion"] = "legacy overlap"
    updated = reevaluate_sls_result(source)
    assert updated["prior_ranking"] == source["ranking"]
    assert updated["prior_ranking_criterion"] == "legacy overlap"
    assert "best_mean_overlap" not in updated
    assert updated["ranking"] == rank_gammas(updated["points"])
    for point in updated["points"]:
        values = [trial["metrics"][HAMMING_METRIC] for trial in point["experiment"]["trials"]]
        assert point["hamming_distance"] == scalar_summary(values)


def test_reanalysis_rejects_sampling_euclidean_calibration_and_corrupt_ids(tiny):
    source = run_experiment(tiny, "numpy")
    for algorithm in ("dppy_exact", "dppy_mcmc"):
        changed = deepcopy(source)
        changed["config"]["search"]["algorithm"] = algorithm
        with pytest.raises(NotSLSComparison):
            reevaluate_sls_result(changed)
    with pytest.raises(NotSLSComparison):
        reevaluate_sls_result({"config": {"dimension": 4}, "trials": [{}]})
    changed = deepcopy(source)
    changed["trials"][0].pop("server")
    with pytest.raises(NotSLSComparison):
        reevaluate_sls_result(changed)
    changed = deepcopy(source)
    changed["trials"][0]["server"]["selected_ids"][0] = -1
    with pytest.raises(ValueError, match="selection IDs"):
        reevaluate_sls_result(changed)
    changed = deepcopy(source)
    changed["trials"][0]["centralized_kmers"][0] = "XXX"
    with pytest.raises(ValueError, match="disagree"):
        reevaluate_sls_result(changed)


def test_reanalysis_writer_provenance_csv_reports_and_no_overwrite(tiny, tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps(run_experiment(tiny, "numpy")))
    before = source.read_bytes()
    target = tmp_path / "analysis"
    updated = reevaluate_file(source, target, make_plots=False)
    assert source.read_bytes() == before
    assert updated["reevaluation"]["source_results_sha256"] == hashlib.sha256(before).hexdigest()
    assert not updated["reevaluation"]["searches_rerun"]
    assert (target / "metrics.csv").exists()
    assert "No searches were rerun" in (target / "report.md").read_text()
    with pytest.raises(FileExistsError):
        reevaluate_file(source, target, make_plots=False)


def test_reanalysis_cli_discovers_sls_and_skips_other_scopes(tiny, tmp_path):
    root = tmp_path / "saved"
    for name, result in (("sls", run_experiment(tiny, "numpy")), ("euclidean", {"config": {"dimension": 4}})):
        directory = root / name; directory.mkdir(parents=True)
        (directory / "results.json").write_text(json.dumps(result))
    output = tmp_path / "output"
    assert main(["--all-saved", "--results-root", str(root), "--output-dir", str(output), "--no-plots"]) == 0
    directory = next(output.iterdir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest["completed"]) == len(manifest["skipped"]) == 1
    assert not manifest["searches_rerun"]
