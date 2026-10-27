from dataclasses import replace
import csv
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.config import (
    DataConfig, ExperimentConfig, OutputConfig, PartitionConfig, SearchConfig,
    SelectionConfig, TrialConfig, load_config,
)
from fed_skdpp.gamma_cli import main
from fed_skdpp.gamma_sweep import (
    GammaSweep, load_gamma_sweep, rank_gammas, run_gamma_sweep, write_gamma_results,
)
from fed_skdpp.kernels import hamming_rbf, stable_logdet
from fed_skdpp.plotting import plot_gamma_overlap


@pytest.fixture
def base():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=5, final_selection_size=8,
                                  forced_set_size=2, final_objective="pooled_context"),
        search=SearchConfig(patience=5), experiment=TrialConfig(seed=42, repetitions=3),
    )


def test_requested_configs_change_only_local_size_repetitions_and_output():
    original = load_config("config.yml")
    current = load_config("configs/gamma_sls.yml")
    assert current == replace(
        original, selection=replace(original.selection, client_selection_size=20),
        experiment=replace(original.experiment, repetitions=3), output=current.output,
    )
    assert current.data.kmer_length == 6 and current.selection.final_selection_size == 100
    assert current.kernel.jitter == 0 and current.search.patience == 50
    assert current.output.save_history and not current.output.save_chain
    sweep = load_gamma_sweep("configs/gamma_sweep.yml")
    assert sweep.gammas == tuple(value / 20 for value in range(1, 21))
    assert sweep.gammas[0] == 0.05 and sweep.gammas[-1] == 1
    assert len(sweep.gammas) == 20


@pytest.mark.parametrize("values", [(), (0,), (-1,), (True,), ("0.5",),
                                   (float("nan"),), (float("inf"),), (0.5, 0.5), (0.5, 0.1)])
def test_invalid_gamma_lists(values):
    with pytest.raises(ValueError):
        GammaSweep(values)


def test_strict_yaml_loader(tmp_path):
    path = tmp_path / "sweep.yml"
    for payload in ("gamma: [0.05, 1]\n", "gammas: 0.5\n", "gammas: [0.5]\nunknown: 1\n"):
        path.write_text(payload)
        with pytest.raises(ValueError):
            load_gamma_sweep(path)


def test_ranking_preserves_all_ties_without_utility_tiebreak():
    points = [{"gamma": gamma, "hamming_distance": {"mean": mean, "std": std, "min": 0, "max": 3, "num_trials": 3}}
              for gamma, mean, std in [(0.2, 1., .5), (0.1, 1., 0.), (0.3, 2., 0.)]]
    ranking = rank_gammas(points)
    assert [row["rank"] for row in ranking] == [1, 1, 3]
    assert [row["gamma"] for row in ranking] == [0.1, 0.2, 0.3]
    with pytest.raises(ValueError):
        rank_gammas([])


def test_sweep_pairs_data_changes_only_gamma_and_uses_all_trials(base):
    before = base.as_dict()
    progress, completed = [], []
    result = run_gamma_sweep(base, GammaSweep((0.05, 0.5, 1.0)), "numpy",
                             on_trial=lambda gamma, row: progress.append((gamma, row["trial"])),
                             on_point=lambda point: completed.append(point["gamma"]))
    assert base.as_dict() == before
    assert len(progress) == 9 and completed == [0.05, 0.5, 1.0]
    reference = result["points"][0]["experiment"]["trials"]
    assert len({trial["collection_sha256"] for trial in reference}) == 3
    for point in result["points"]:
        gamma = point["gamma"]
        assert point["experiment"]["config"] == replace(base, kernel=replace(base.kernel, gamma=gamma)).as_dict()
        values = []
        for a, trial in zip(reference, point["experiment"]["trials"]):
            assert trial["collection_sha256"] == a["collection_sha256"]
            assert trial["client_partitions"] == a["client_partitions"]
            assert trial["client_disclosed_context_ids"] == a["client_disclosed_context_ids"]
            assert trial["centralized"]["initial_ids"] == trial["server"]["initial_ids"]
            central, server = set(trial["centralized"]["selected_ids"]), set(trial["server"]["selected_ids"])
            assert len(central) == len(server) == 8
            assert server <= set(trial["server_pool_ids"])
            values.append(len(central & server))
            alphabet = {char: index for index, char in enumerate(base.data.alphabet)}
            encoded = np.asarray([[alphabet[char] for char in word] for word in trial["global_collection"]])
            contexts = trial["client_disclosed_context_ids"]
            stages = [(trial["centralized"], trial["server_shared_context_ids"]),
                      (trial["server"], trial["server_shared_context_ids"]), *zip(trial["clients"], contexts)]
            for stage, forced in stages:
                joint = sorted(set(forced) | set(stage["selected_ids"]))
                expected = stable_logdet(hamming_rbf(encoded[joint], gamma, base.kernel.jitter))
                assert stage["logdet"] == pytest.approx(expected, abs=1e-10)
                history = [row["logdet"] for row in stage["history"]]
                assert np.all(np.diff(history) >= 0)
        assert point["overlap"]["mean"] == np.mean(values)
        assert point["overlap"]["std"] == np.std(values, ddof=1)
    lowest = min(point["hamming_distance"]["mean"] for point in result["points"])
    assert result["best_mean_nearest_hamming"] == lowest
    assert result["best_gammas"] == sorted(point["gamma"] for point in result["points"] if point["hamming_distance"]["mean"] == lowest)
    assert "pool_centralized_overlap" not in json.dumps(result)


def test_invalid_backend_or_mcmc_fails_before_progress(base):
    with pytest.raises(ValueError, match="backend"):
        run_gamma_sweep(base, GammaSweep((0.5,)), "unknown")
    with pytest.raises(ValueError, match="not MCMC"):
        run_gamma_sweep(replace(base, search=SearchConfig(algorithm="dppy_mcmc")), GammaSweep((0.5,)))


def test_single_trial_and_optional_history(base, tmp_path):
    base = replace(base, experiment=TrialConfig(repetitions=1), output=OutputConfig(save_history=False))
    result = run_gamma_sweep(base, GammaSweep((0.5,)), "numpy")
    assert result["points"][0]["overlap"]["std"] == 0
    target = write_gamma_results(result, tmp_path)
    assert not (target / "search_history.csv").exists()


def test_writer_preserves_outputs_and_uses_gamma_columns(base, tmp_path):
    result = run_gamma_sweep(base, GammaSweep((0.1, 0.5)), "numpy")
    first = write_gamma_results(result, tmp_path)
    second = write_gamma_results(result, tmp_path)
    assert first != second
    assert json.loads((first / "results.json").read_text())["best_gammas"] == result["best_gammas"]
    for name in ("metrics.csv", "hamming_distance_summary.csv", "overlap_summary.csv", "search_history.csv"):
        with (first / name).open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert "gamma" in reader.fieldnames and "client_selection_size" not in reader.fieldnames
            assert list(reader)
    with (first / "gamma_ranking.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
        assert len(rows) == 2 and int(rows[0]["rank"]) == 1
    assert not list(first.glob("*pool_retention*"))


def test_plot_and_cli_run_without_pool_retention(base, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("dppy")
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump(base.as_dict()))
    sweep = tmp_path / "sweep.yml"
    sweep.write_text("gammas: [0.05, 0.5]\n")
    assert main(["--config", str(config), "--sweep-config", str(sweep)]) == 0
    target = next((tmp_path / "results").iterdir())
    result = json.loads((target / "results.json").read_text())
    assert result["backend"] == "dppy"
    assert result["metadata"]["client_contexts_disclosed"]
    assert (target / "gamma_vs_nearest_hamming.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in (target / "gamma_vs_nearest_hamming.svg").read_text()
    assert not list(target.glob("*pool_retention*"))
    plot_gamma_overlap(result, target)
