from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.centralized_pool import (
    CentralizedPoolConfig, CentralizedSampling, CentralizedSizeSweep, SavedPoolSource,
    load_centralized_pool_config, read_distributed_source, run_centralized_pool_experiment,
)
from fed_skdpp.centralized_pool_reporting import (
    centralized_summaries, plot_centralized_pool,
    write_centralized_pool_report, write_centralized_pool_results,
)
from fed_skdpp.config import DataConfig, PartitionConfig, TrialConfig
from fed_skdpp.kernels import hamming_rbf, stable_logdet
from fed_skdpp.variable_dpp import selected_indices
from fed_skdpp.variable_pool import VariablePoolConfig, run_variable_pool_experiment


pytest.importorskip("dppy")


@pytest.fixture
def source(tmp_path):
    config = VariablePoolConfig(data=DataConfig(kmer_length=3, global_collection_size=24),
                                partition=PartitionConfig(num_clients=3),
                                experiment=TrialConfig(seed=42, repetitions=2))
    result = run_variable_pool_experiment(config)
    path = tmp_path / "distributed.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return path, result


def small_config(path):
    return CentralizedPoolConfig(source=SavedPoolSource(str(path)),
                                 sweep=CentralizedSizeSweep(minimum_selection_size=2,
                                                            maximum_selection_size=12,
                                                            selection_size_interval=2))


def test_exact_original_data_full_global_kernel_and_every_score(source, monkeypatch):
    path, original = source
    before = path.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Centralized baseline must reuse data and must not draw local DPPs")

    monkeypatch.setattr("fed_skdpp.data.sample_kmers", forbidden)
    monkeypatch.setattr("fed_skdpp.variable_dpp.OrdinaryDPPSampler.sample", forbidden)
    result = run_centralized_pool_experiment(small_config(path))
    assert result["inherited_config"]["kernel"]["gamma"] == .6
    assert result["source"]["sha256"] == hashlib.sha256(before).hexdigest()
    assert path.read_bytes() == before
    for trial, saved in zip(result["trials"], original["trials"]):
        assert trial["global_collection"] == saved["global_collection"]
        assert trial["collection_sha256"] == saved["collection_sha256"]
        assert trial["population_size"] == 24
        assert trial["dense_kernel_bytes"] == 24 * 24 * 8
        assert trial["sweep"]["diagnostics"]["population_size"] == 24
        encoded = np.array([["ACGT".index(x) for x in word] for word in saved["global_collection"]])
        matrix = hamming_rbf(encoded, .6)
        assert [row["selection_size"] for row in trial["sweep"]["records"]] == [2, 4, 6, 8, 10, 12]
        for row in trial["sweep"]["records"]:
            ids = selected_indices(row, trial["sweep"]["permutations"])
            assert len(ids) == len(set(ids)) == row["selection_size"]
            assert set(ids) <= set(range(24))
            assert row["logdet"] == pytest.approx(stable_logdet(matrix[np.ix_(ids, ids)]), abs=1e-12)
        distributed = {r["selection_size"]: r["logdet"] for r in saved["sweep"]["records"]}
        expected_common = sorted(set(distributed) & {2, 4, 6, 8, 10, 12})
        assert [r["selection_size"] for r in trial["comparison"]] == expected_common
        for row in trial["comparison"]:
            assert row["distributed_logdet"] == distributed[row["selection_size"]]
            assert row["logdet_gap_central_minus_distributed"] == pytest.approx(
                row["centralized_logdet"] - row["distributed_logdet"])


def test_reproducible_separate_streams_and_global_rng_unchanged(source):
    path, _ = source
    config = small_config(path)
    state = np.random.get_state()
    first = run_centralized_pool_experiment(config)
    second = run_centralized_pool_experiment(config)
    changed_seed = run_centralized_pool_experiment(replace(config, experiment=CentralizedSampling(seed=43)))
    for a, b, changed in zip(first["trials"], second["trials"], changed_seed["trials"]):
        assert a["collection_sha256"] == changed["collection_sha256"]
        assert a["sweep"]["records"] == b["sweep"]["records"]
        assert a["sweep"]["permutations"] == b["sweep"]["permutations"]
        assert a["sweep"]["permutations"] != changed["sweep"]["permutations"]
        assert a["rng_seed_components"] == [42, a["trial"], 4, 0]
    after = np.random.get_state()
    assert state[0] == after[0] and np.array_equal(state[1], after[1]) and state[2:] == after[2:]


@pytest.mark.parametrize("corruption", ["hash", "duplicate", "alphabet", "schema", "trial_id", "pool", "score"])
def test_source_validation_rejects_tampering_without_regeneration(source, corruption):
    path, result = source
    if corruption == "hash":
        result["trials"][0]["collection_sha256"] = "bad"
    elif corruption == "duplicate":
        result["trials"][0]["global_collection"][0] = result["trials"][0]["global_collection"][1]
    elif corruption == "alphabet":
        result["trials"][0]["global_collection"][0] = "NNN"
    elif corruption == "schema":
        result["schema"] = "other_experiment"
    elif corruption == "trial_id":
        result["trials"][1]["trial"] = result["trials"][0]["trial"]
    elif corruption == "pool":
        result["trials"][0]["pool_ids"][0] = 100
    else:
        result["trials"][0]["sweep"]["records"][0]["logdet"] = float("inf")
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError):
        read_distributed_source(path)


def test_size_grid_defaults_endpoints_and_capacity(source):
    sizes = CentralizedSizeSweep().sizes()
    assert sizes[0] == 10 and sizes[-1] == 5000 and len(sizes) == 999
    assert np.all(np.diff(sizes) == 5)
    assert CentralizedSizeSweep(10, 22, 5).sizes() == [10, 15, 20, 22]
    assert CentralizedSizeSweep(10, 10, 5).sizes() == [10]
    path, _ = source
    with pytest.raises(ValueError, match="saved global dataset"):
        run_centralized_pool_experiment(CentralizedPoolConfig(SavedPoolSource(str(path))))
    with pytest.raises(ValueError):
        CentralizedSizeSweep(10, 5, 5)
    with pytest.raises(ValueError):
        CentralizedSizeSweep(selection_size_interval=0)
    with pytest.raises(ValueError):
        CentralizedSampling(seed=-1)
    with pytest.raises(ValueError):
        SavedPoolSource("")


def test_strict_yaml_relative_source_paths_and_inherited_settings(source, tmp_path):
    path, _ = source
    config_path = tmp_path / "central.yml"
    config_path.write_text("source:\n  results: distributed.json\n", encoding="utf-8")
    config = load_centralized_pool_config(config_path)
    assert config.source.results == str(path.resolve())
    assert config.sweep.maximum_selection_size == 5000 and not config.output.save_history
    for raw in ({"source": {"results": "distributed.json"}, "kernel": {"gamma": .25}},
                {"source": {"results": "distributed.json"}, "sweep": {"maximum_size": 4}},
                {"source": {"results": "distributed.json"}, "experiment": {"repetitions": 1}},
                {"sweep": {}}, {"source": []}):
        config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError):
            load_centralized_pool_config(config_path)


def test_reports_plots_persist_scores_and_refuse_overwrites(source, tmp_path):
    pytest.importorskip("matplotlib")
    path, _ = source
    result = run_centralized_pool_experiment(small_config(path))
    directory = tmp_path / "artifacts"
    directory.mkdir()
    write_centralized_pool_results(result, directory)
    paths = plot_centralized_pool(result, directory)
    report = write_centralized_pool_report(result, directory)
    assert len(paths) == 4 and all(path.stat().st_size > 100 for path in paths)
    assert (directory / "centralized_logdet.csv").is_file()
    assert (directory / "centralized_summary.csv").is_file()
    assert (directory / "inherited_config.yml").is_file()
    stored = json.loads((directory / "results.json").read_text())
    assert stored["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "not its spectral Algorithm 2" in report.read_text()
    assert "missing data are not extrapolated" in report.read_text()
    summary, gaps = centralized_summaries(result)
    assert all(row["num_trials"] == 2 for row in summary)
    assert all(row["num_trials"] <= 2 for row in gaps)
    with pytest.raises(FileExistsError):
        write_centralized_pool_results(result, directory)


def test_cli_uses_saved_source_and_writes_separate_directory(source, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from fed_skdpp.centralized_pool_cli import main

    path, _ = source
    config_path = tmp_path / "central.yml"
    config_path.write_text(yaml.safe_dump({"source": {"results": str(path)},
                                         "sweep": {"minimum_selection_size": 2,
                                                   "maximum_selection_size": 8,
                                                   "selection_size_interval": 2}}), encoding="utf-8")
    assert main(["--config", str(config_path), "--output-dir", str(tmp_path / "runs")]) == 0
    assert "Completed" in capsys.readouterr().out
    directories = list((tmp_path / "runs").glob("centralized_global_pool_kdpp_*"))
    assert len(directories) == 1 and (directories[0] / "report.md").exists()
