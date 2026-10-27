from dataclasses import replace
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.backends import KernelObjective
from fed_skdpp.cli import main, write_results
from fed_skdpp.config import (
    DataConfig, ExperimentConfig, OutputConfig, PartitionConfig,
    SearchConfig, SelectionConfig, TrialConfig,
)
from fed_skdpp.experiment import run_experiment, run_trial
from fed_skdpp.metrics import jaccard


@pytest.fixture
def small_config():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=5, final_selection_size=5),
        search=SearchConfig(patience=5), experiment=TrialConfig(seed=17, repetitions=2),
    )


def _without_timings(value):
    if isinstance(value, dict):
        return {key: _without_timings(item) for key, item in value.items() if "seconds" not in key and key != "backend"}
    if isinstance(value, (list, tuple)):
        return [_without_timings(item) for item in value]
    return value


def test_two_stage_pipeline_and_global_ids(small_config):
    result = run_trial(small_config, "numpy", 0)
    assert len(result["global_collection"]) == 40
    assert len(result["server_pool_ids"]) == 20
    assert len(result["centralized"]["selected_ids"]) == 5
    assert len(result["server"]["selected_ids"]) == 5
    for partition, client in zip(result["client_partitions"], result["clients"]):
        assert len(client["selected_ids"]) == 5
        assert set(client["selected_ids"]) <= set(partition)
    assert set(result["server"]["selected_ids"]) <= set(result["server_pool_ids"])
    assert result["centralized"]["initial_ids"] == result["server"]["initial_ids"]
    assert result["centralized"]["initial_logdet"] == pytest.approx(result["server"]["initial_logdet"])
    metrics = result["metrics"]
    assert metrics["federated_total_proposals"] == metrics["client_proposals"] + metrics["server_proposals"]
    assert metrics["estimated_client_upload_bytes"] == 20 * 8
    assert metrics["logdet_gap_fed_minus_central"] == pytest.approx(metrics["federated_logdet"] - metrics["centralized_logdet"])
    assert metrics["jaccard"] == jaccard(tuple(result["centralized"]["selected_ids"]), tuple(result["server"]["selected_ids"]))


def test_trial_reproducibility_and_independence(small_config):
    left = run_trial(small_config, "numpy", 0)
    right = run_trial(small_config, "numpy", 0)
    assert _without_timings(left) == _without_timings(right)
    other = run_trial(small_config, "numpy", 1)
    assert left["collection_sha256"] != other["collection_sha256"]


def test_one_client_without_compression_matches_central(small_config):
    config = replace(small_config, partition=PartitionConfig(num_clients=1), selection=SelectionConfig(client_selection_size=40, final_selection_size=5))
    result = run_trial(config, "numpy", 0)
    assert result["centralized"]["selected_ids"] == result["server"]["selected_ids"]
    assert result["metrics"]["jaccard"] == 1
    assert result["metrics"]["logdet_gap_fed_minus_central"] == 0


def test_dppy_is_a_real_container_with_shared_search(small_config):
    pytest.importorskip("dppy")
    from dppy.finite_dpps import FiniteDPP
    objective = KernelObjective(np.array([[0, 0], [1, 1]]), small_config.kernel, small_config.numerics, "dppy")
    assert isinstance(objective.dpp, FiniteDPP)
    assert objective.matrix is objective.dpp.L
    numpy_result = run_experiment(small_config, "numpy")
    dppy_result = run_experiment(small_config, "dppy")
    assert _without_timings(numpy_result) == _without_timings(dppy_result)


def test_jaccard_semantics():
    assert jaccard((1, 2), (2, 3)) == pytest.approx(1 / 3)
    assert jaccard((1, 2), (1, 2)) == 1
    assert jaccard((1,), (2,)) == 0
    assert jaccard((), ()) == 1


def test_output_records_config_and_never_overwrites(tmp_path, small_config):
    result = run_experiment(small_config, "numpy")
    first = write_results(result, tmp_path)
    second = write_results(result, tmp_path)
    assert first != second
    assert (first / "results.json").exists()
    assert (first / "metrics.csv").exists()
    assert (first / "search_history.csv").exists()
    assert (first / "config.yml").exists()
    parsed = json.loads((first / "results.json").read_text())
    assert parsed["config"]["search"]["patience"] == 5
    assert parsed["trials"][0]["collection_sha256"] == result["trials"][0]["collection_sha256"]


def test_history_can_be_disabled(small_config):
    config = replace(small_config, output=OutputConfig(save_history=False))
    result = run_trial(config, "numpy", 0)
    assert result["centralized"]["history"] == ()
    assert result["server"]["history"] == ()


def test_cli_loads_config_and_writes_results(tmp_path, small_config, capsys):
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.safe_dump(small_config.as_dict()), encoding="utf-8")
    output = tmp_path / "output"
    assert main("numpy", ["--config", str(config_path), "--output-dir", str(output)]) == 0
    assert "Results:" in capsys.readouterr().out
    result_path = next(output.glob("*/results.json"))
    result = json.loads(result_path.read_text())
    assert result["metadata"]["dtype"] == "float64"
    assert result["metadata"]["privacy_guarantee"] is False
    assert len(result["metadata"]["source_sha256"]) == 64


def test_cli_reports_invalid_config(tmp_path, capsys):
    config_path = tmp_path / "config.yml"
    config_path.write_text("kernel:\n  gamma: 0\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main("numpy", ["--config", str(config_path)])
    assert error.value.code == 2
    assert "strictly positive" in capsys.readouterr().err


def test_independent_initialization_mode(small_config):
    config = replace(small_config, search=SearchConfig(patience=5, initialization="independent_uniform"))
    result = run_trial(config, "numpy", 0)
    assert set(result["centralized"]["initial_ids"]) <= set(range(40))
    assert set(result["server"]["initial_ids"]) <= set(result["server_pool_ids"])
