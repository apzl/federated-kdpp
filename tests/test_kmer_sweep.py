from dataclasses import replace
import csv
import json

import numpy as np
import pytest
import yaml

from fed_skdpp.config import (
    DataConfig, ExperimentConfig, KernelConfig, OutputConfig, PartitionConfig,
    SearchConfig, SelectionConfig, TrialConfig, load_config,
)
from fed_skdpp.kernels import hamming_rbf, stable_logdet
from fed_skdpp.kmer_cli import main
from fed_skdpp.kmer_sweep import (
    KmerLengthSweep, load_kmer_sweep, run_kmer_sweep, write_kmer_results,
)


@pytest.fixture
def base():
    return ExperimentConfig(
        data=DataConfig(kmer_length=3, global_collection_size=40),
        partition=PartitionConfig(num_clients=4),
        selection=SelectionConfig(client_selection_size=5, final_selection_size=8,
                                  forced_set_size=2, final_objective="pooled_context"),
        kernel=KernelConfig(gamma=0.25), search=SearchConfig(patience=5),
        experiment=TrialConfig(seed=42, repetitions=3),
    )


def test_requested_configuration_and_grid_preserve_scientific_settings():
    gamma_base = load_config("configs/gamma_sls.yml")
    base = load_config("configs/kmer_sls.yml")
    assert base == replace(gamma_base, kernel=replace(gamma_base.kernel, gamma=0.25))
    assert base.search.patience == 50 and base.search.max_iterations is None
    assert base.selection.client_selection_size == 20
    assert base.selection.final_selection_size == 100
    assert base.experiment.repetitions == 3 and base.kernel.jitter == 0
    sweep = load_kmer_sweep("configs/kmer_sweep.yml")
    assert sweep.kmer_lengths == tuple(range(4, 21, 2))
    assert sweep.skip_infeasible


@pytest.mark.parametrize("values", [(), (0,), (-1,), (True,), (2.5,), ("6",), (6, 6), (8, 6)])
def test_invalid_length_lists(values):
    with pytest.raises(ValueError):
        KmerLengthSweep(values)


def test_strict_loader_and_skip_policy(tmp_path):
    path = tmp_path / "sweep.yml"
    for payload in ("lengths: [4, 6]\n", "kmer_lengths: 6\n", "kmer_lengths: [4]\nextra: 1\n",
                    "kmer_lengths: [4]\nskip_infeasible: 1\n"):
        path.write_text(payload)
        with pytest.raises(ValueError):
            load_kmer_sweep(path)
    path.write_text("kmer_lengths: [6, 8]\n")
    assert not load_kmer_sweep(path).skip_infeasible


def test_preflight_failure_does_not_start_any_trial(base):
    progress = []
    with pytest.raises(ValueError, match="Infeasible kmer_length=2"):
        run_kmer_sweep(base, KmerLengthSweep((2, 3)), on_trial=lambda *args: progress.append(args))
    assert not progress
    with pytest.raises(ValueError, match="No feasible"):
        run_kmer_sweep(base, KmerLengthSweep((1, 2), skip_infeasible=True))
    with pytest.raises(ValueError, match="backend"):
        run_kmer_sweep(base, KmerLengthSweep((3,)), "invalid")
    with pytest.raises(ValueError, match="not MCMC"):
        run_kmer_sweep(replace(base, search=SearchConfig(algorithm="dppy_mcmc")), KmerLengthSweep((3,)))


def test_explicit_skip_records_infeasibility_without_reconfiguring(base):
    skipped, progress = [], []
    result = run_kmer_sweep(base, KmerLengthSweep((2, 3), skip_infeasible=True), "numpy",
                            on_trial=lambda length, row: progress.append((length, row["trial"])),
                            on_skip=skipped.append)
    assert result["skipped_points"] == skipped
    assert skipped[0]["kmer_length"] == 2
    assert skipped[0]["unique_kmer_capacity"] == 16
    assert skipped[0]["required_collection_size"] == 40
    assert progress == [(3, 0), (3, 1), (3, 2)]
    assert result["points"][0]["experiment"]["config"] == base.as_dict()
    assert "pool_centralized_overlap" not in json.dumps(result)


def test_only_length_changes_with_paired_structure_and_shared_gamma(base):
    before = base.as_dict()
    completed = []
    result = run_kmer_sweep(base, KmerLengthSweep((3, 4, 6)), "numpy",
                            on_point=lambda point: completed.append(point["kmer_length"]))
    assert base.as_dict() == before and completed == [3, 4, 6]
    reference = result["points"][0]["experiment"]["trials"]
    for point in result["points"]:
        length = point["kmer_length"]
        expected_config = replace(base, data=replace(base.data, kmer_length=length))
        assert point["experiment"]["config"] == expected_config.as_dict()
        trials = point["experiment"]["trials"]
        assert len({trial["collection_sha256"] for trial in trials}) == 3
        overlaps = []
        for original, trial in zip(reference, trials):
            assert trial["client_partitions"] == original["client_partitions"]
            assert trial["client_disclosed_context_ids"] == original["client_disclosed_context_ids"]
            if length != 3:
                assert trial["collection_sha256"] != original["collection_sha256"]
            assert len(set(trial["global_collection"])) == 40
            assert all(len(word) == length for word in trial["global_collection"])
            assert trial["centralized"]["initial_ids"] == trial["server"]["initial_ids"]
            contexts = trial["client_disclosed_context_ids"]
            for a, b in zip(original["clients"], trial["clients"]):
                assert a["initial_ids"] == b["initial_ids"]
            alphabet = {char: index for index, char in enumerate(base.data.alphabet)}
            encoded = np.asarray([[alphabet[char] for char in word] for word in trial["global_collection"]])
            stages = [(trial["centralized"], trial["server_shared_context_ids"]),
                      (trial["server"], trial["server_shared_context_ids"]), *zip(trial["clients"], contexts)]
            for stage, context in stages:
                selected = set(stage["selected_ids"])
                assert len(selected) == stage["selection_size"]
                assert not selected & set(context)
                joint = sorted(selected | set(context))
                score = stable_logdet(hamming_rbf(encoded[joint], 0.25, 0))
                assert stage["logdet"] == pytest.approx(score, abs=1e-10)
                assert stage["stop_reason"] == "patience" and stage["stalled_iterations"] == 5
                assert np.all(np.diff([step["logdet"] for step in stage["history"]]) >= 0)
            a, b = set(trial["centralized"]["selected_ids"]), set(trial["server"]["selected_ids"])
            assert b <= set(trial["server_pool_ids"])
            overlaps.append(len(a & b))
            assert trial["metrics"]["intersection_size"] == overlaps[-1]
            assert trial["metrics"]["jaccard"] == pytest.approx(len(a & b) / len(a | b))
        assert point["overlap"]["mean"] == np.mean(overlaps)
        assert point["overlap"]["std"] == np.std(overlaps, ddof=1)


def test_fixed_seed_reproducibility(base):
    base = replace(base, experiment=TrialConfig(seed=42, repetitions=1), output=OutputConfig(save_history=False))
    a = run_kmer_sweep(base, KmerLengthSweep((3, 4)), "numpy")
    b = run_kmer_sweep(base, KmerLengthSweep((3, 4)), "numpy")
    for pa, pb in zip(a["points"], b["points"]):
        assert pa["overlap"]["std"] == 0
        ta, tb = pa["experiment"]["trials"][0], pb["experiment"]["trials"][0]
        assert ta["collection_sha256"] == tb["collection_sha256"]
        for stage in ("centralized", "server"):
            assert ta[stage]["selected_ids"] == tb[stage]["selected_ids"]
            assert ta[stage]["logdet"] == tb[stage]["logdet"]


def test_writer_preserves_prior_outputs_and_records_skips(base, tmp_path):
    result = run_kmer_sweep(base, KmerLengthSweep((2, 3), True), "numpy")
    first = write_kmer_results(result, tmp_path)
    second = write_kmer_results(result, tmp_path)
    assert first != second and "numpy_kmer_sweep" in first.name
    assert json.loads((first / "results.json").read_text())["skipped_points"] == result["skipped_points"]
    for filename in ("hamming_distance_summary.csv", "overlap_summary.csv", "metrics.csv", "search_history.csv", "skipped_points.csv"):
        with (first / filename).open(newline="") as handle:
            reader = csv.DictReader(handle)
            assert "kmer_length" in reader.fieldnames
            rows = list(reader)
            assert rows
            assert {row["kmer_length"] for row in rows} == ({"2"} if filename == "skipped_points.csv" else {"3"})
    assert not list(first.glob("*pool_retention*"))


def test_cli_and_plots_include_infeasible_point_and_no_pool_retention(base, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("dppy")
    config = tmp_path / "config.yml"
    config.write_text(yaml.safe_dump(base.as_dict()))
    sweep = tmp_path / "sweep.yml"
    sweep.write_text("kmer_lengths: [2, 3, 4]\nskip_infeasible: true\n")
    assert main(["--config", str(config), "--sweep-config", str(sweep)]) == 0
    target = next((tmp_path / "results").iterdir())
    result = json.loads((target / "results.json").read_text())
    assert result["backend"] == "dppy" and result["metadata"]["client_contexts_disclosed"]
    assert result["base_config"]["kernel"]["gamma"] == 0.25
    for name in ("kmer_length_nearest_hamming", "kmer_length_metrics"):
        assert (target / f"{name}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert "<svg" in (target / f"{name}.svg").read_text()
    assert "N/A" in (target / "kmer_length_nearest_hamming.svg").read_text()
    assert not list(target.glob("*pool_retention*"))
