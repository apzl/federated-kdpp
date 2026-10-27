"""Reproducible client-suggestion-size sweeps of the existing experiment.

No kernel, objective, swap rule, privacy assumption, or initialization policy
is changed here. Equal trial indices reuse data/partition/context/RNG streams
across parameter values. Under paired_uniform, the shared final initialization
depends on the suggestion pool, so the centralized result is re-run per value
rather than incorrectly treated as one fixed reference across the sweep.
"""

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable
import uuid

import numpy as np
import yaml

from .config import ExperimentConfig
from .experiment import run_experiment
from .metrics import hamming_comparison_definition, hamming_sweep_summaries


@dataclass(frozen=True)
class ClientSelectionSweep:
    client_selection_sizes: tuple[int, ...] = tuple(range(10, 91, 10))

    def __post_init__(self) -> None:
        values = self.client_selection_sizes
        if not values or any(type(value) is not int or value < 1 for value in values):
            raise ValueError("client_selection_sizes must be a nonempty list of positive integers")
        if len(set(values)) != len(values) or tuple(sorted(values)) != tuple(values):
            raise ValueError("client_selection_sizes must be unique and increasing")
        object.__setattr__(self, "client_selection_sizes", tuple(values))

    def as_dict(self) -> dict:
        return {"client_selection_sizes": list(self.client_selection_sizes)}


def load_sweep_config(path: str | Path) -> ClientSelectionSweep:
    """Read a strict YAML mapping containing only client_selection_sizes."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"client_selection_sizes"}:
        raise ValueError("sweep config must contain only client_selection_sizes")
    if not isinstance(raw["client_selection_sizes"], list):
        raise ValueError("client_selection_sizes must be a YAML list")
    return ClientSelectionSweep(tuple(raw["client_selection_sizes"]))


def _paired_data_signature(trial: dict) -> tuple:
    return (
        trial["collection_sha256"],
        tuple(tuple(part) for part in trial["client_partitions"]),
        tuple(tuple(context) for context in trial.get("client_disclosed_context_ids", [])),
    )


def run_client_selection_sweep(
    base: ExperimentConfig, sweep: ClientSelectionSweep, backend: str,
    on_trial: Callable[[int, dict], None] | None = None,
    on_point: Callable[[dict], None] | None = None,
    artifact_directory: Path | None = None,
) -> dict:
    """Run the original experiment for each feasible client selection size.

    Validate ALL points before running anything. The base config is immutable;
    only selection.client_selection_size changes. Retain each full experiment
    result, including selections, seeds, config, scores, and search histories.
    SLS uses symmetric cross-set nearest-neighbor Hamming distance as its
    primary comparison. Legacy overlap counts exclude the shared fixed G0.
    """
    if backend not in {"numpy", "dppy"}:
        raise ValueError("backend must be numpy or dppy")
    configs = []
    for size in sweep.client_selection_sizes:
        try:
            configs.append(replace(base, selection=replace(base.selection, client_selection_size=size)))
        except ValueError as exc:
            raise ValueError(f"Infeasible client_selection_size={size}: {exc}") from exc
    points = []
    signatures = None
    for size, config in zip(sweep.client_selection_sizes, configs):
        callback = None if on_trial is None else lambda row, size=size: on_trial(size, row)
        artifacts = artifact_directory / f"client_size_{size}" if artifact_directory else None
        experiment = run_experiment(config, backend, callback, artifacts)
        current = [_paired_data_signature(trial) for trial in experiment["trials"]]
        if signatures is None:
            signatures = current
        elif current != signatures:
            raise AssertionError("Each sweep value must reuse the same trial data, partitions, and contexts")
        overlap = np.array([trial["metrics"]["intersection_size"] for trial in experiment["trials"]], dtype=np.float64)
        point = {
            "client_selection_size": size,
            "overlap": {
                "mean": float(overlap.mean()),
                "std": float(overlap.std(ddof=1)) if len(overlap) > 1 else 0.0,
                "min": int(overlap.min()), "max": int(overlap.max()),
                "num_trials": len(overlap),
            },
            "experiment": experiment,
            **hamming_sweep_summaries(experiment),
        }
        points.append(point)
        if on_point is not None:
            on_point(point)
    return {
        "backend": backend, "base_config": base.as_dict(), "sweep_config": sweep.as_dict(),
        **({"selection_comparison": hamming_comparison_definition()}
           if base.search.algorithm == "stochastic_local_search" else {}),
        "scope": "parameter sweep of the unchanged experiment; pooled-context mode is non-private",
        "overlap_definition": "|G1_centralized intersection G1_federated|; shared fixed G0 is EXCLUDED",
        "dispersion": "sample standard deviation across paired trials; not a confidence interval",
        "pairing": "same trial data, partitions, contexts, and RNG streams across sizes",
        "initialization_caveat": (
            "exact samples have no initialization; central/server streams are independent; "
            "centralized exact sample is identical across sizes for each paired dataset"
            if base.search.algorithm == "dppy_exact" else
            "paired_uniform initializes both final methods from their parameter-specific server pool; "
            "centralized selection is re-run and can change across sizes"
        ),
        "compute_budget_caveat": (
            "one exact sample per stage; ten clients plus server add compute; no optimization budget"
            if base.search.algorithm == "dppy_exact" else
            "equal chain transitions per stage; lazy steps are not proposals; local preprocessing adds compute; no mixing guarantee"
            if base.search.algorithm == "dppy_mcmc" else
            f"fixed proposal budgets: local={base.search.local_proposal_budget}, global={base.search.global_proposal_budget}; "
            "no possible swap returns zero proposals; local preprocessing adds compute"
            if base.search.local_proposal_budget is not None or base.search.global_proposal_budget is not None else
            "unchanged patience stopping does not equalize actual proposal/compute budgets"
        ),
        "points": points,
    }


def write_sweep_results(
    result: dict, directory: Path, run_directory: Path | None = None,
    parameter: str = "client_selection_size",
) -> Path:
    """Save a unique sweep directory with full JSON, YAML, and tidy CSVs."""
    if parameter not in {"client_selection_size", "gamma", "kmer_length"}:
        raise ValueError("Unsupported sweep parameter")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if run_directory is None:
        name = {"gamma": "gamma_sweep", "kmer_length": "kmer_sweep",
                "client_selection_size": "client_selection_sweep"}[parameter]
        run_directory = directory / f"{result['backend']}_{name}_{stamp}_{uuid.uuid4().hex[:8]}"
        run_directory.mkdir(parents=True, exist_ok=False)
    elif any((run_directory / name).exists() for name in (
        "results.json", "config.yml", "sweep.yml", "overlap_summary.csv", "hamming_distance_summary.csv", "metrics.csv", "search_history.csv", "report.md",
    )):
        raise FileExistsError("Refusing to overwrite existing sweep artifacts")
    with (run_directory / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    for name, payload in (("config.yml", result["base_config"]), ("sweep.yml", result["sweep_config"])):
        with (run_directory / name).open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
    summary_rows = [
        {parameter: point[parameter], **point["overlap"]}
        for point in result["points"]
    ]
    metric_rows = [
        {parameter: point[parameter], **trial["metrics"]}
        for point in result["points"] for trial in point["experiment"]["trials"]
    ]
    for filename, rows in (("overlap_summary.csv", summary_rows), ("metrics.csv", metric_rows)):
        with (run_directory / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if all("hamming_distance" in point for point in result["points"]):
        rows = [{parameter: point[parameter], **point["hamming_distance"],
                 "normalized_mean": point["hamming_distance_normalized"]["mean"],
                 "normalized_std": point["hamming_distance_normalized"]["std"]}
                for point in result["points"]]
        with (run_directory / "hamming_distance_summary.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if result["base_config"]["output"]["save_history"]:
        with (run_directory / "search_history.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = [parameter, "trial", "stage", "iteration", "logdet", "accepted", "improved", "stalled_iterations"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for point in result["points"]:
                for trial in point["experiment"]["trials"]:
                    stages = [("centralized", trial["centralized"]), ("server", trial["server"])]
                    stages.extend((f"client_{index}", stage) for index, stage in enumerate(trial["clients"]))
                    for name, stage in stages:
                        for step in stage["history"]:
                            writer.writerow({
                                parameter: point[parameter],
                                "trial": trial["trial"], "stage": name, **step,
                            })
    if all("hamming_distance" in point for point in result["points"]):
        from .hamming_reporting import write_hamming_report
        write_hamming_report(result, run_directory / "report.md")
    return run_directory
