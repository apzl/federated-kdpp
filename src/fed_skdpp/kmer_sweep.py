"""Sequence-length sensitivity of the existing forced-set SLS baseline.

Only data.kmer_length changes; the expansion paper's Section III swap rule,
forced-set objective and explicitly non-private pooled-context protocol stay
unchanged. Uniform unique sequences change with length, but client/context
index assignments and stochastic streams remain paired by trial ID.
"""

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .config import ExperimentConfig
from .experiment import run_experiment
from .metrics import hamming_comparison_definition, hamming_sweep_summaries
from .sweep import write_sweep_results


@dataclass(frozen=True)
class KmerLengthSweep:
    kmer_lengths: tuple[int, ...] = tuple(range(4, 21, 2))
    skip_infeasible: bool = False

    def __post_init__(self) -> None:
        values = tuple(self.kmer_lengths)
        if not values or any(type(value) is not int or value < 1 for value in values):
            raise ValueError("kmer_lengths must be a nonempty list of positive integers")
        if tuple(sorted(set(values))) != values:
            raise ValueError("kmer_lengths must be unique and increasing")
        if type(self.skip_infeasible) is not bool:
            raise ValueError("skip_infeasible must be a boolean")
        object.__setattr__(self, "kmer_lengths", values)

    def as_dict(self) -> dict:
        return {"kmer_lengths": list(self.kmer_lengths), "skip_infeasible": self.skip_infeasible}


def load_kmer_sweep(path: str | Path) -> KmerLengthSweep:
    """Read a strict YAML grid; infeasible points require explicit opt-in."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if (not isinstance(raw, dict) or "kmer_lengths" not in raw
            or set(raw) - {"kmer_lengths", "skip_infeasible"}
            or not isinstance(raw["kmer_lengths"], list)):
        raise ValueError("length sweep config requires a kmer_lengths list and optional skip_infeasible")
    return KmerLengthSweep(tuple(raw["kmer_lengths"]), raw.get("skip_infeasible", False))


def run_kmer_sweep(
    base: ExperimentConfig, sweep: KmerLengthSweep, backend: str = "dppy",
    on_trial: Callable[[int, dict], None] | None = None,
    on_point: Callable[[dict], None] | None = None,
    on_skip: Callable[[dict], None] | None = None,
) -> dict:
    """Run paired SLS comparisons, replacing ONLY the sequence length.

    Validate all points before executing trials. Infeasible points either fail
    or are explicitly recorded, never repaired with duplicates, extra jitter,
    smaller collections or changed client settings. Central/server starts match
    within a trial; they can differ across lengths because local pools change.
    Equal patience does not equalize actual proposal counts or compute.
    """
    if backend not in {"numpy", "dppy"}:
        raise ValueError("backend must be numpy or dppy")
    if base.search.algorithm != "stochastic_local_search":
        raise ValueError("This length sweep requires stochastic_local_search, not MCMC")
    configs, skipped = [], []
    for length in sweep.kmer_lengths:
        try:
            config = replace(base, data=replace(base.data, kmer_length=length))
        except ValueError as exc:
            if not sweep.skip_infeasible:
                raise ValueError(f"Infeasible kmer_length={length}: {exc}") from exc
            skipped.append({
                "kmer_length": length, "reason": str(exc),
                "unique_kmer_capacity": len(base.data.alphabet) ** length,
                "required_collection_size": base.data.global_collection_size,
            })
        else:
            configs.append((length, config))
    if not configs:
        raise ValueError("No feasible k-mer lengths; no trials were run")
    if on_skip is not None:
        for point in skipped:
            on_skip(point)
    points, reference = [], None
    for length, config in configs:
        callback = None if on_trial is None else lambda row, length=length: on_trial(length, row)
        experiment = run_experiment(config, backend, callback)
        structure = [
            (trial["client_partitions"], trial.get("client_disclosed_context_ids", []))
            for trial in experiment["trials"]
        ]
        if reference is None:
            reference = structure
        elif structure != reference:
            raise AssertionError("Trial partitions and context IDs must stay paired across lengths")
        values = np.asarray([trial["metrics"]["intersection_size"] for trial in experiment["trials"]], dtype=np.float64)
        point = {
            "kmer_length": length,
            "overlap": {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": int(values.min()), "max": int(values.max()), "num_trials": len(values),
            },
            "experiment": experiment,
            **hamming_sweep_summaries(experiment),
        }
        points.append(point)
        if on_point is not None:
            on_point(point)
    return {
        "backend": backend, "base_config": base.as_dict(), "sweep_config": sweep.as_dict(),
        "selection_comparison": hamming_comparison_definition(),
        "scope": "length sensitivity of the unchanged SLS suggestion baseline; pooled-context mode is NON-PRIVATE",
        "overlap_definition": "intersection of the two FINAL G1 sets; fixed G0 excluded; no pool-retention analysis",
        "dispersion": "sample standard deviation across configured trials; not a confidence interval",
        "pairing": "fresh uniform unique sequences per length/trial; same partitions, context item indices and RNG streams across lengths",
        "gamma_policy": "one fixed gamma shared by every client, centralized reference, server and evaluator at every length",
        "initialization_caveat": "paired final initialization within a point; length-dependent suggestions can change initial IDs across lengths",
        "compute_budget_caveat": "same patience rule, not equal proposal counts or compute budgets",
        "kernel_caveat": "longer sequences change raw Hamming distances and the kernel at fixed gamma; raw logdet across lengths is not a common utility scale",
        "evaluation_caveat": "synthetic sensitivity experiment; a gamma selected on the same seeds is exploratory tuning, not held-out validation",
        "skipped_points": skipped, "points": points,
    }


def write_kmer_results(result: dict, directory: Path) -> Path:
    """Write full shared sweep artifacts and an explicit infeasibility CSV."""
    target = write_sweep_results(result, directory, parameter="kmer_length")
    if result["skipped_points"]:
        with (target / "skipped_points.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["skipped_points"][0]))
            writer.writeheader()
            writer.writerows(result["skipped_points"])
    return target
