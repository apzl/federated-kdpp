"""Paired gamma sensitivity of the existing forced-set SLS experiment.

Only the configured Hamming RBF gamma changes. It is shared by clients,
centralized reference, server and evaluation. This does not modify the
expansion paper's Section III swap rule, its forced-set objective, or the
explicitly non-private pooled-context protocol. Ranking measures agreement
between finite searches, not optimal diversity or held-out performance.
"""

import csv
from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from .config import ExperimentConfig
from .experiment import run_experiment
from .metrics import hamming_comparison_definition, hamming_sweep_summaries
from .sweep import _paired_data_signature, write_sweep_results


@dataclass(frozen=True)
class GammaSweep:
    gammas: tuple[float, ...] = tuple(index / 20 for index in range(1, 21))

    def __post_init__(self) -> None:
        if not self.gammas or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0 for value in self.gammas
        ):
            raise ValueError("gammas must be finite positive numbers")
        values = tuple(float(value) for value in self.gammas)
        if tuple(sorted(set(values))) != values:
            raise ValueError("gammas must be unique and increasing")
        object.__setattr__(self, "gammas", values)

    def as_dict(self) -> dict:
        return {"gammas": list(self.gammas)}


def load_gamma_sweep(path: str | Path) -> GammaSweep:
    """Read a strict YAML list, allowing grids without floating accumulation."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) != {"gammas"} or not isinstance(raw["gammas"], list):
        raise ValueError("gamma sweep config must contain only a gammas list")
    return GammaSweep(tuple(raw["gammas"]))


def rank_gammas(points: list[dict]) -> list[dict]:
    """Rank ascending symmetric nearest-neighbor Hamming distance, keeping ties.

    Input points have gamma and hamming_distance mean/std/min/max/num_trials.
    Competition ranks use only the mean, not logdet, overlap, or dispersion. Gamma ascending
    is display order within a tie, not a scientific preference. Empty input
    fails; the mean must be finite and cardinality/uniqueness is checked by the
    experiment runner. No best-single-trial selection is used.
    """
    if not points or any("hamming_distance" not in point or not math.isfinite(point["hamming_distance"]["mean"])
                         or point["hamming_distance"]["mean"] < 0 for point in points):
        raise ValueError("Ranking requires finite nonnegative observed Hamming distance means")
    ordered = sorted(points, key=lambda point: (point["hamming_distance"]["mean"], point["gamma"]))
    rows = []
    previous = None
    rank = 0
    for position, point in enumerate(ordered, 1):
        distance = point["hamming_distance"]
        if previous is None or distance["mean"] != previous:
            rank = position
        previous = distance["mean"]
        rows.append({
            "rank": rank, "gamma": point["gamma"], "mean_nearest_hamming": distance["mean"],
            "std_nearest_hamming": distance["std"], "min_nearest_hamming": distance["min"],
            "max_nearest_hamming": distance["max"], "num_trials": distance["num_trials"],
        })
    return rows


def run_gamma_sweep(
    base: ExperimentConfig, sweep: GammaSweep, backend: str = "dppy",
    on_trial: Callable[[float, dict], None] | None = None,
    on_point: Callable[[dict], None] | None = None,
) -> dict:
    """Run paired SLS trials per gamma with identical data/context/RNG streams.

    Base config is immutable; replace ONLY kernel.gamma for each point. All
    stages consume that same config, so no separate local/server bandwidth
    exists. Centralized and server initial sets match within each trial/point;
    across gammas they can differ because local suggestions change. Patience
    remains unchanged and does not equalize proposal counts. Pool retention
    is deliberately not calculated or used for ranking.
    """
    if backend not in {"numpy", "dppy"}:
        raise ValueError("backend must be numpy or dppy")
    if base.search.algorithm != "stochastic_local_search":
        raise ValueError("This gamma sweep requires stochastic_local_search, not MCMC")
    configs = [replace(base, kernel=replace(base.kernel, gamma=gamma)) for gamma in sweep.gammas]
    points = []
    signatures = None
    for gamma, config in zip(sweep.gammas, configs):
        callback = None if on_trial is None else lambda row, gamma=gamma: on_trial(gamma, row)
        experiment = run_experiment(config, backend, callback)
        current = [_paired_data_signature(trial) for trial in experiment["trials"]]
        if signatures is None:
            signatures = current
        elif current != signatures:
            raise AssertionError("Gamma points must share the same trial data, partitions and contexts")
        overlap = np.asarray([trial["metrics"]["intersection_size"] for trial in experiment["trials"]], dtype=np.float64)
        point = {
            "gamma": gamma,
            "overlap": {
                "mean": float(overlap.mean()),
                "std": float(overlap.std(ddof=1)) if len(overlap) > 1 else 0.0,
                "min": int(overlap.min()), "max": int(overlap.max()), "num_trials": len(overlap),
            },
            "experiment": experiment,
            **hamming_sweep_summaries(experiment),
        }
        points.append(point)
        if on_point is not None:
            on_point(point)
    ranking = rank_gammas(points)
    return {
        "backend": backend, "base_config": base.as_dict(), "sweep_config": sweep.as_dict(),
        "selection_comparison": hamming_comparison_definition(),
        "scope": "NON-PRIVATE pooled-context SLS baseline; gamma sensitivity, not private federation",
        "ranking_criterion": "minimize mean symmetric nearest-neighbor Hamming distance across all paired trials",
        "overlap_definition": "intersection of the two FINAL G1 sets; forced G0 excluded; not local-pool overlap",
        "dispersion": "sample standard deviation across paired trials, not a confidence interval",
        "pairing": "same trial data, client partitions, contexts and RNG streams across all gamma values",
        "gamma_policy": "one gamma shared by every client, centralized reference, server and evaluator",
        "initialization_caveat": "final methods share initial IDs within a point; gamma-dependent local pools can change initialization between points",
        "compute_budget_caveat": "same patience rule, not equal proposal or compute budgets",
        "selection_caveat": "best observed agreement on these tuning trials; not a universal gamma optimum or a held-out test result",
        "points": points, "ranking": ranking,
        "best_gammas": [row["gamma"] for row in ranking if row["rank"] == 1],
        "best_mean_nearest_hamming": ranking[0]["mean_nearest_hamming"],
    }


def write_gamma_results(result: dict, directory: Path) -> Path:
    """Save shared sweep artifacts and lowest-distance gamma ranking."""
    target = write_sweep_results(result, directory, parameter="gamma")
    with (target / "gamma_ranking.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["ranking"][0]))
        writer.writeheader()
        writer.writerows(result["ranking"])
    return target
