"""Postprocess saved sweeps to separate local exclusion from final disagreement.

Counts refer only to new-item G1 selections, not fixed contexts. The full-universe
centralized result is a heuristic or sample reference, not a unique or optimal answer.
Membership disagreement does not measure the utility lost through compression.
"""

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np


COUNT_METRICS = (
    "pool_centralized_overlap", "excluded_locally", "final_overlap",
    "available_not_selected_by_server",
)


def _unique_ids(values: list[int], label: str, universe_size: int) -> set[int]:
    if any(type(value) is not int or not 0 <= value < universe_size for value in values):
        raise ValueError(f"{label} must contain integer IDs in the public universe")
    result = set(values)
    if len(result) != len(values):
        raise ValueError(f"{label} must contain unique IDs")
    return result


def analyze_pool_retention(sweep: dict) -> dict:
    """Derive local-pool retention from an unmodified saved sweep payload.

    Let P be server_pool_ids, A the full-universe centralized selected_ids,
    and B the server selected_ids (all in global collection index space).
    Return integer per-trial counts and float64 mean/sample-SD summaries:
    retained=|P intersect A|, excluded=|A minus P|, final=|A intersect B|,
    and available_not_selected=|(A intersect P) minus B|.

    Require B subset P, final cardinality from config, and consistent pool IDs.
    Reject malformed/inconsistent records rather than inferring lost candidates.
    Do not re-run optimization, mutate input, or export private memberships.
    """
    config = sweep["base_config"]
    universe_size = config["data"]["global_collection_size"]
    selection_size = config["selection"]["final_selection_size"]
    points = []
    for point in sweep["points"]:
        rows = []
        for trial in point["experiment"]["trials"]:
            pool = _unique_ids(trial["server_pool_ids"], "local pool", universe_size)
            central = _unique_ids(trial["centralized"]["selected_ids"], "centralized G1", universe_size)
            server = _unique_ids(trial["server"]["selected_ids"], "server G1", universe_size)
            if len(central) != selection_size or len(server) != selection_size:
                raise ValueError("Final selections must have the configured G1 cardinality")
            if not server <= pool:
                raise ValueError("Server G1 must be a subset of the local pool")
            suggestions = set().union(*(
                _unique_ids(client["selected_ids"], "client suggestions", universe_size)
                for client in trial["clients"]
            ))
            if pool != suggestions:
                raise ValueError("Local pool must be exactly the union of client suggestions")
            context = set(trial.get("server_shared_context_ids", []))
            if context & (pool | central | server):
                raise ValueError("New-item selections must exclude the fixed shared context")
            retained = central & pool
            rows.append({
                "trial": trial["trial"], "server_pool_size": len(pool),
                "centralized_selection_size": len(central),
                "pool_centralized_overlap": len(retained),
                "excluded_locally": len(central - pool),
                "final_overlap": len(central & server),
                "available_not_selected_by_server": len(retained - server),
            })
        if not rows:
            raise ValueError("Each sweep size must contain at least one trial")
        summary = {}
        for metric in COUNT_METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=np.float64)
            summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": int(values.min()), "max": int(values.max()),
            }
        points.append({
            "client_selection_size": point["client_selection_size"],
            "num_trials": len(rows), "summary": summary, "trials": rows,
        })
    if not points:
        raise ValueError("Sweep must contain at least one evaluated size")
    return {
        "backend": sweep["backend"], "base_config": config,
        "sweep_config": sweep["sweep_config"],
        "original_experiment_metadata": sweep.get("metadata", {}),
        "definition": "P=union of local suggestions; A=full-universe centralized G1; B=server G1",
        "dispersion": "sample standard deviation across paired trials, not a confidence interval",
        "interpretation": "excluded_locally + available_not_selected_by_server = |A minus B|; not utility loss",
        "initialization_caveat": sweep.get("initialization_caveat", ""),
        "points": points,
    }


def write_pool_retention(result: dict, directory: Path) -> Path:
    """Save count-only analysis in a new directory, preserving source artifacts."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = directory / f"{result['backend']}_pool_retention_{stamp}_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    with (target / "analysis.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    trials = [
        {"client_selection_size": point["client_selection_size"], **row}
        for point in result["points"] for row in point["trials"]
    ]
    summaries = [
        {
            "client_selection_size": point["client_selection_size"], "num_trials": point["num_trials"],
            **{f"{metric}_{stat}": value for metric, stats in point["summary"].items() for stat, value in stats.items()},
        }
        for point in result["points"]
    ]
    for name, rows in (("pool_retention_trials.csv", trials), ("pool_retention_summary.csv", summaries)):
        with (target / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return target
