"""Re-evaluate saved k-mer SLS selections without invoking an optimizer.

This evaluates final public G1 sets only. Private contexts are not needed for
Hamming comparison. Original files are read-only; new compact artifacts keep
their provenance, selections, scores and configuration, but omit trajectories.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from .cli import allocate_run_directory
from .gamma_sweep import rank_gammas
from .hamming_reporting import write_hamming_report
from .metrics import (HAMMING_KEYS, HAMMING_METRIC, NORMALIZED_HAMMING_METRIC,
                      hamming_comparison_definition, hamming_sweep_summaries,
                      nearest_neighbor_hamming, scalar_summary)


class NotSLSComparison(ValueError):
    """A saved record is outside the requested paired k-mer SLS scope."""


def _compact_copy(value):
    if isinstance(value, dict):
        return {key: [] if key == "history" else _compact_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_compact_copy(item) for item in value]
    return value


def _validate_experiment(experiment):
    config = experiment.get("config", {})
    if "data" not in config or config.get("search", {}).get("algorithm") != "stochastic_local_search":
        raise NotSLSComparison("not a k-mer stochastic-local-search experiment")
    if not experiment.get("trials") or any("server" not in trial or "centralized" not in trial for trial in experiment["trials"]):
        raise NotSLSComparison("no paired centralized/server final selections (e.g. budget calibration only)")


def _selected_words(trial, stage, config):
    collection = trial.get("global_collection")
    if not isinstance(collection, list) or len(collection) != config["data"]["global_collection_size"]:
        raise ValueError("Missing or invalid saved public global_collection")
    ids = trial[stage].get("selected_ids")
    if (not isinstance(ids, list) or len(ids) != config["selection"]["final_selection_size"]
            or any(type(index) is not int or not 0 <= index < len(collection) for index in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Invalid saved final selection IDs or cardinality")
    words = [collection[index] for index in ids]
    length, alphabet = config["data"]["kmer_length"], config["data"]["alphabet"]
    if any(not isinstance(word, str) or len(word) != length or set(word) - set(alphabet) for word in words):
        raise ValueError("Invalid saved selected k-mer length or alphabet")
    key = "centralized_kmers" if stage == "centralized" else "federated_kmers"
    if key in trial and trial[key] != words:
        raise ValueError("Saved selected IDs disagree with the saved selected k-mers")
    return np.asarray([list(word) for word in words], dtype="U1")


def _enrich_experiment(experiment):
    _validate_experiment(experiment)
    for trial in experiment["trials"]:
        left = _selected_words(trial, "centralized", experiment["config"])
        right = _selected_words(trial, "server", experiment["config"])
        trial["metrics"].update(nearest_neighbor_hamming(left, right))
    summary = experiment.setdefault("summary", {})
    for key in HAMMING_KEYS:
        stats = scalar_summary([trial["metrics"][key] for trial in experiment["trials"]])
        summary[key] = {"mean": stats["mean"], "std": stats["std"]}
    experiment["selection_comparison"] = hamming_comparison_definition()


def reevaluate_sls_result(source: dict) -> dict:
    """Return an evaluation-only copy; never change data, sets or SLS scores.

    Accept the standard run schema or client-size/gamma/k-mer sweep schema.
    Non-SLS, Euclidean and central-only calibration records are rejected. Bad
    saved sequence/ID data fail rather than being guessed or silently repaired.
    Original search trajectories remain available only in the source record.
    """
    if "points" in source:
        if not source["points"]:
            raise ValueError("Cannot re-evaluate an empty sweep")
        for point in source["points"]:
            _validate_experiment(point.get("experiment", {}))
    else:
        _validate_experiment(source)
    result = _compact_copy(source)
    if "points" in result:
        for point in result["points"]:
            _enrich_experiment(point["experiment"])
            point.update(hamming_sweep_summaries(point["experiment"]))
        if "gamma" in result["points"][0]:
            if "ranking" in result:
                result["prior_ranking"] = result["ranking"]
                result["prior_ranking_criterion"] = result.get("ranking_criterion", "overlap")
            result["prior_best_gammas"] = result.get("best_gammas", [])
            if "best_mean_overlap" in result:
                result["prior_best_mean_overlap"] = result.pop("best_mean_overlap")
            result["ranking"] = rank_gammas(result["points"])
            result["ranking_criterion"] = "minimize mean symmetric nearest-neighbor Hamming distance across all paired trials"
            result["best_gammas"] = [row["gamma"] for row in result["ranking"] if row["rank"] == 1]
            result["best_mean_nearest_hamming"] = result["ranking"][0]["mean_nearest_hamming"]
    else:
        _enrich_experiment(result)
    result["selection_comparison"] = hamming_comparison_definition()
    return result


def _write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reevaluate_file(source_path: Path, directory: Path, make_plots: bool = True) -> dict:
    """Read a source once, hash it, and write fresh non-overwriting artifacts."""
    source_path = source_path.resolve()
    raw = source_path.read_bytes()
    result = reevaluate_sls_result(json.loads(raw))
    digest = hashlib.sha256()
    for name in ("metrics.py", "hamming_analysis.py", "hamming_reporting.py", "plotting.py", "exact_plotting.py", "gamma_sweep.py"):
        digest.update(name.encode())
        digest.update(Path(__file__).with_name(name).read_bytes())
    result["reevaluation"] = {
        "source_results_path": str(source_path), "source_results_sha256": hashlib.sha256(raw).hexdigest(),
        "evaluation_source_sha256": digest.hexdigest(), "created_utc": datetime.now(timezone.utc).isoformat(),
        "searches_rerun": False, "selections_changed": False, "logdet_scores_changed": False,
        "history_policy": "large trajectories omitted in this compact copy; source artifacts unchanged",
        "direction": "lower symmetric nearest-neighbor Hamming distance means closer final G1 sets",
    }
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    parameter = None
    if "points" in result:
        parameter = next(key for key in ("gamma", "kmer_length", "client_selection_size") if key in result["points"][0])
        rows = [{parameter: point[parameter], **trial["metrics"]}
                for point in result["points"] for trial in point["experiment"]["trials"]]
        stats = [{parameter: point[parameter], **point["hamming_distance"],
                  "normalized_mean": point["hamming_distance_normalized"]["mean"],
                  "normalized_std": point["hamming_distance_normalized"]["std"]}
                 for point in result["points"]]
        _write_csv(directory / "hamming_distance_summary.csv", stats)
        if parameter == "gamma":
            _write_csv(directory / "gamma_ranking.csv", result["ranking"])
    else:
        rows = [trial["metrics"] for trial in result["trials"]]
    _write_csv(directory / "metrics.csv", rows)
    write_hamming_report(result, directory / "report.md")
    if make_plots:
        if parameter:
            from .plotting import plot_nearest_hamming_sweep, plot_sweep_metrics
            plot_nearest_hamming_sweep(result, directory, parameter)
            if parameter != "gamma":
                plot_sweep_metrics(result, directory, parameter)
        else:
            from .exact_plotting import plot_sls_comparison
            plot_sls_comparison(result, directory)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Re-evaluate saved k-mer SLS G1 sets by symmetric nearest-neighbor Hamming distance")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results", type=Path, nargs="+", help="One or more existing results.json files")
    source.add_argument("--all-saved", action="store_true", help="Inspect results-root/*/results.json; skip non-comparable records")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        paths = sorted(args.results_root.glob("*/results.json")) if args.all_saved else args.results
        if not paths:
            raise ValueError("No saved result files found")
        directory = allocate_run_directory(args.output_dir, "sls_hamming_reanalysis")
        completed, skipped = [], []
        for path in paths:
            target = directory / path.parent.name
            try:
                result = reevaluate_file(path, target, make_plots=not args.no_plots)
            except NotSLSComparison as exc:
                skipped.append({"source": str(path.resolve()), "reason": str(exc)})
                print(f"Skipped {path.parent.name}: {exc}", flush=True)
                continue
            points = result.get("points")
            count = sum(len(point["experiment"]["trials"]) for point in points) if points else len(result["trials"])
            item = {"source": str(path.resolve()), "directory": str(target.resolve()), "paired_trials": count,
                    "points": len(points) if points else 1,
                    "source_sha256": result["reevaluation"]["source_results_sha256"]}
            if not points:
                item["mean_nearest_hamming"] = result["summary"][HAMMING_METRIC]["mean"]
                print(f"Updated {path.parent.name}: {count} trials; mean nearest Hamming={item['mean_nearest_hamming']:.6f}", flush=True)
            else:
                print(f"Updated {path.parent.name}: {len(points)} points, {count} paired trials", flush=True)
            completed.append(item)
        manifest = {"primary_comparison": hamming_comparison_definition(), "completed": completed,
                    "skipped": skipped, "searches_rerun": False}
        with (directory / "manifest.json").open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, allow_nan=False)
        lines = ["# SLS Hamming-distance re-evaluation", "",
                 "Saved selections only; no searches rerun and no original artifacts overwritten. "
                 "The primary comparison now uses symmetric cross-set nearest-neighbor Hamming distance. "
                 "Lower means closer sequences; shared k-mers contribute zero. "
                 "Overlap/Jaccard remain legacy diagnostics. The logdet optimization is unchanged.", "",
                 "| Source run | Parameter points | Paired trials | Updated report |", "|---|---:|---:|---|"]
        for item in completed:
            name = Path(item["source"]).parent.name
            lines.append(f"| {name} | {item['points']} | {item['paired_trials']} | [Report]({name}/report.md) |")
        lines.extend(["", "## Skipped records", ""])
        lines.extend(f"- `{Path(item['source']).parent.name}`: {item['reason']}." for item in skipped)
        with (directory / "report.md").open("x", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        print(f"Re-evaluated {len(completed)} records; skipped {len(skipped)}. Results: {directory}", flush=True)
        return 0 if completed else 1
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
