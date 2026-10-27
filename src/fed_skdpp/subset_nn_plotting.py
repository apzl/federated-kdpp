"""Per-trial within-subset NN boxes with lower-tail separation percentiles.

Plot cached per-selected-item distances only: no sampling, distance calculation,
or change to the context-free centralized/distributed baseline is performed.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .cli import _metadata, allocate_run_directory
from .nn_trial_plotting import _draw_trial_panels


def per_trial_subset_distance_stats(result: dict) -> list[dict]:
    """Summarize raw nearest-other distances within each saved size-b subset.

    Require one finite integer Hamming count per selected ID, with self-matches
    already excluded by the source analysis. Validate IDs and optional query
    metadata without requiring fields absent from older analyses. Distinct items
    with identical features may have zero distance. Percentiles use NumPy's
    explicit linear interpolation. No inputs are changed or recomputed.
    """
    if result.get("metric") != "within_subset_mean_nearest_hamming":
        raise ValueError("Require within-subset nearest-other distances, not global-to-selected coverage")
    n = result["data_config"]["global_collection_size"]
    length = result["data_config"]["kmer_length"]
    size = result["selection_size"]
    if (type(n) is not int or n < 2 or type(length) is not int or length < 1
            or type(size) is not int or not 2 <= size <= n):
        raise ValueError("Invalid population, sequence length or within-subset size")
    trials = result.get("trials", [])
    if (not trials or any(type(t["trial"]) is not int or t["trial"] < 0 for t in trials)
            or len({t["trial"] for t in trials}) != len(trials)):
        raise ValueError("Need nonempty trials with unique nonnegative integer IDs")
    rows = []
    for trial in trials:
        for method in ("centralized", "federated"):
            entry = trial[method]
            ids = np.asarray(entry["selected_ids"])
            values = np.asarray(entry["nearest_hamming"], dtype=np.float64)
            if (ids.shape != (size,) or ids.dtype.kind not in "iu"
                    or len(np.unique(ids)) != size or np.any(ids < 0) or np.any(ids >= n)):
                raise ValueError("Expected unique integer selected IDs within the global collection")
            if (values.shape != (size,) or not np.isfinite(values).all()
                    or np.any(values < 0) or np.any(values > length)
                    or np.any(values != np.floor(values))):
                raise ValueError("Expected one finite integer Hamming distance per selected item")
            if (entry.get("query_population_size", size) != size
                    or entry.get("query_ids", ids.tolist()) != ids.tolist()):
                raise ValueError("Within-subset queries must be the selected items, in the saved order")
            p05, p10, q1, median, q3 = np.percentile(values, [5, 10, 25, 50, 75], method="linear")
            rows.append({"trial": trial["trial"], "trial_display": trial["trial"] + 1,
                         "method": method, "count": size,
                         "min": float(values.min()), "max": float(values.max()),
                         "mean": float(values.mean()), "median": float(median),
                         "q1": float(q1), "q3": float(q3), "p05": float(p05), "p10": float(p10)})
    return rows


def write_subset_distance_plots(result: dict, directory: Path) -> tuple[list[Path], list[dict]]:
    """Write paired per-trial boxes, CSVs and report into an empty directory.

    Boxes describe selected-item distances, not independent trial replications.
    Whiskers show actual extrema. Higher lower-tail percentiles mean fewer
    very close neighbors at that quantile, not guaranteed optimal diversity.
    """
    rows = per_trial_subset_distance_stats(result)
    if any(directory.iterdir()):
        raise FileExistsError("Refusing to overwrite existing within-subset plots")
    with (directory / "distance_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = [[row for row in rows if row["trial"] == t["trial"]] for t in result["trials"]]
    comparisons = [{"trial": c["trial"], "trial_display": c["trial_display"],
                    **{f"{key}_centralized_minus_federated": c[key] - f[key]
                       for key in ("mean", "median", "p05", "p10")}} for c, f in groups]
    with (directory / "paired_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    paths = _draw_trial_panels(result, groups, directory, "within_subset_nn_by_trial", within_subset=True)
    for group in groups:
        paths.extend(_draw_trial_panels(result, [group], directory,
                                       f"trial_{group[0]['trial_display']}_within_subset_nn", within_subset=True))
    lines = ["# Within-subset nearest-neighbor separation: centralized vs federated", "",
             "For each selected item $x\\in G1$, the distance is "
             "$\\min_{y\\in G1\\setminus\\{x\\}} d_H(x,y)$. Self-matches are excluded. "
             "This is not the distance from the full global collection to a subset. "
             "Each trial pairs the two methods on the same global dataset. "
             "These are the existing context-free, non-private baseline draws.", "",
             "Each box contains all selected items' distances for one method in one trial. "
             "Boxes span Q1–Q3, whiskers show the actual min–max, black lines are medians, "
             "and red diamonds are means. Green downward triangles mark the 5th percentile; "
             "purple upward triangles mark the 10th percentile. Percentile markers have "
             "horizontal offsets so ties are visible. All distances are raw mismatch counts. "
             "Percentiles use NumPy's linear interpolation, including for discrete distances.", "",
             "Higher means and lower-tail percentiles indicate greater nearest-neighbor "
             "separation at the same subset size; they are descriptive metrics, not a guarantee "
             "of optimal diversity. Dependent per-item distances are not independent data trials.", "",
             "![All paired trials](within_subset_nn_by_trial.png)", "",
             "| Trial | Method | Count | Min | Max | Mean | Median | 5th percentile | 10th percentile |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['trial_display']} | {row['method']} | {row['count']} | "
                     f"{row['min']:g} | {row['max']:g} | {row['mean']:.6f} | {row['median']:g} | "
                     f"{row['p05']:g} | {row['p10']:g} |")
    lines += ["", "## Paired differences (centralized minus federated)", "",
              "Positive values favor centralized nearest-neighbor separation.", "",
              "| Trial | Mean gap | Median gap | 5th percentile gap | 10th percentile gap |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {row['trial_display']} | {row['mean_centralized_minus_federated']:.6f} | "
                     f"{row['median_centralized_minus_federated']:g} | "
                     f"{row['p05_centralized_minus_federated']:g} | "
                     f"{row['p10_centralized_minus_federated']:g} |")
    lines += ["", "No samples or distances were recomputed. Original analyses and previous plots remain unchanged.", ""]
    for group in groups:
        trial = group[0]["trial_display"]
        lines += [f"[Separate Trial {trial} image](trial_{trial}_within_subset_nn.png)", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return paths, rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-results", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("results"), type=Path)
    args = parser.parse_args(argv)
    path = args.analysis_results.resolve()
    payload = path.read_bytes()
    result = json.loads(payload)
    per_trial_subset_distance_stats(result)
    for key in ("centralized", "federated"):
        source = Path(result["inputs"][f"{key}_results"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != result["inputs"][f"{key}_sha256"]:
            raise ValueError(f"{key} source no longer matches the saved analysis")
    directory = allocate_run_directory(args.output_dir.resolve(), "within_subset_nn_percentiles")
    paths, rows = write_subset_distance_plots(result, directory)
    metadata = {"analysis_results": str(path), "analysis_sha256": hashlib.sha256(payload).hexdigest(),
                "inputs": result["inputs"], "metric": result["metric"], "sampling_rerun": False,
                "distances_recomputed": False, "self_matches": "excluded in source analysis",
                "whiskers": "actual minimum and maximum", "percentile_method": "linear",
                "percentiles": [5, 10], "statistics": rows, "metadata": _metadata()}
    metadata["metadata"]["execution"] = "plotting saved within-subset nearest-other distances only"
    (directory / "plot_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Combined plot: {paths[0]}")
    print(f"Statistics: {directory / 'distance_stats.csv'}")
    print(f"Results: {directory}")
    return 0
