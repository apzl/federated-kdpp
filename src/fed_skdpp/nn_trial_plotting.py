"""Per-trial global-to-selected distance boxes from an existing NN analysis.

These boxes describe per-global-item distances within one trial, not a
distribution of trial means or independent experimental replications.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .cli import _metadata, allocate_run_directory


def per_trial_distance_stats(result: dict) -> list[dict]:
    """Summarize every global item's distance, with no exclusions or pooling.

    Require a coverage analysis with one nonnegative integer Hamming count per
    original global ID. Selected items must remain at zero. Returns min, max,
    mean, median and quartiles for each method in each trial. Distances and
    selected IDs are not modified. No sampling or distance computation occurs.
    """
    if result.get("metric") != "global_to_subset_mean_nearest_hamming":
        raise ValueError("Per-trial plots require global-to-selected distances, not within-subset distances")
    n = result["data_config"]["global_collection_size"]
    length = result["data_config"]["kmer_length"]
    selection_size = result["selection_size"]
    if (type(n) is not int or n < 1 or type(length) is not int or length < 1
            or type(selection_size) is not int or not 1 <= selection_size <= n):
        raise ValueError("Invalid analysis population, sequence length or selection size")
    trials = result.get("trials", [])
    if not trials or len({trial["trial"] for trial in trials}) != len(trials):
        raise ValueError("Need nonempty trials with unique IDs")
    rows = []
    for trial in trials:
        for method in ("centralized", "federated"):
            entry = trial[method]
            values = np.asarray(entry["nearest_hamming"], dtype=np.float64)
            ids = np.asarray(entry["selected_ids"], dtype=np.int64)
            if (entry["query_population_size"] != n or entry["query_ids"] != list(range(n))
                    or values.shape != (n,) or not np.isfinite(values).all()
                    or np.any(values < 0) or np.any(values > length)
                    or np.any(values != np.floor(values))):
                raise ValueError("Expected one finite integer Hamming distance for EVERY global item")
            if (ids.shape != (selection_size,) or len(np.unique(ids)) != selection_size
                    or np.any(ids < 0) or np.any(ids >= n) or np.any(values[ids] != 0)):
                raise ValueError("Invalid selection IDs or nonzero distances for selected global items")
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            rows.append({"trial": trial["trial"], "trial_display": trial["trial"] + 1,
                         "method": method, "count": n,
                         "min": float(values.min()), "max": float(values.max()),
                         "mean": float(values.mean()), "median": float(median),
                         "q1": float(q1), "q3": float(q3)})
    return rows


def _draw_trial_panels(
    result: dict, groups: list[list[dict]], directory: Path, name: str, *, within_subset: bool = False,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    columns = len(groups)
    fig, axes = plt.subplots(1, columns, figsize=(max(6, 4.7 * columns), 7.8 if within_subset else 7.1),
                             sharey=True, squeeze=False)
    upper = max(row["max"] for group in groups for row in group)
    for ax, group in zip(axes[0], groups):
        boxes = [{"q1": row["q1"], "med": row["median"], "q3": row["q3"],
                  "mean": row["mean"], "whislo": row["min"], "whishi": row["max"],
                  "fliers": []} for row in group]
        artists = ax.bxp(boxes, positions=[1, 2], widths=.48, patch_artist=True,
                         manage_ticks=False, showmeans=True, showfliers=False,
                         meanprops={"marker": "D", "markerfacecolor": "#B63B32",
                                    "markeredgecolor": "#B63B32", "markersize": 6},
                         medianprops={"color": "black", "linewidth": 1.6},
                         whiskerprops={"color": "#333333", "linewidth": 1},
                         capprops={"color": "#333333", "linewidth": 1.3})
        for patch, color in zip(artists["boxes"], ("#5B8DB8", "#D9A15A")):
            patch.set(facecolor=color, alpha=.48)
        if within_subset:
            # Horizontal offsets distinguish tied percentiles; y-values are exact.
            for position, row in enumerate(group, start=1):
                ax.plot(position - .10, row["p05"], marker="v", color="#197A50",
                        markersize=8, markeredgecolor="white", markeredgewidth=.5, zorder=5)
                ax.plot(position + .10, row["p10"], marker="^", color="#7044A3",
                        markersize=8, markeredgecolor="white", markeredgewidth=.5, zorder=5)
        ax.set_title(f"Trial {group[0]['trial_display']}", fontsize=12)
        ax.set_xlim(.5, 2.5)
        ax.set_ylim(-.5, max(1, upper) + .7)
        ax.set_xticks([1, 2], ["Centralized", "Federated"])
        ax.grid(axis="y", alpha=.2)
        stat_fields = ["min", "max", "mean", "median"] + (["p05", "p10"] if within_subset else [])
        table = ax.table(cellText=[
            [f"{row[field]:.4f}" if field == "mean" else f"{row[field]:g}" for row in group]
            for field in stat_fields],
            rowLabels=["Min", "Max", "Mean", "Median"] + (["5th percentile", "10th percentile"] if within_subset else []),
            colLabels=["Centralized", "Federated"], cellLoc="center",
            bbox=[.10, -.69, .87, .56] if within_subset else [.04, -.48, .92, .35])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for cell in table.get_celld().values():
            cell.set_edgecolor("#D6DCE1")
            cell.set_linewidth(.6)
    handles = [Line2D([], [], color="black", linewidth=1.6, label="Median"),
               Line2D([], [], color="#B63B32", marker="D", linestyle="none", label="Mean")]
    if within_subset:
        axes[0, 0].set_ylabel("Within-subset nearest-other Hamming distance\n(mismatching positions; higher = greater separation)")
        title = "Within-subset nearest-neighbor distances: centralized vs federated\n"
        title += (f"{result['selection_size']:,} selected {result['data_config']['kmer_length']}-mers "
                  f"per method; gamma = {result['kernel_config']['gamma']}")
        handles += [Line2D([], [], color="#197A50", marker="v", linestyle="none", label="5th percentile"),
                    Line2D([], [], color="#7044A3", marker="^", linestyle="none", label="10th percentile")]
        caption = ("Each box contains the selected items' nearest-other distances in one trial; self-matches excluded.\n"
                   "Boxes = Q1–Q3; whiskers = actual min–max. Percentile markers offset horizontally when tied.")
    else:
        axes[0, 0].set_ylabel("Hamming distance from a global item\nto its nearest selected item")
        title = ("Global → selected: per-item distance distributions\n"
                 f"{result['data_config']['global_collection_size']:,} global items → "
                 f"{result['selection_size']:,} selected {result['data_config']['kmer_length']}-mers; "
                 f"gamma = {result['kernel_config']['gamma']}")
        caption = ("Each box contains all per-global-item distances from one trial. Lower = better coverage.\n"
                   "Boxes = Q1–Q3; whiskers = actual min–max. Selected items contribute zero.")
    fig.suptitle(title, fontsize=14, y=.98)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .885),
               ncol=len(handles), frameon=False)
    fig.text(.5, .02, caption, ha="center", fontsize=9)
    fig.subplots_adjust(left=.14 if columns == 1 and within_subset else .11 if columns == 1 else .08,
                        right=.98, top=.80, bottom=.43 if within_subset else .39, wspace=.30)
    paths = []
    for suffix in ("png", "svg"):
        path = directory / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def write_per_trial_distance_plots(result: dict, directory: Path) -> tuple[list[Path], list[dict]]:
    """Create combined panels, individual trial figures and exact summary CSV.

    Whiskers use actual min/max, not the default 1.5-IQR convention. A box
    describes a distance population and is not a claim of iid observations.
    The destination must be an existing empty directory.
    """
    rows = per_trial_distance_stats(result)
    if any(directory.iterdir()):
        raise FileExistsError("Refusing to overwrite existing per-trial plots")
    with (directory / "distance_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = [[row for row in rows if row["trial"] == trial["trial"]] for trial in result["trials"]]
    paths = _draw_trial_panels(result, groups, directory, "global_to_subset_nn_by_trial")
    for group in groups:
        paths.extend(_draw_trial_panels(result, [group], directory,
                                       f"trial_{group[0]['trial_display']}_global_to_subset_nn"))
    lines = ["# Per-trial global-to-selected distance distributions", "",
             "For every global item $x\\in S$, use $\\min_{y\\in G1}d_H(x,y)$. "
             "Each method's box in each trial contains all global items' distances, "
             "not trial-average values or distances within the selected set. "
             "Selected global items contribute zero, and lower means better global coverage.", "",
             "Boxes span the 25th–75th percentiles; black lines are medians and red diamonds are means. "
             "Whiskers run to the actual minimum and maximum, not 1.5-IQR cutoffs. "
             "The table below each panel shows min, max, mean and median.", "",
             "![All trials](global_to_subset_nn_by_trial.png)", "",
             "| Trial | Method | Count | Min | Max | Mean | Median |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['trial_display']} | {row['method']} | {row['count']} | "
                     f"{row['min']:g} | {row['max']:g} | {row['mean']:.6f} | {row['median']:g} |")
    lines += ["", "The per-item distances are dependent descriptive observations; "
              "they are not additional independent data trials. Each trial has its own "
              "10,000-item dataset, shared by the two methods within that trial. "
              "No samples or distances were recomputed; the source analysis and prior plots remain unchanged.", ""]
    for group in groups:
        trial = group[0]["trial_display"]
        lines += [f"[Separate Trial {trial} image](trial_{trial}_global_to_subset_nn.png)", ""]
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
    per_trial_distance_stats(result)  # Validate before allocating output.
    for key in ("centralized", "federated"):
        source = Path(result["inputs"][f"{key}_results"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != result["inputs"][f"{key}_sha256"]:
            raise ValueError(f"{key} source no longer matches the saved analysis")
    directory = allocate_run_directory(args.output_dir.resolve(), "global_to_subset_nn_trial_boxes")
    paths, rows = write_per_trial_distance_plots(result, directory)
    metadata = {"analysis_results": str(path), "analysis_sha256": hashlib.sha256(payload).hexdigest(),
                "metric": result["metric"], "sampling_rerun": False, "distances_recomputed": False,
                "whiskers": "actual minimum and maximum", "statistics": rows, "metadata": _metadata()}
    metadata["metadata"]["execution"] = "plotting saved per-global-item distances only"
    (directory / "plot_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Combined plot: {paths[0]}")
    print(f"Statistics: {directory / 'distance_stats.csv'}")
    print(f"Results: {directory}")
    return 0
