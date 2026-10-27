"""Explicitly directed nearest-neighbor diagnostics: coverage or separation.

Analyze existing context-free centralized/distributed draws without resampling
or changing their target, kernel, logdet objective or saved experiment files.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .centralized_pool import read_distributed_source
from .cli import _metadata, allocate_run_directory
from .metrics import scalar_summary
from .variable_dpp import selected_indices


def within_subset_nearest_hamming(encoded: np.ndarray) -> np.ndarray:
    """Return exact int64 (n,) nearest OTHER-row raw Hamming distances.

    Input has shape (n>=2, positive sequence length), with finite numeric or
    string symbols. Self means the same row index, not identical features:
    distinct duplicate rows correctly have distance zero. No distance to the
    full dataset, kernel transformation, optimization or random draws occur.
    Uses an (n,n) integer work matrix and excludes the diagonal explicitly.
    """
    encoded = np.asarray(encoded)
    if encoded.ndim != 2 or len(encoded) < 2 or encoded.shape[1] < 1:
        raise ValueError("Within-subset nearest neighbors require at least two nonempty sequences")
    if encoded.dtype.kind not in "biufUS":
        raise ValueError("Hamming symbols must be finite numeric values or strings")
    if encoded.dtype.kind in "biuf" and not np.isfinite(encoded).all():
        raise ValueError("Hamming symbols must be finite")
    n, length = encoded.shape
    distances = np.zeros((n, n), dtype=np.int64)
    for position in range(length):
        distances += encoded[:, position, None] != encoded[None, :, position]
    np.fill_diagonal(distances, length + 1)
    return distances.min(axis=1)


def global_to_subset_nearest_hamming(
    global_items: np.ndarray, selected_items: np.ndarray, batch_size: int = 512,
) -> np.ndarray:
    """Return raw int64 (N,) distances from GLOBAL rows to nearest selected row.

    Inputs are nonempty (N,length) and (b,length) arrays of finite numeric or
    string symbols. Include exact matches: selected items in the global set
    correctly receive zero. This is NOT nearest-other-selected separation,
    the reverse direction, or a symmetric distance. Lower mean means better
    global-dataset coverage. Chunked work arrays have at most batch_size*b
    entries; no full global N*N distance matrix is materialized.
    """
    global_items, selected_items = np.asarray(global_items), np.asarray(selected_items)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    for values in (global_items, selected_items):
        if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
            raise ValueError("Global and selected items must be nonempty sequence arrays")
        if values.dtype.kind not in "biufUS":
            raise ValueError("Hamming symbols must be finite numeric values or strings")
        if values.dtype.kind in "biuf" and not np.isfinite(values).all():
            raise ValueError("Hamming symbols must be finite")
    if global_items.shape[1] != selected_items.shape[1]:
        raise ValueError("Global and selected sequence lengths must match")
    nearest = np.empty(len(global_items), dtype=np.int64)
    for start in range(0, len(global_items), batch_size):
        query = global_items[start:start + batch_size]
        distances = np.zeros((len(query), len(selected_items)), dtype=np.int64)
        for position in range(global_items.shape[1]):
            distances += query[:, position, None] != selected_items[None, :, position]
        nearest[start:start + len(query)] = distances.min(axis=1)
    return nearest


def analyze_saved_subsets(
    path: str | Path, selection_size: int = 1000, *, metric: str = "within_subset",
) -> dict:
    """Compare trial MEANS at one shared size; validate data/source provenance.

    Report one independent data-trial average per method, not n independent
    replicates per method. SciPy independently checks every per-item distance.
    Missing sizes fail explicitly rather than silently dropping paired trials.
    """
    from scipy.spatial.distance import cdist

    if metric not in {"within_subset", "global_to_subset"}:
        raise ValueError("metric must be within_subset or global_to_subset")
    coverage = metric == "global_to_subset"
    if type(selection_size) is not int or selection_size < (1 if coverage else 2):
        raise ValueError("selection_size must be at least one for coverage or at least two within a subset")
    path = Path(path).resolve()
    payload = path.read_bytes()
    central = json.loads(payload)
    if central.get("schema") != "centralized_saved_global_pool_exact_size_sweep_v1":
        raise ValueError("Expected a saved centralized global-pool sweep")
    source, digest = read_distributed_source(central["source"]["results"])
    if digest != central["source"]["sha256"]:
        raise ValueError("Distributed source hash no longer matches the centralized run")
    if central["inherited_config"] != {key: source["config"][key] for key in ("data", "kernel", "numerics")}:
        raise ValueError("Centralized and distributed configurations do not match")
    data = source["config"]["data"]
    federated = {trial["trial"]: trial for trial in source["trials"]}
    if (len({trial["trial"] for trial in central["trials"]}) != len(central["trials"])
            or {trial["trial"] for trial in central["trials"]} != set(federated)):
        raise ValueError("Centralized and federated trial populations do not match")
    trials = []
    for ct in central["trials"]:
        ft = federated[ct["trial"]]
        if (ct["global_collection"] != ft["global_collection"]
                or ct["collection_sha256"] != ft["collection_sha256"]):
            raise ValueError("Paired trial datasets or global-ID orders differ")
        encoded = np.asarray([[data["alphabet"].index(symbol) for symbol in word]
                              for word in ct["global_collection"]], dtype=np.int64)
        row = {"trial": ct["trial"], "collection_sha256": ct["collection_sha256"]}
        for method, saved in (("centralized", ct), ("federated", ft)):
            records = [r for r in saved["sweep"]["records"] if r["selection_size"] == selection_size]
            if len(records) != 1:
                raise ValueError(f"Trial {ct['trial']} / {method} has no unique size-{selection_size} draw")
            ids = selected_indices(records[0], saved["sweep"]["permutations"])
            if method == "federated":
                ids = np.asarray(ft["pool_ids"], dtype=np.int64)[ids]
            if (len(ids) != selection_size or len(np.unique(ids)) != len(ids)
                    or np.any(ids < 0) or np.any(ids >= len(encoded))):
                raise ValueError("Saved sample violates cardinality, uniqueness or global-ID bounds")
            query_ids = np.arange(len(encoded)) if coverage else ids
            if coverage:
                nearest = global_to_subset_nearest_hamming(encoded, encoded[ids])
                # Independent batched implementation; DO NOT exclude selected
                # global rows or mask any diagonal in the coverage calculation.
                reference = np.empty(len(encoded), dtype=np.int64)
                for start in range(0, len(encoded), 512):
                    distances = np.rint(cdist(encoded[start:start + 512], encoded[ids],
                                              metric="hamming") * data["kmer_length"])
                    reference[start:start + len(distances)] = distances.min(axis=1).astype(np.int64)
                if not np.all(nearest[ids] == 0):
                    raise ValueError("Selected global items must have zero coverage distance")
            else:
                nearest = within_subset_nearest_hamming(encoded[ids])
                distances = np.rint(cdist(encoded[ids], encoded[ids], metric="hamming") * data["kmer_length"])
                np.fill_diagonal(distances, np.inf)
                reference = distances.min(axis=1)
            if not np.array_equal(nearest, reference):
                raise ValueError("Independent SciPy nearest-neighbor distance check failed")
            row[method] = {
                "selected_ids": ids.tolist(), "query_ids": query_ids.tolist(),
                "query_population_size": len(query_ids), "nearest_hamming": nearest.tolist(),
                "mean_nearest_hamming": float(nearest.mean()),
                "normalized_mean_nearest_hamming": float(nearest.mean() / data["kmer_length"]),
                "independent_distance_check_passed": True,
            }
        row["gap_centralized_minus_federated"] = (row["centralized"]["mean_nearest_hamming"]
                                                   - row["federated"]["mean_nearest_hamming"])
        trials.append(row)
    summary = {method: scalar_summary([trial[method]["mean_nearest_hamming"] for trial in trials])
               for method in ("centralized", "federated")}
    summary["gap_centralized_minus_federated"] = scalar_summary(
        [trial["gap_centralized_minus_federated"] for trial in trials])
    return {
        "metric": "global_to_subset_mean_nearest_hamming" if coverage else "within_subset_mean_nearest_hamming",
        "selection_size": selection_size,
        "definition": ("mean over EVERY x in global S of min over y in selected Y of d_H(x,y)" if coverage
                       else "mean over x in selected Y of min over other y in Y of d_H(x,y)"),
        "direction": ("lower means better global-dataset coverage" if coverage
                      else "higher means greater local nearest-neighbor separation"),
        "units": ("raw mismatching positions; selected global items included with zero distance" if coverage
                  else "raw number of mismatching positions; self-matches excluded"),
        "boxplot_unit": "one mean per paired data trial, not individual items as independent replicates",
        "data_config": data, "kernel_config": source["config"]["kernel"],
        "resampling": False,
        "inputs": {"centralized_results": str(path), "centralized_sha256": hashlib.sha256(payload).hexdigest(),
                   "federated_results": central["source"]["results"], "federated_sha256": digest},
        "trials": trials, "summary": summary,
    }


def write_subset_nn_analysis(result: dict, directory: Path) -> list[Path]:
    """Write a fresh analysis, paired-trial mean boxplot, CSV data and report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    if any(directory.iterdir()):
        raise FileExistsError("Refusing to overwrite existing nearest-neighbor artifacts")
    coverage = result["metric"] == "global_to_subset_mean_nearest_hamming"
    plot_name = "global_to_subset_nn_boxplot" if coverage else "within_subset_nn_boxplot"
    with (directory / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with (directory / "trial_means.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["trial", "centralized", "federated", "gap_centralized_minus_federated"])
        writer.writeheader()
        for trial in result["trials"]:
            writer.writerow({"trial": trial["trial"],
                             **{m: trial[m]["mean_nearest_hamming"] for m in ("centralized", "federated")},
                             "gap_centralized_minus_federated": trial["gap_centralized_minus_federated"]})
    with (directory / "per_item_distances.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "method", "global_id",
                         "nearest_selected_item_hamming" if coverage else "nearest_other_selected_item_hamming"])
        for trial in result["trials"]:
            for method in ("centralized", "federated"):
                for item, distance in zip(trial[method]["query_ids"], trial[method]["nearest_hamming"]):
                    writer.writerow([trial["trial"], method, item, distance])
    fig, ax = plt.subplots(figsize=(8, 6))
    methods = ("centralized", "federated")
    boxes = ax.boxplot([[t[method]["mean_nearest_hamming"] for t in result["trials"]] for method in methods],
                      positions=[1, 2], widths=.42, patch_artist=True, manage_ticks=False,
                      showfliers=False, medianprops={"color": "black", "linewidth": 1.5})
    for patch, color in zip(boxes["boxes"], ("#5B8DB8", "#D9A15A")):
        patch.set(facecolor=color, alpha=.45)
    for index, trial in enumerate(result["trials"]):
        values = [trial[method]["mean_nearest_hamming"] for method in methods]
        ax.plot([1, 2], values, color="0.65", linewidth=.9, alpha=.7, zorder=2)
        ax.scatter([1, 2], values, color="#263747", s=44, marker=("o", "s", "^")[index % 3],
                   zorder=3, label=f"Trial {trial['trial'] + 1}")
    ax.set_xticks([1, 2], ["Centralized", "Federated"])
    ax.set_xlim(.55, 2.45)
    ax.set_ylabel("Trial-average Hamming distance\n(global item → nearest selected item)" if coverage else
                  "Trial-average nearest-neighbor Hamming distance\n(mismatching positions; self excluded)")
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    heading = "Global → selected: nearest-neighbor coverage" if coverage else "Within-subset nearest-neighbor diversity"
    population = f"{result['data_config']['global_collection_size']:,} global → " if coverage else ""
    ax.set_title(f"{heading}\n{population}"
                 f"{result['selection_size']:,} selected {result['data_config']['kmer_length']}-mers; "
                 f"gamma = {result['kernel_config']['gamma']}")
    ax.grid(axis="y", alpha=.2)
    ax.legend(title="Paired data trials", fontsize="small", loc="best")
    meaning = ("Lower = better global coverage; selected items contribute zero." if coverage
               else "Higher = greater local separation.")
    fig.text(.5, .035, f"Each point is one trial mean; n = {len(result['trials'])} trials per method.\n"
             f"{meaning} Boxes: median/IQR; whiskers: 1.5 IQR.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .1, 1, 1))
    paths = []
    for suffix in ("png", "svg"):
        target = directory / f"{plot_name}.{suffix}"
        fig.savefig(target, dpi=180, bbox_inches="tight")
        paths.append(target)
    plt.close(fig)
    description = ("For each global item $x\\in S$, find its nearest selected item $y\\in Y$ and compute "
                   "$|S|^{-1}\\sum_{x\\in S}\\min_{y\\in Y}d_H(x,y)$. "
                   "Average over every global item, not just the selected items. "
                   "Selected global items contribute zero; they are not removed or self-masked. "
                   "Lower values mean better representation of the global dataset. "
                   "This is the requested global-to-selected measurement. The earlier within-subset plot "
                   "answers a different question and must not be used for this comparison."
                   if coverage else
                   "For each saved selected set $Y$, compute "
                   "$|Y|^{-1}\\sum_{x\\in Y}\\min_{y\\in Y\\setminus\\{x\\}} d_H(x,y)$. "
                   "Distances are raw mismatch counts, not RBF distances. Self-matches are excluded. "
                   "Higher values indicate greater local nearest-neighbor separation, not a guarantee "
                   "of higher logdet or better full-dataset coverage.")
    lines = [f"# {heading}", "",
             f"Selection size {result['selection_size']}; {result['data_config']['kmer_length']}-mers; "
             f"Hamming RBF sampling gamma {result['kernel_config']['gamma']}.", "",
             description, "",
             "Each box contains one average per data trial. Per-item distances are saved "
             "for inspection but are not treated as independent experimental replicates. "
             "Each trial used a fresh global dataset and fresh client partition; centralized "
             "and federated methods share exactly the same dataset within a trial. "
             "Datasets and local pools were held fixed across selection sizes within each source trial. "
             "No sampling or search was rerun.", "",
             f"![Trial-average boxplot]({plot_name}.png)", "",
             "| Trial | Centralized | Federated | Centralized minus federated |",
             "| --- | ---: | ---: | ---: |"]
    for trial in result["trials"]:
        lines.append(f"| {trial['trial'] + 1} | {trial['centralized']['mean_nearest_hamming']:.6f} | "
                     f"{trial['federated']['mean_nearest_hamming']:.6f} | {trial['gap_centralized_minus_federated']:+.6f} |")
    for method in methods:
        stats = result["summary"][method]
        lines += ["", f"{method.capitalize()}: mean {stats['mean']:.6f}; sample SD {stats['std']:.6f}."]
    lines += ["", "Only three data trials are available in the original experiment; "
              "this is a descriptive comparison, not a statistical significance or optimality claim. "
              "Every per-item distance was independently verified using SciPy. "
              "Dataset pairing and input-file hashes were checked. CSV/JSON trial IDs are zero-based; "
              "plot/report trial labels are one-based.", "",
              f"Centralized inputs: `{result['inputs']['centralized_results']}`", "",
              f"Federated inputs: `{result['inputs']['federated_results']}`", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--centralized-results", type=Path, required=True)
    parser.add_argument("--selection-size", type=int, default=1000)
    parser.add_argument("--metric", choices=("global_to_subset", "within_subset"), default="global_to_subset",
                        help="Default: global-to-selected coverage; within_subset is a different diagnostic")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    result = analyze_saved_subsets(args.centralized_results, args.selection_size, metric=args.metric)
    result["metadata"] = _metadata()
    result["metadata"]["execution"] = "saved-subset evaluation only; no resampling or optimizer changes"
    directory = allocate_run_directory(args.output_dir.resolve(),
                                       "global_to_subset_nn" if args.metric == "global_to_subset" else "within_subset_nn")
    write_subset_nn_analysis(result, directory)
    print(json.dumps(result["summary"], indent=2))
    print(f"Plot: {directory / ('global_to_subset_nn_boxplot.png' if args.metric == 'global_to_subset' else 'within_subset_nn_boxplot.png')}")
    print(f"Results: {directory}")
    return 0
