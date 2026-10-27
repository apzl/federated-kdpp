"""Reports of cross-set Hamming agreement, not changes to SLS optimization."""

from pathlib import Path

from .metrics import HAMMING_METRIC, NORMALIZED_HAMMING_METRIC


def write_hamming_report(result: dict, path: Path) -> None:
    """Write an exclusive Markdown report for a run or parameter sweep."""
    config = result.get("base_config", result.get("config"))
    lines = ["# SLS comparison by nearest-neighbor Hamming distance", "",
             "The primary comparison is symmetric cross-set nearest-neighbor Hamming distance. "
             "Lower means more similar final selections, not necessarily better diversity or utility.", "",
             "For centralized $A=G1_c$ and federated $B=G1_f$:", "",
             "$$D(A,B)=\\frac12\\left(\\frac1{|A|}\\sum_{a\\in A}\\min_{b\\in B}d_H(a,b)"
             "+\\frac1{|B|}\\sum_{b\\in B}\\min_{a\\in A}d_H(a,b)\\right).$$", "",
             "Hamming distance counts mismatching positions. Exact matches contribute zero; "
             "multiple items may choose the same nearest neighbor. This is not a one-to-one "
             "assignment and is not guaranteed to satisfy the triangle inequality. "
             "Fixed context items are excluded. Normalized distance divides by k-mer length.", "",
             "The SLS kernel, objective, proposal mechanism, initialization and stopping rule "
             "are unchanged. SLS still optimizes logdet. Overlap/Jaccard remain legacy diagnostics, "
             "not the primary comparison or gamma-ranking criterion.", ""]
    if "reevaluation" in result:
        audit = result["reevaluation"]
        lines.extend(["## Re-evaluation of saved selections", "",
                      f"Source: `{audit['source_results_path']}`.", "",
                      f"Source SHA-256: `{audit['source_results_sha256']}`.", "",
                      "No searches were rerun. Saved data, selections, logdet scores, seeds, budgets "
                      "and measured search runtimes are retained. Large trajectories are omitted "
                      "from this evaluation-only copy; the source artifacts are unchanged.", ""])
    pooled = config["selection"].get("final_objective") == "pooled_context"
    lines.extend(["## Setting", "",
                  f"Public collection: {config['data']['global_collection_size']} items; "
                  f"clients: {config['partition']['num_clients']}; "
                  f"final G1 size: {config['selection']['final_selection_size']}; "
                  f"trials per point: {config['experiment']['repetitions']}.", "",
                  f"Base k-mer length: {config['data']['kmer_length']}; "
                  f"base gamma: {config['kernel']['gamma']}; local G1 size: {config['selection']['client_selection_size']}; "
                  f"fixed context per client: {config['selection'].get('forced_set_size', 0)}. "
                  "A swept parameter overrides its base value at the corresponding point.", "",
                  "Non-private pooled-context baseline: client contexts are disclosed to the server."
                  if pooled else "Two-stage suggestion baseline; no privacy guarantee is established.", ""])
    if "points" in result:
        parameter = next(key for key in ("gamma", "kmer_length", "client_selection_size") if key in result["points"][0])
        lines.extend(["## Primary comparison", "",
                      f"| {parameter} | Mean nearest Hamming | Sample SD | Normalized mean | Normalized SD | Trials |",
                      "|---:|---:|---:|---:|---:|---:|"])
        for point in result["points"]:
            raw, normalized = point["hamming_distance"], point["hamming_distance_normalized"]
            lines.append(f"| {point[parameter]} | {raw['mean']:.6f} | {raw['std']:.6f} | "
                         f"{normalized['mean']:.6f} | {normalized['std']:.6f} | {raw['num_trials']} |")
        if parameter == "kmer_length":
            lines.extend(["", "Raw distances have a different maximum at each length. The normalized "
                          "column and second plot divide by that length; neither scale alone removes "
                          "all changes to the underlying sequence space or sampling problem."])
        if parameter == "gamma":
            lines.extend(["", f"Lowest observed mean distance: {result['best_mean_nearest_hamming']:.6f}; "
                          f"gamma value(s): {', '.join(str(value) for value in result['best_gammas'])}.", "",
                          "Rankings use the mean over every paired trial; exact mean ties share a rank. "
                          "This is tuning for sequence agreement with a heuristic reference, not an "
                          "optimal-diversity bandwidth or held-out validation.", "",
                          "| Rank | Gamma | Mean nearest Hamming | Sample SD |", "|---:|---:|---:|---:|"])
            for row in result["ranking"]:
                lines.append(f"| {row['rank']} | {row['gamma']} | {row['mean_nearest_hamming']:.6f} | {row['std_nearest_hamming']:.6f} |")
    else:
        lines.extend(["## Paired trials", "",
                      "| Trial | Central to fed | Fed to central | Symmetric distance | Normalized | Central logdet | Fed logdet |",
                      "|---:|---:|---:|---:|---:|---:|---:|"])
        for trial in result["trials"]:
            row = trial["metrics"]
            lines.append(f"| {trial['trial'] + 1} | {row['centralized_to_federated_mean_nearest_hamming']:.6f} | "
                         f"{row['federated_to_centralized_mean_nearest_hamming']:.6f} | {row[HAMMING_METRIC]:.6f} | "
                         f"{row[NORMALIZED_HAMMING_METRIC]:.6f} | {row['centralized_logdet']:.6f} | {row['federated_logdet']:.6f} |")
        raw = result["summary"][HAMMING_METRIC]
        normalized = result["summary"][NORMALIZED_HAMMING_METRIC]
        lines.extend(["", f"Mean raw distance: {raw['mean']:.6f} ± {raw['std']:.6f} sample SD; "
                      f"normalized: {normalized['mean']:.6f} ± {normalized['std']:.6f}."])
    lines.extend(["", "Dispersion is sample standard deviation over paired trials, not a confidence "
                  "interval. This evaluation does not establish a MAP optimum, sampling correctness, "
                  "or privacy. Original membership metrics and any pool-retention analysis are "
                  "diagnostically different from sequence distance.", ""])
    with path.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
