"""Centralized size-sweep scores and data-paired distributed comparisons."""

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .metrics import scalar_summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def centralized_summaries(result: dict) -> tuple[list[dict], list[dict]]:
    """Sample SD across saved data trials; pair gaps only at shared sizes."""
    central, gaps = {}, {}
    for trial in result["trials"]:
        for row in trial["sweep"]["records"]:
            central.setdefault(row["selection_size"], []).append(row["logdet"])
        for row in trial["comparison"]:
            gaps.setdefault(row["selection_size"], []).append(row["logdet_gap_central_minus_distributed"])
    return ([{"selection_size": size, **scalar_summary(values)} for size, values in sorted(central.items())],
            [{"selection_size": size, **scalar_summary(values)} for size, values in sorted(gaps.items())])


def write_centralized_pool_results(result: dict, directory: Path) -> None:
    """Persist original global IDs through compact permutation/slice records."""
    if any(directory.iterdir()):
        raise FileExistsError("Refusing to overwrite centralized experiment artifacts")
    with (directory / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with (directory / "config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result["config"], handle, sort_keys=False)
    with (directory / "inherited_config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result["inherited_config"], handle, sort_keys=False)
    central, comparison = [], []
    for trial in result["trials"]:
        central.extend({"trial": trial["trial"], "selection_size": row["selection_size"],
                        "logdet": row["logdet"], "proposal_round": row["proposal_round"]}
                       for row in trial["sweep"]["records"])
        comparison.extend({"trial": trial["trial"], **row} for row in trial["comparison"])
    _write_csv(directory / "centralized_logdet.csv", central)
    _write_csv(directory / "comparison.csv", comparison)
    summary, gap_summary = centralized_summaries(result)
    _write_csv(directory / "centralized_summary.csv", summary)
    _write_csv(directory / "comparison_summary.csv", gap_summary)


def plot_centralized_pool(result: dict, directory: Path) -> list[Path]:
    """All 999 sizes per trial; comparisons stop where distributed sizes stop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    trials = result["trials"]
    colors = plt.cm.tab10(np.linspace(0, .9, len(trials)))

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    inherited = result["inherited_config"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for trial, color in zip(trials, colors):
        rows = trial["sweep"]["records"]
        ax.plot([row["selection_size"] for row in rows], [row["logdet"] for row in rows],
                color=color, lw=1.2, label=f"Trial {trial['trial'] + 1}")
    ax.set(xlabel="Centralized subset size k", ylabel="logdet of centralized subset",
           title=f"Exact centralized k-DPP on all {inherited['data']['global_collection_size']:,} items\n"
                 f"{inherited['data']['kmer_length']}-mers / Hamming RBF gamma={inherited['kernel']['gamma']}")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, "centralized_logdet")

    fig, axes = plt.subplots(len(trials), 2, figsize=(14, max(4, 3.4 * len(trials))), squeeze=False)
    for trial, color, pair in zip(trials, colors, axes):
        central = trial["sweep"]["records"]
        common = trial["comparison"]
        pair[0].plot([r["selection_size"] for r in central], [r["logdet"] for r in central],
                     color=color, lw=1.3, label="Centralized: all global items")
        pair[0].plot([r["selection_size"] for r in common], [r["distributed_logdet"] for r in common],
                     color="black", lw=1.1, ls="--", label=f"Distributed: pool m={trial['distributed_pool_size']}")
        pair[0].set(xlabel="Subset size k", ylabel="Subset logdet", title=f"Trial {trial['trial'] + 1}: identical global data")
        pair[0].grid(alpha=.2)
        pair[0].legend(fontsize="small")
        pair[1].plot([r["selection_size"] for r in common],
                     [r["logdet_gap_central_minus_distributed"] for r in common], color=color, lw=1)
        pair[1].axhline(0, color="black", ls="--", lw=.8)
        pair[1].set(xlabel="Shared subset size k", ylabel="Centralized − distributed logdet",
                    title="Positive favors centralized draw; no optimality guarantee")
        pair[1].grid(alpha=.2)
    save(fig, "centralized_vs_distributed")
    return paths


def write_centralized_pool_report(result: dict, directory: Path) -> Path:
    """Describe the exact law, source binding, independent streams and caveats."""
    inherited, config = result["inherited_config"], result["config"]
    sizes = config["sweep"]
    lines = ["# Centralized k-DPP on the original global collection", "",
             f"Reused all {len(result['trials'])} saved distributed datasets, each with "
             f"{inherited['data']['global_collection_size']:,} unique {inherited['data']['kmer_length']}-mers. "
             "The candidate universe is the full collection, **not** the merged local selections. "
             "Original word order/global IDs and SHA-256 hashes are verified; no data or partitions are regenerated.", "",
             f"Sweep: {sizes['minimum_selection_size']} through {sizes['maximum_selection_size']}, "
             f"step {sizes['selection_size_interval']}; one exact draw per size per source trial. "
             f"Kernel: $L_{{xy}}=\\exp(-{inherited['kernel']['gamma']}d_H(x,y))$, "
             f"raw mismatch counts, unit quality and explicit jitter {inherited['kernel']['jitter']}. "
             "Kernel and numerical choices are inherited from the distributed source.", "",
             "## Target and sampler", "",
             "The centralized law is $P(Y\\mid|Y|=k)=\\det(L_Y)/e_k(\\lambda(L_S))$ "
             "over all saved global items. In contrast, the distributed server conditions on its "
             "random local-suggestion pool. This is a context-free, non-private centralized reference: "
             "all private contexts are empty and there is no optimization, encryption, SLS or MCMC.", "",
             "Reuse the tested exact rejection extension from `variable_dpp.py`, not DPPy's "
             "spectral API. For unit-diagonal kernels, Hadamard's inequality gives "
             "$\\det(L_Y)\\leq1$. A uniform size-$k$ proposal accepted with probability "
             "$\\det(L_Y)$ has exactly the required determinant-weighted law in ideal arithmetic. "
             "Prefix Cholesky factors and shared proposal permutations/uniform thresholds amortize "
             "the sweep. Each size returns its first acceptance, with no rejection-budget cutoff. "
             "Draws across sizes are correlated and not necessarily nested; different data trials "
             "use separate streams. This run's requested sizes are at most half the universe, "
             "so only the direct rejection branch is needed.", "",
             f"Centralized RNG seed {config['experiment']['seed']}, "
             "`SeedSequence([seed, source_trial_id, 4, 0])`; this is a different stream from "
             "the distributed server. Pairing refers to the exact same datasets, not matching "
             "sample IDs or shared initial states. Exactness in ideal arithmetic is distinct "
             "from recorded float64 errors. No eigenvalue clipping, low-rank approximation or implicit jitter.", "",
             "![Centralized logdet](centralized_logdet.png)", "",
             "![Paired comparison](centralized_vs_distributed.png)", "",
             "## Per-trial endpoints", "",
             "| Trial | Dataset items | Sizes sampled | Final size | Final logdet | Seconds |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for trial in result["trials"]:
        rows = trial["sweep"]["records"]
        lines.append(f"| {trial['trial'] + 1} | {trial['population_size']} | {len(rows)} | "
                     f"{rows[-1]['selection_size']} | {rows[-1]['logdet']:.9g} | {trial['total_seconds']:.1f} |")
    summary, _ = centralized_summaries(result)
    by_size = {row["selection_size"]: row for row in summary}
    checkpoints = [size for size in (10, 100, 500, 1000, 2500, 5000) if size in by_size]
    lines += ["", "## Centralized logdet across trials", "",
              "| Selection size | Mean logdet | Sample SD | Trials |", "| ---: | ---: | ---: | ---: |"]
    for size in checkpoints:
        row = by_size[size]
        lines.append(f"| {size} | {row['mean']:.9g} | {row['std']:.6g} | {row['num_trials']} |")
    lines += ["", "## Interpretation", "",
              "Logdet is a sampling diagnostic, not a monotonically improving search objective. "
              "The unit-diagonal kernel gives nonpositive logdets, and larger selections ordinarily "
              "have more negative scores. Compare centralized and distributed draws at the **same size**, "
              "not scores at different cardinalities. Independent exact samples are not optimal and "
              "either method's realized score can be higher. Three data trials are preliminary, "
              "not a significance claim about the distributions.", "",
              "The comparison uses only sizes present in both saved sweeps. Distributed pools "
              "below 5,000 do not supply a size-5,000 comparison; missing data are not extrapolated "
              "or silently filled. `comparison_summary.csv` reports the number of paired trials "
              "at every size, which can fall below three near the endpoint.", "",
              "## Files and reproducibility", "",
              "`centralized_logdet.csv` tracks every score; `centralized_summary.csv` gives means "
              "and sample SDs. `comparison.csv` and `comparison_summary.csv` contain the paired "
              "gap (centralized minus distributed) at common sizes. PNG and SVG plots are saved.", "",
              "`results.json` contains original public words, dataset hashes, source-file hash, "
              "source path, seed components, resolved configuration, sampler diagnostics, "
              "sample-invariant checks, direct-Cholesky spot checks, runtimes and memory metadata. "
              "Each sample is stored losslessly as a slice of a recorded permutation; "
              "`selected_indices(record, trial[\"sweep\"][\"permutations\"])` returns original global IDs.", "",
              "The centralized reference explicitly materializes a dense global kernel (800 MB "
              "for 10,000 items, plus work arrays). It is not a scalable federated implementation. "
              "It uses no communication rounds or client uploads. Peak RSS covers the numerical "
              "process and is reported separately from the distributed source's resource use.", "",
              f"Source results: `{result['source']['results']}`", "",
              f"Source SHA-256: `{result['source']['sha256']}`", "",
              "Reference: [Kulesza–Taskar (2011), Section 3, Eq. (12)]"
              "(../../literature/k-DPPs%20Fixed-Size%20Determinantal%20Point%20Processes.pdf). "
              "The target is from the paper; the rejection implementation is an extension, "
              "not its spectral Algorithm 2.", ""]
    path = directory / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
