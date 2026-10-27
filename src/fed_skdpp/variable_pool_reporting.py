"""CSV, reproducibility report and figures for the variable-pool baseline."""

import csv
import json
from pathlib import Path

import numpy as np
import yaml

from .metrics import scalar_summary


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_variable_pool_results(result: dict, directory: Path) -> None:
    """Write into a fresh run directory; no previous result is overwritten."""
    if any(directory.iterdir()):
        raise FileExistsError("Refusing to overwrite variable-pool artifacts")
    with (directory / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with (directory / "config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result["config"], handle, sort_keys=False)
    local_rows, server_rows, client_rows = [], [], []
    by_size: dict[int, list[float]] = {}
    for trial in result["trials"]:
        number = trial["trial"]
        for client in trial["clients"]:
            local_rows.append({"trial": number, "client": client["client"],
                               "population_size": client["population_size"],
                               "local_selection_size": client["selection_size"],
                               "local_logdet": client["logdet"],
                               "expected_size": client["expected_size"],
                               "size_variance": client["size_variance"],
                               "local_sample_log_probability": client["sample_log_probability"]})
        for record in trial["sweep"]["records"]:
            size = record["selection_size"]
            by_size.setdefault(size, []).append(record["logdet"])
            server_rows.append({"trial": number, "pool_size": trial["pool_size"],
                                "selection_size": size, "logdet": record["logdet"],
                                "method": record["method"], "proposal_round": record["proposal_round"],
                                "permutation_index": record["permutation_index"]})
            for client, (count, score) in enumerate(zip(record["client_selection_sizes"], record["client_logdets"])):
                client_rows.append({"trial": number, "selection_size": size, "client": client,
                                    "client_contribution_size": count, "client_contribution_logdet": score})
    _csv(directory / "local_clients.csv", local_rows)
    _csv(directory / "server_logdet.csv", server_rows)
    _csv(directory / "client_logdet_by_size.csv", client_rows)
    _csv(directory / "server_summary.csv", [
        {"selection_size": size, **scalar_summary(scores)} for size, scores in sorted(by_size.items())
    ])


def plot_variable_pool(result: dict, directory: Path) -> list[Path]:
    """Global logdet, per-size client boxes, identifiable client heatmaps/local boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    trials = result["trials"]
    num_trials = len(trials)
    num_clients = result["config"]["partition"]["num_clients"]
    colors = plt.cm.tab10(np.linspace(0, 0.9, num_trials))
    paths = []

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = directory / f"{name}.{extension}"
            fig.savefig(path, dpi=170, bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    for trial, color in zip(trials, colors):
        rows = trial["sweep"]["records"]
        sizes = [row["selection_size"] for row in rows]
        scores = [row["logdet"] for row in rows]
        label = f"Trial {trial['trial'] + 1}: m={trial['pool_size']}"
        ax.plot(sizes, scores, lw=1.1, color=color, label=label)
        ax.scatter(sizes[-1:], scores[-1:], color=color, marker="x", zorder=3)
    ax.set(xlabel="Server subset size k", ylabel="logdet of full server subset",
           title="Ordinary local DPPs → exact server k-DPP\n20-mers / gamma=0.6" if (
               result["config"]["data"]["kmer_length"] == 20 and result["config"]["kernel"]["gamma"] == 0.6)
           else "Ordinary local DPPs → exact server k-DPP")
    ax.grid(alpha=.2)
    ax.legend()
    save(fig, "server_logdet")

    fig, axes = plt.subplots(num_trials, 1, figsize=(15, max(4, 3.2 * num_trials)), squeeze=False)
    interval = result["config"]["sweep"]["selection_size_interval"]
    for trial, color, ax in zip(trials, colors, axes[:, 0]):
        rows = trial["sweep"]["records"]
        sizes = np.asarray([row["selection_size"] for row in rows])
        # Every sampled size has a real box, not a downsampled or binned score.
        ax.boxplot([row["client_logdets"] for row in rows], positions=sizes,
                   widths=max(.3, interval * .6), manage_ticks=False, showfliers=True,
                   patch_artist=True, boxprops={"facecolor": color, "alpha": .45, "linewidth": .4},
                   medianprops={"color": "black", "linewidth": .5},
                   whiskerprops={"color": color, "linewidth": .4},
                   capprops={"color": color, "linewidth": .4},
                   flierprops={"marker": ".", "markersize": 1, "alpha": .3, "markeredgecolor": color})
        ax.set(title=f"Trial {trial['trial'] + 1}: one box per k, {num_clients} client scores per box; m={trial['pool_size']}",
               xlabel="Server subset size k", ylabel="Client contribution logdet")
        ax.grid(axis="y", alpha=.2)
    save(fig, "client_logdet_boxplots_by_size")

    fig, axes = plt.subplots(num_trials, 2, figsize=(15, max(4, 3.1 * num_trials)), squeeze=False)
    for trial, pair in zip(trials, axes):
        rows = trial["sweep"]["records"]
        sizes = np.asarray([row["selection_size"] for row in rows], dtype=float)
        # Nonuniform final endpoint gets its actual cell boundaries.
        edges = (np.r_[sizes[0] - .5, sizes[0] + .5] if len(sizes) == 1 else
                 np.r_[sizes[0] - (sizes[1] - sizes[0]) / 2,
                       (sizes[:-1] + sizes[1:]) / 2,
                       sizes[-1] + (sizes[-1] - sizes[-2]) / 2])
        for ax, key, label in zip(pair, ("client_logdets", "client_selection_sizes"),
                                  ("Client contribution logdet", "Number of contributed items")):
            values = np.asarray([row[key] for row in rows]).T
            mesh = ax.pcolormesh(edges, np.arange(num_clients + 1) + .5, values,
                                 shading="flat", cmap="viridis")
            ax.set(xlabel="Server subset size k", ylabel="Client (displayed 1-based)",
                   title=f"Trial {trial['trial'] + 1}: {label}")
            ax.set_yticks(np.arange(num_clients) + 1)
            fig.colorbar(mesh, ax=ax, pad=.02)
    save(fig, "client_contribution_heatmaps")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, label in zip(axes, ("selection_size", "logdet"),
                              ("Local ordinary-DPP sample size", "Local ordinary-DPP logdet")):
        scores = [[trial["clients"][client][key] for trial in trials] for client in range(num_clients)]
        ax.boxplot(scores, positions=np.arange(num_clients) + 1, manage_ticks=False,
                   showfliers=False, widths=.5)
        for trial, color in zip(trials, colors):
            ax.scatter(np.arange(num_clients) + 1, [client[key] for client in trial["clients"]],
                       color=color, s=18, alpha=.8, label=f"Trial {trial['trial'] + 1}")
        ax.set(xlabel="Client (displayed 1-based)", ylabel=label, title=label)
        ax.set_xticks(np.arange(num_clients) + 1)
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize="small")
    save(fig, "local_client_boxplots")
    return paths


def write_variable_pool_report(result: dict, directory: Path) -> Path:
    """Explicit laws, coupling, contribution semantics, runtime and limitations."""
    config = result["config"]
    lines = ["# Ordinary-local-DPP / exact-server-k-DPP baseline", "",
             f"{config['partition']['num_clients']} clients, "
             f"{config['data']['global_collection_size'] // config['partition']['num_clients']} "
             f"unique {config['data']['kmer_length']}-mers each; "
             f"{config['experiment']['repetitions']} independent data trials, seed {config['experiment']['seed']}.", "",
             f"Kernel: $L_{{xy}}=\\exp(-{config['kernel']['gamma']} d_H(x,y))$ "
             f"with raw mismatch count; explicit jitter {config['kernel']['jitter']}. "
             "Unit quality scores. All computations are float64.", "",
             "## Setting and target", "",
             "Each client makes one ordinary, variable-size DPP draw: "
             "$P(A_i)=\\det(L_{A_i})/\\det(I+L_{S_i})$. "
             "The disjoint client partitions produce a merged pool $P=\\bigcup_i A_i$, "
             "with random size $m$. Local sample sizes are neither fixed nor tuned.", "",
             f"On the same realized pool, sweep subset sizes from {config['sweep']['minimum_selection_size']} "
             f"through $m$ in steps of {config['sweep']['selection_size_interval']}. "
             "Append $m$ as an extra endpoint if it misses the interval grid. "
             "If the pool is smaller than the minimum, report its endpoint only, without redrawing it.", "",
             "For each size $k$, draw exactly from "
             "$P(Y\\mid P,|Y|=k)=\\det(L_Y)/e_k(\\lambda(L_P))$. "
             "One draw per size per trial. Draws across sizes share proposal permutations and "
             "uniform thresholds; they have the correct individual marginal laws but are correlated "
             "and not necessarily nested. Different trials are independent. There is no SLS or MCMC.", "",
             "This is a **context-free, non-private suggestion-aggregation baseline**: "
             "all $G0_i$ are empty, synthetic features and partitions are public to the evaluator, "
             "and clients send selected IDs with their identities in plaintext. No privacy, "
             "semi-supervised utility, MAP optimality, or equivalence to a full-universe k-DPP is claimed.", "",
             "## Scores and plots", "",
             "The global curve scores the whole server draw, $\\log\\det(L_Y)$. "
             "At every size, each client's score is "
             "$\\log\\det(L_{Y\\cap S_i})$, using only its contribution to that server draw. "
             "The boxplot at each size contains these client scores within one trial. "
             "Empty intersections have logdet zero, as $\\det(L_\\varnothing)=1$. "
             "These contribution scores are not logdet of the full shared output as seen by a client; "
             "that value would be identical for all clients in this context-free experiment.", "",
             "Scores across sizes or unequal client contribution counts are not size-controlled "
             "diversity comparisons. Counts are saved and plotted. With unit diagonal and no jitter, "
             "logdet is nonpositive, and adding points to a fixed set cannot increase it. "
             "Independent/coupled non-nested exact draws are not a monotone optimization trajectory.", "",
             "![Server logdet](server_logdet.png)", "",
             "![Client boxes at every size](client_logdet_boxplots_by_size.png)", "",
             "![Individual client scores and counts](client_contribution_heatmaps.png)", "",
             "![Initial local draws, boxes across trials](local_client_boxplots.png)", "",
             "## Results", "",
             "| Trial | Local sample sizes (client order) | Pool m | Sizes sampled | Full-pool logdet | Seconds |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for trial in result["trials"]:
        sizes = ", ".join(str(client["selection_size"]) for client in trial["clients"])
        last = trial["sweep"]["records"][-1]
        lines.append(f"| {trial['trial'] + 1} | {sizes} | {trial['pool_size']} | "
                     f"{len(trial['sweep']['records'])} | {last['logdet']:.9g} | {trial['total_seconds']:.1f} |")
    lines += ["", "## Exact rejection implementation", "",
              "The local sampler is native `DPPy.sample_exact(mode=\"GS\")`, using the "
              "ordinary spectral Bernoulli-eigenvector construction in Kulesza–Taskar (2011), "
              "Algorithm 1. Server sampling is an exact rejection **extension**, not DPPy's "
              "fixed-size spectral API or the Kulesza eigenvector-selection implementation.", "",
              "For constant diagonal $d$, define $Q=L/d$, which leaves every fixed-size law "
              "unchanged. Hadamard's inequality gives $\\det(Q_Y)\\leq1$. "
              "For sizes at most half the pool, propose uniform subsets and accept with "
              "probability $\\det(Q_Y)$. For larger sizes, uniformly propose the omitted set "
              "$Z=P\\setminus Y$ and use Jacobi's identity "
              "$\\det(Q_Y)=\\det(Q)\\det((Q^{-1})_Z)$. "
              "The product of the largest $|Z|$ diagonal entries of $Q^{-1}$ is a valid "
              "size-dependent Hadamard upper bound. Rejection then weights the complement by "
              "exactly the required principal determinant. At $k=m$, return the pool deterministically.", "",
              "A uniform permutation supplies uniform subsets of every prefix size; a shared "
              "uniform threshold tests each size's own rejection probability. Each size returns "
              "its first acceptance. Coupling proposals across sizes preserves each marginal "
              "rejection law. One prefix Cholesky scores many sizes per proposal, avoiding "
              "repeated decompositions and overflowing elementary-symmetric-polynomial tables. "
              "No rejection-budget cutoff, eigenvalue clipping, low-rank approximation, or "
              "implicit jitter is used. Float64 roundoff is assessed separately from ideal-arithmetic exactness.", "",
              "## Validation and reproducibility", "",
              "`results.json` stores the full public data, partitions, local samples, seeds, "
              "kernel/numerical configuration and compact server samples. A server record's "
              "`permutation_index`, `slice_start` and `slice_stop` select MATRIX-local pool indices "
              "from its trial's `sweep.permutations`; map through `pool_ids` for global IDs. "
              "This stores every output without repeating millions of IDs in JSON.", "",
              "CSV: `local_clients.csv`, `server_logdet.csv`, `client_logdet_by_size.csv`, "
              "and `server_summary.csv`. The summary includes trial counts because random pool "
              "sizes give different endpoints. Client IDs in CSV/JSON are zero-based; plot labels are one-based.", ""]
    for trial in result["trials"]:
        diagnostics = trial["sweep"]["diagnostics"]
        lines.append(f"- Trial {trial['trial'] + 1}: direct/complement proposal rounds "
                     f"{diagnostics['rounds']}; direct-Cholesky spot-check max error "
                     f"{diagnostics.get('direct_cholesky_max_error', 0):.3g}.")
    lines += ["", "The run is a sequential local simulation, not a network or encrypted protocol. "
              "Logical int64-ID communication counts include one local-upload batch and one "
              "batch broadcasting all requested server outputs to all clients. Framing, network "
              "latency, failures and non-IID populations are not modeled. Peak RSS metadata covers "
              "the whole process. Dense kernels are built only per client and for the realized pool, "
              "not for the full 10,000-item universe.", "",
              "References: [Kulesza–Taskar (2011), Eqs. (4)/(12), Algorithms 1/2]"
              "(../../literature/k-DPPs%20Fixed-Size%20Determinantal%20Point%20Processes.pdf), "
              "[Kulesza–Taskar survey, Sections 2.2 and 2.4.4]"
              "(../../literature/Determinantal%20Point%20Processes%20for%20Machine%20Learning.pdf).", ""]
    path = directory / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
