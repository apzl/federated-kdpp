"""Standalone paired-trial plots for exact sampling and fixed-budget SLS."""

from pathlib import Path

import numpy as np

from .metrics import HAMMING_METRIC, NORMALIZED_HAMMING_METRIC


def plot_exact_comparison(result: dict, directory: Path) -> tuple[Path, Path]:
    """Plot actual paired trials: G1 overlap, Jaccard, utility, paired gap.

    Each point represents one realized sample in one paired data trial.
    Shared fixed G0 is excluded from membership similarity measurements.
    There is no optimization trajectory or best-of-repeated-samples selection.
    """
    if result["config"]["search"]["algorithm"] != "dppy_exact":
        raise ValueError("Exact comparison plot requires dppy_exact results")
    return _plot_trial_comparison(result, directory, exact=True)


def plot_fixed_budget_comparison(result: dict, directory: Path) -> tuple[Path, Path]:
    """Plot SLS nearest Hamming distance, joint utilities, and paired gaps."""
    search = result["config"]["search"]
    if (search["algorithm"] != "stochastic_local_search"
            or search.get("local_proposal_budget") is None
            or search.get("global_proposal_budget") is None):
        raise ValueError("Fixed-budget comparison plot requires local and global SLS budgets")
    return _plot_trial_comparison(result, directory, exact=False)


def plot_sls_comparison(result: dict, directory: Path) -> tuple[Path, Path]:
    """Standalone SLS comparison, with either patience or fixed stopping."""
    if result["config"]["search"]["algorithm"] != "stochastic_local_search":
        raise ValueError("SLS comparison requires stochastic_local_search")
    return _plot_trial_comparison(result, directory, exact=False)


def _plot_trial_comparison(result: dict, directory: Path, exact: bool) -> tuple[Path, Path]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    rows = [trial["metrics"] for trial in result["trials"]]
    x = np.arange(1, len(rows) + 1)
    config = result["config"]
    figure = Figure(figsize=(12, 7.2), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    hamming = not exact and all(HAMMING_METRIC in row for row in rows)
    panels = (
        (axes[0, 0], HAMMING_METRIC, "Symmetric nearest Hamming\n(lower means closer)", config["data"]["kmer_length"], ".3f"),
        (axes[0, 1], NORMALIZED_HAMMING_METRIC, "Normalized nearest Hamming\n(distance / length)", 1, ".4f"),
    ) if hamming else (
        (axes[0, 0], "intersection_size", "Shared G1 items", config["selection"]["final_selection_size"], ".0f"),
        (axes[0, 1], "jaccard", "G1 Jaccard similarity", 1, ".3f"),
    )
    for ax, key, label, limit, fmt in panels:
        values = np.array([row[key] for row in rows])
        bars = ax.bar(x, values, color="#2563eb", width=0.5)
        ax.bar_label(bars, labels=[format(v, fmt) for v in values], padding=4, fontsize=10)
        ax.axhline(values.mean(), color="#334155", linestyle="--", linewidth=1,
                   label=f"Mean {values.mean():.2f}" if key == "intersection_size" else f"Mean {values.mean():.3f}")
        ax.set_ylim(0, limit * 1.04)
        ax.set_ylabel(label)
        ax.legend(frameon=False, loc="upper right", fontsize=9)
    for key, label, color, marker in (
        ("centralized_logdet", "Centralized", "#0f766e", "o"),
        ("federated_logdet", "Two-stage federated", "#d97706", "s"),
    ):
        axes[1, 0].plot(x, [row[key] for row in rows], marker=marker,
                        linestyle="none", color=color, label=label, markersize=7)
    for index, row in enumerate(rows):
        axes[1, 0].plot([x[index], x[index]], [row["centralized_logdet"], row["federated_logdet"]],
                        color="#94a3b8", linewidth=1, zorder=0)
    pooled = config["selection"].get("final_objective", "regular") == "pooled_context"
    axes[1, 0].set_ylabel("Joint logdet (pooled G0 + G1)" if pooled else "Regular G1 logdet")
    axes[1, 0].legend(frameon=False, fontsize=9)
    gaps = [row["logdet_gap_fed_minus_central"] for row in rows]
    bars = axes[1, 1].bar(x, gaps, width=0.5, color="#7c3aed")
    axes[1, 1].bar_label(bars, fmt="%+.3f", padding=4, fontsize=10)
    axes[1, 1].axhline(0, color="#334155", linewidth=1)
    axes[1, 1].set_ylabel("Logdet gap: federated minus central")
    axes[1, 1].margins(y=0.22)
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xlabel("Paired trial")
        ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fixed = config["search"].get("local_proposal_budget") is not None and config["search"].get("global_proposal_budget") is not None
    figure.suptitle("DPPy Kulesza–Taskar exact k-DPP sampling" if exact else
                   "SLS with fixed local and global proposal budgets" if fixed else
                   "SLS: centralized vs two-stage selection", fontsize=16, y=0.98)
    budget_note = ("One draw per stage; independent centralized and server sampling streams" if exact else
                   f"Local: {config['search']['local_proposal_budget']:,} proposals · "
                   f"central/server: {config['search']['global_proposal_budget']:,} proposals · paired initialization"
                   if fixed else f"Patience {config['search']['patience']} · unchanged logdet optimization")
    figure.text(0.5, 0.925,
                f"{config['data']['global_collection_size']:,} {config['data']['kmer_length']}-mers · "
                f"gamma={config['kernel']['gamma']} · {config['partition']['num_clients']} clients · "
                f"local G1={config['selection']['client_selection_size']} · final G1={config['selection']['final_selection_size']}\n"
                + budget_note,
                ha="center", va="top", fontsize=10, color="#475569")
    figure.text(0.5, 0.02,
                ("Non-private pooled-context baseline. " if pooled else "Two-stage suggestion baseline. ")
                + ("Logdet is evaluated on complete samples; no swap iterations." if exact else
                   "Cross-set nearest-neighbor distance; exact matches contribute zero; fixed G0 excluded."),
                ha="center", fontsize=9, color="#64748b")
    figure.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.10, wspace=0.28, hspace=0.40)
    name = "exact_comparison" if exact else "sls_fixed_budget_comparison" if fixed else "sls_hamming_comparison"
    paths = (directory / f"{name}.png", directory / f"{name}.svg")
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite plot: {path}")
    figure.savefig(paths[0], dpi=180, bbox_inches="tight")
    figure.savefig(paths[1], bbox_inches="tight")
    return paths
