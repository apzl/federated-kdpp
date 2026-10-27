"""Optional, headless plots from saved numerical experiment results."""

from pathlib import Path

import numpy as np

from .metrics import HAMMING_METRIC, NORMALIZED_HAMMING_METRIC


def plot_nearest_hamming_sweep(
    result: dict, directory: Path, parameter: str = "client_selection_size",
) -> tuple[Path, Path]:
    """Primary SLS comparison: symmetric cross-set nearest Hamming distance.

    Lower is closer; both directions have equal weight, shared items contribute
    zero, G0 is excluded. Error bars are sample SD, not confidence intervals.
    Length sweeps additionally show distance divided by sequence length.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if parameter not in {"client_selection_size", "gamma", "kmer_length"}:
        raise ValueError("Unsupported Hamming plot parameter")
    config, points = result["base_config"], result["points"]
    length_sweep = parameter == "kmer_length"
    x = np.asarray([point[parameter] for point in points], dtype=float)
    figure = Figure(figsize=(12.4 if length_sweep else 10.8, 6.3), facecolor="white")
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(1, 2 if length_sweep else 1))
    spacing = float(np.min(np.diff(x))) if len(x) > 1 else 1.0
    for ax, normalized in zip(axes, (False, True)):
        key = "hamming_distance_normalized" if normalized else "hamming_distance"
        metric = NORMALIZED_HAMMING_METRIC if normalized else HAMMING_METRIC
        means = np.array([point[key]["mean"] for point in points])
        errors = np.array([point[key]["std"] for point in points])
        for index, point in enumerate(points):
            values = [trial["metrics"][metric] for trial in point["experiment"]["trials"]]
            offsets = np.linspace(-spacing * .08, spacing * .08, len(values)) if len(values) > 1 else [0.]
            ax.scatter(x[index] + offsets, values, color="#94a3b8", s=23, alpha=.7,
                       label="Individual paired trials" if index == 0 else None, zorder=2)
        ax.errorbar(x, means, yerr=errors, color="#2563eb", marker="o", markersize=6,
                    linewidth=2, capsize=4, label="Mean distance ± sample SD", zorder=3)
        if parameter == "gamma":
            best = np.isin(x, result["best_gammas"])
            ax.scatter(x[best], means[best], marker="*", s=220, color="#7c3aed",
                       edgecolors="white", label="Lowest observed mean distance", zorder=4)
        ticks = result["sweep_config"]["kmer_lengths"] if length_sweep else x
        ax.set_xticks(ticks)
        skipped = {point["kmer_length"] for point in result.get("skipped_points", [])}
        ax.set_xticklabels([f"{value}\nN/A" if value in skipped else
                            f"{value:.2f}" if parameter == "gamma" else f"{value:g}"
                            for value in ticks], fontsize=8)
        for value in skipped:
            ax.axvspan(value - .4, value + .4, color="#f59e0b", alpha=.08)
        ax.set_xlabel({"client_selection_size": "Client suggestion size (k-mers per client)",
                       "gamma": "Hamming RBF gamma (same at every stage)",
                       "kmer_length": "k-mer length"}[parameter], labelpad=10)
        ax.set_ylabel("Normalized mean nearest-neighbor Hamming distance" if normalized else
                      "Symmetric mean nearest-neighbor Hamming distance", labelpad=10)
        upper = max(float(np.max(means + errors)), max(point[key]["max"] for point in points))
        ax.set_ylim(0, 1.05 if normalized else max(.1, upper * 1.23))
        ax.grid(axis="y", color="#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8.2, frameon=False, loc="upper left")
    search = config["search"]
    budget = (f"local {search['local_proposal_budget']:,}, global {search['global_proposal_budget']:,} proposals"
              if search.get("local_proposal_budget") is not None and search.get("global_proposal_budget") is not None
              else f"patience {search['patience']}")
    figure.suptitle("Centralized vs two-stage SLS: nearest-neighbor sequence distance", fontsize=14, y=.975)
    length_label = "varying length" if length_sweep else f"{config['data']['kmer_length']}-mers"
    gamma_label = "gamma sweep" if parameter == "gamma" else f"gamma={config['kernel']['gamma']}"
    local_label = "local G1 sweep" if parameter == "client_selection_size" else f"local G1={config['selection']['client_selection_size']}"
    figure.text(.5, .91,
                f"N={config['data']['global_collection_size']:,} {length_label} · {config['partition']['num_clients']} clients · "
                f"{local_label} · final G1={config['selection']['final_selection_size']} · {gamma_label}\n"
                f"{budget} · {config['experiment']['repetitions']} paired trials per point · lower is closer, not more diverse",
                ha="center", fontsize=8.7, color="#475569")
    figure.text(.5, .045, "Mean of both directed nearest-neighbor means; many-to-one matches allowed; exact matches = 0; fixed G0 excluded.",
                ha="center", fontsize=8, color="#475569")
    figure.text(.5, .018, "Non-private baseline. Search still optimizes logdet. " +
                ("Normalized distance divides by length; compare raw distances across lengths with care." if length_sweep else
                 "Agreement with a heuristic reference is not optimality or privacy."),
                ha="center", fontsize=8, color="#64748b")
    figure.subplots_adjust(left=.09, right=.98, bottom=.18, top=.81, wspace=.30)
    name = {"client_selection_size": "client_selection_size_hamming_distance",
            "gamma": "gamma_vs_nearest_hamming", "kmer_length": "kmer_length_nearest_hamming"}[parameter]
    paths = (directory / f"{name}.png", directory / f"{name}.svg")
    for path in paths:
        figure.savefig(path, dpi=180 if path.suffix == ".png" else None, bbox_inches="tight")
    return paths


def plot_client_selection_overlap(result: dict, directory: Path) -> tuple[Path, Path]:
    """Plot exact G1 overlap count vs client suggestion size as PNG and SVG.

    Points and error bars are the mean and sample SD across configured trials.
    Gray dots show individual trial overlaps; only feasible, actually evaluated
    sizes are plotted. The fixed G0 is excluded from all overlap measurements.
    """
    if all("hamming_distance" in point for point in result["points"]):
        return plot_nearest_hamming_sweep(result, directory)
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError("Plotting requires Matplotlib; install with python -m pip install -e '.[plot]'") from exc
    config = result["base_config"]
    points = result["points"]
    x = np.array([point["client_selection_size"] for point in points])
    means = np.array([point["overlap"]["mean"] for point in points])
    errors = np.array([point["overlap"]["std"] for point in points])
    figure = Figure(figsize=(10.2, 6.2), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    for index, point in enumerate(points):
        values = [trial["metrics"]["intersection_size"] for trial in point["experiment"]["trials"]]
        offsets = np.linspace(-0.65, 0.65, len(values)) if len(values) > 1 else [0.0]
        axes.scatter(
            x[index] + np.asarray(offsets), values, s=26, color="#94a3b8", alpha=0.7,
            label="Individual paired trials" if index == 0 else None, zorder=2,
        )
    axes.errorbar(
        x, means, yerr=errors, color="#2563eb", linewidth=2.2,
        marker="o", markersize=7, markerfacecolor="white", markeredgewidth=2,
        capsize=5, elinewidth=1.4, label="Mean overlap ± sample SD", zorder=3,
    )
    axes.set_xlabel("client_selection_size (k-mers suggested per client)", fontsize=11, labelpad=10)
    axes.set_ylabel("Overlap size (shared G1 k-mers)", fontsize=11, labelpad=10)
    axes.set_xticks(x)
    selection_size = config["selection"]["final_selection_size"]
    axes.set_ylim(0, max(1, selection_size * 1.06))
    axes.margins(x=0.04)
    axes.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    axes.set_axisbelow(True)
    for name in ("top", "right"):
        axes.spines[name].set_visible(False)
    axes.legend(loc="upper left", frameon=False, fontsize=10)
    figure.suptitle("Overlap between centralized and federated selections", fontsize=16, y=0.97)
    pooled_size = config["partition"]["num_clients"] * config["selection"]["forced_set_size"]
    pooled = config["selection"]["final_objective"] == "pooled_context"
    context_label = f"pooled G0: {pooled_size}" if pooled else f"local G0/client: {config['selection']['forced_set_size']}"
    subtitle = (
        f"{config['data']['global_collection_size']:,} public {config['data']['kmer_length']}-mers · "
        f"{config['partition']['num_clients']} clients · {context_label} · final G1: {selection_size}\n"
        f"Hamming RBF gamma={config['kernel']['gamma']} · "
        + (f"DPPy MCMC: {config['search']['chain_steps']:,} steps · " if config['search']['algorithm'] == "dppy_mcmc"
           else f"DPPy exact: {config['search']['exact_mode']} · " if config['search']['algorithm'] == "dppy_exact"
           else f"SLS budgets: local {config['search']['local_proposal_budget']:,}, global {config['search']['global_proposal_budget']:,} · "
           if config['search'].get('local_proposal_budget') is not None and config['search'].get('global_proposal_budget') is not None
           else f"{config['search']['patience']} stalled proposals · ")
        +
        f"{config['experiment']['repetitions']} paired trials per size"
    )
    axes.set_title(subtitle, fontsize=10, color="#475569", pad=17)
    figure.text(
        0.5, 0.025,
        ("Non-private pooled-context baseline. " if pooled else "Two-stage suggestion baseline; no privacy guarantee. ")
        + ("Exact sampling: fixed central reference per dataset; independent server stream."
           if config['search']['algorithm'] == "dppy_exact" else
           "Centralized reference re-run per size with pool-dependent paired initialization."),
        ha="center", fontsize=8.2, color="#64748b",
    )
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.15, top=0.79)
    png = directory / "client_selection_size_overlap.png"
    svg = directory / "client_selection_size_overlap.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    return png, svg


def plot_pool_retention(result: dict, directory: Path) -> tuple[Path, Path]:
    """Plot |pool intersect central G1| and |server G1 intersect central G1|.

    Input is the count-only analyze_pool_retention output. Mean/sample-SD
    curves distinguish local exclusion from retained-but-not-selected items;
    neither difference estimates a logdet penalty or an optimality gap.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise RuntimeError("Plotting requires Matplotlib; install with python -m pip install -e '.[plot]'") from exc
    config = result["base_config"]
    points = result["points"]
    x = np.asarray([point["client_selection_size"] for point in points])
    selection_size = config["selection"]["final_selection_size"]
    figure = Figure(figsize=(10.4, 6.4), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_subplot(111)
    for metric, label, color, marker in (
        ("pool_centralized_overlap", "Centralized G1 items retained in local pool", "#0f766e", "o"),
        ("final_overlap", "Centralized G1 items in server final G1", "#2563eb", "s"),
    ):
        means = [point["summary"][metric]["mean"] for point in points]
        errors = [point["summary"][metric]["std"] for point in points]
        axes.errorbar(
            x, means, yerr=errors, color=color, marker=marker, markersize=6,
            linewidth=2, capsize=4, elinewidth=1.3, markerfacecolor="white",
            markeredgewidth=1.7, label=label, zorder=3,
        )
    retained = np.asarray([point["summary"]["pool_centralized_overlap"]["mean"] for point in points])
    final = np.asarray([point["summary"]["final_overlap"]["mean"] for point in points])
    axes.fill_between(x, retained, selection_size, color="#f59e0b", alpha=0.10, zorder=1)
    axes.fill_between(x, final, retained, color="#0f766e", alpha=0.07, zorder=1)
    axes.axhline(selection_size, color="#64748b", linewidth=1, linestyle="--", label=f"All {selection_size} centralized G1 items")
    axes.set_xlabel("client_selection_size (k-mers suggested per client)", fontsize=11, labelpad=10)
    axes.set_ylabel("Centralized G1 items retained / shared", fontsize=11, labelpad=10)
    axes.set_xticks(x)
    axes.set_ylim(0, max(1, selection_size * 1.07))
    axes.margins(x=0.04)
    axes.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    axes.set_axisbelow(True)
    for name in ("top", "right"):
        axes.spines[name].set_visible(False)
    sampling = config["search"]["algorithm"] == "dppy_mcmc"
    axes.legend(loc="upper left" if sampling else "lower right", frameon=True,
                facecolor="white", framealpha=0.95, fontsize=9)
    figure.suptitle("Where centralized candidates are lost in the two-stage search", fontsize=15, y=0.975)
    axes.set_title(
        f"{config['data']['global_collection_size']:,} public {config['data']['kmer_length']}-mers · "
        f"{config['partition']['num_clients']} clients · final G1: {selection_size} · "
        f"seed {config['experiment']['seed']}\n"
        f"Hamming RBF gamma={config['kernel']['gamma']} · "
        + (f"DPPy MCMC: {config['search']['chain_steps']:,} steps · " if config['search']['algorithm'] == "dppy_mcmc" else "")
        +
        f"{config['experiment']['repetitions']} paired trials · error bars: sample SD",
        fontsize=10, color="#475569", pad=16,
    )
    pooled = config["selection"]["final_objective"] == "pooled_context"
    figure.text(
        0.5, 0.045,
        "Amber: absent from local pool. Green: available, but not selected by server. Fixed G0 excluded.",
        ha="center", fontsize=8.6, color="#475569",
    )
    figure.text(
        0.5, 0.017,
        ("Non-private pooled-context baseline. " if pooled else "No privacy guarantee. ")
        + ("Centralized sample re-run per size; overlap is not sampling correctness or utility loss."
           if config["search"]["algorithm"] == "dppy_mcmc" else
           "Centralized heuristic re-run per size; membership disagreement is not utility loss."),
        ha="center", fontsize=8.2, color="#64748b",
    )
    figure.subplots_adjust(left=0.10, right=0.97, bottom=0.17, top=0.80)
    png = directory / "client_selection_size_pool_retention.png"
    svg = directory / "client_selection_size_pool_retention.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    return png, svg


def plot_sweep_metrics(
    result: dict, directory: Path, parameter: str = "client_selection_size",
) -> tuple[Path, Path]:
    """Plot logdet/gap plus nearest Hamming for SLS, Jaccard for samplers."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    config = result["base_config"]
    points = result["points"]
    if parameter not in {"client_selection_size", "kmer_length"}:
        raise ValueError("Unsupported metrics plot parameter")
    length_sweep = parameter == "kmer_length"
    x = [point[parameter] for point in points]
    figure = Figure(figsize=(11.6, 4.8), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    hamming = (config["search"]["algorithm"] == "stochastic_local_search"
               and all(HAMMING_METRIC in point["experiment"]["summary"] for point in points))
    groups = (
        (("centralized_logdet", "Centralized", "#0f766e"), ("federated_logdet", "Two-stage", "#2563eb")),
        (("logdet_gap_fed_minus_central", "Two-stage minus centralized", "#7c3aed"),),
        ((HAMMING_METRIC if hamming else "jaccard", "Nearest Hamming (lower is closer)" if hamming else "Final G1 Jaccard", "#2563eb"),),
    )
    for ax, metrics, label in zip(axes, groups, ("Joint logdet (pooled G0 + G1)", "Paired logdet gap", "Symmetric nearest-neighbor Hamming" if hamming else "Jaccard similarity")):
        for metric, legend, color in metrics:
            stats = [point["experiment"]["summary"][metric] for point in points]
            ax.errorbar(x, [v["mean"] for v in stats], yerr=[v["std"] for v in stats],
                        marker="o", markersize=4, capsize=3, color=color, label=legend)
        ax.set_ylabel(label)
        ax.set_xlabel("k-mer length" if length_sweep else "Client suggestion size")
        ax.set_xticks(x)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", color="#e2e8f0")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    axes[1].axhline(0, color="#64748b", linestyle="--", linewidth=1)
    axes[2].set_ylim(bottom=0)
    if not hamming:
        axes[2].set_ylim(top=1.06)
    sampling = config["search"]["algorithm"] == "dppy_mcmc"
    exact = config["search"]["algorithm"] == "dppy_exact"
    figure.suptitle("k-mer length sensitivity: centralized vs two-stage SLS" if length_sweep else
                   "DPPy conditional exact k-DPP: centralized vs two-stage" if exact else
                   "DPPy conditional k-DPP MCMC: terminal-state comparison" if sampling else
                   "Centralized vs two-stage SLS comparison", fontsize=14)
    budget = (f"{config['search']['chain_steps']:,} transitions/stage" if sampling else
              f"one exact draw/stage ({config['search']['exact_mode']})" if exact else
              f"local={config['search']['local_proposal_budget']:,}, global={config['search']['global_proposal_budget']:,} proposals"
              if config['search'].get('local_proposal_budget') is not None and config['search'].get('global_proposal_budget') is not None else
              f"patience={config['search']['patience']}")
    figure.text(0.5, 0.89,
                f"Final G1={config['selection']['final_selection_size']} · gamma={config['kernel']['gamma']} · "
                f"{budget} · {config['experiment']['repetitions']} paired trials · mean ± sample SD",
                ha="center", fontsize=9, color="#475569")
    caveat = ("Length changes the kernel; compare methods within each length, not raw logdet across lengths." if length_sweep else
              "Independent central/server draws; overlap is not a sampling-convergence criterion." if exact else
              "Finite chain budget does not establish convergence or a MAP optimum." if sampling else
              "Finite stochastic search does not establish a MAP optimum.")
    figure.text(0.5, 0.02, "Non-private pooled-context baseline. " + caveat,
                ha="center", fontsize=8, color="#64748b")
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.81, wspace=0.35)
    name = ("kmer_length_metrics" if length_sweep else "client_selection_size_exact_metrics" if exact
            else "client_selection_size_sls_fixed_budget_metrics" if config['search'].get('global_proposal_budget') is not None
            else "client_selection_size_mcmc_metrics")
    png, svg = directory / f"{name}.png", directory / f"{name}.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    return png, svg


def plot_mcmc_trace(result: dict, directory: Path) -> tuple[Path, Path]:
    """Show the two final-stage traces for trial 0 at the base suggestion size."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    config = result["base_config"]
    point = next((p for p in result["points"] if p["client_selection_size"] == config["selection"]["client_selection_size"]), result["points"][0])
    trial = point["experiment"]["trials"][0]
    figure = Figure(figsize=(10.5, 5.4), facecolor="white")
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    for stage, color, label in (("centralized", "#0f766e", "Centralized"), ("server", "#2563eb", "Two-stage server")):
        with np.load(trial[stage]["chain_file"], allow_pickle=False) as saved:
            scores = saved["joint_logdet"]
        ax.plot(np.arange(len(scores)), scores, linewidth=0.65, alpha=0.8, color=color, label=label)
        ax.scatter(len(scores) - 1, scores[-1], color=color, s=35, zorder=4)
    ax.axvline(config["search"]["burn_in"], linestyle="--", color="#64748b", linewidth=1, label="Diagnostic burn-in boundary")
    ax.set_xlabel("Chain transition (0 = initial state)")
    ax.set_ylabel("Joint logdet (pooled G0 + G1)")
    ax.set_title(f"MCMC traces: trial 1, client suggestion size {point['client_selection_size']}, "
                 f"final G1={config['selection']['final_selection_size']}")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", color="#e2e8f0")
    ax.spines[["top", "right"]].set_visible(False)
    figure.text(0.5, 0.02, "Terminal states are returned, not maximum-logdet states. This trace alone does not establish mixing.",
                ha="center", fontsize=8, color="#64748b")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.88)
    png, svg = directory / "mcmc_terminal_chain_trace.png", directory / "mcmc_terminal_chain_trace.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    return png, svg


def plot_gamma_overlap(result: dict, directory: Path) -> tuple[Path, Path]:
    """Plot observed final-selection agreement vs gamma, with all rank-one ties.

    Error bars are sample SD over paired trials, not confidence intervals.
    Raw logdet is not plotted/ranked across different kernels. Private contexts
    are fixed, and excluded from overlap counts. No pool-retention metrics.
    """
    if all("hamming_distance" in point for point in result["points"]):
        return plot_nearest_hamming_sweep(result, directory, "gamma")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    config = result["base_config"]
    points = result["points"]
    x = np.asarray([point["gamma"] for point in points])
    means = np.asarray([point["overlap"]["mean"] for point in points])
    errors = np.asarray([point["overlap"]["std"] for point in points])
    figure = Figure(figsize=(11.2, 6.3), facecolor="white")
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    spacing = float(np.min(np.diff(x))) if len(x) > 1 else 0.05
    for index, point in enumerate(points):
        values = [trial["metrics"]["intersection_size"] for trial in point["experiment"]["trials"]]
        offsets = np.linspace(-spacing * 0.10, spacing * 0.10, len(values))
        ax.scatter(x[index] + offsets, values, color="#94a3b8", s=20, alpha=0.65,
                   label="Individual paired trials" if index == 0 else None, zorder=2)
    ax.errorbar(x, means, yerr=errors, color="#2563eb", marker="o", markersize=5,
                linewidth=1.8, capsize=3, label="Mean final G1 overlap ± sample SD", zorder=3)
    best = np.isin(x, result["best_gammas"])
    ax.scatter(x[best], means[best], marker="*", s=230, color="#7c3aed",
               edgecolors="white", linewidth=0.8, label="Highest observed mean overlap", zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{value:.2f}" for value in x], fontsize=8)
    ax.set_xlabel("Hamming RBF gamma (same value at all stages)", fontsize=11, labelpad=10)
    ax.set_ylabel(f"Centralized ∩ federated final G1 (out of {config['selection']['final_selection_size']})",
                  fontsize=11, labelpad=10)
    upper = max(float(np.max(means + errors)), max(point["overlap"]["max"] for point in points))
    ax.set_ylim(0, max(10, upper * 1.18))
    ax.margins(x=0.025)
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=True, framealpha=0.95, fontsize=9)
    figure.suptitle("Gamma sensitivity: centralized vs federated SLS agreement", fontsize=15, y=0.97)
    ax.set_title(
        f"{config['data']['global_collection_size']:,} public {config['data']['kmer_length']}-mers · "
        f"{config['partition']['num_clients']} clients · local G1={config['selection']['client_selection_size']} · "
        f"final G1={config['selection']['final_selection_size']}\n"
        f"{config['search']['patience']} stalled proposals · seed {config['experiment']['seed']} · "
        f"{config['experiment']['repetitions']} paired trials/gamma · fixed G0 excluded from overlap",
        fontsize=10, color="#475569", pad=15,
    )
    winners = ", ".join(f"{value:.2f}" for value in result["best_gammas"])
    figure.text(0.5, 0.04, f"Best observed gamma: {winners} · mean overlap: {result['best_mean_overlap']:.2f}",
                 ha="center", color="#6d28d9", fontsize=9)
    figure.text(0.5, 0.015,
                "Non-private pooled-context baseline. Tuned for agreement, not maximum diversity; no held-out validation.",
                ha="center", color="#64748b", fontsize=8)
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.17, top=0.80)
    png, svg = directory / "gamma_vs_final_overlap.png", directory / "gamma_vs_final_overlap.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    return png, svg


def plot_kmer_overlap(result: dict, directory: Path) -> tuple[Path, Path]:
    """Plot final-set overlap across lengths; infeasible points have no value."""
    if all("hamming_distance" in point for point in result["points"]):
        return plot_nearest_hamming_sweep(result, directory, "kmer_length")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    config, points = result["base_config"], result["points"]
    x = np.asarray([point["kmer_length"] for point in points])
    means = np.asarray([point["overlap"]["mean"] for point in points])
    errors = np.asarray([point["overlap"]["std"] for point in points])
    figure = Figure(figsize=(10.5, 6.2), facecolor="white")
    FigureCanvasAgg(figure)
    ax = figure.add_subplot(111)
    for index, point in enumerate(points):
        values = [trial["metrics"]["intersection_size"] for trial in point["experiment"]["trials"]]
        ax.scatter(x[index] + np.linspace(-0.12, 0.12, len(values)), values,
                   color="#94a3b8", s=25, alpha=0.65, zorder=2,
                   label="Individual paired trials" if index == 0 else None)
    ax.errorbar(x, means, yerr=errors, color="#2563eb", marker="o", linewidth=1.8,
                capsize=4, label="Mean final G1 overlap ± sample SD", zorder=3)
    skipped = {point["kmer_length"] for point in result["skipped_points"]}
    ticks = result["sweep_config"]["kmer_lengths"]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{value}\nN/A" if value in skipped else str(value) for value in ticks])
    for value in skipped:
        ax.axvspan(value - 0.4, value + 0.4, color="#f59e0b", alpha=0.08)
    ax.set_xlabel("k-mer length", fontsize=11, labelpad=10)
    ax.set_ylabel(f"Centralized ∩ federated final G1 (out of {config['selection']['final_selection_size']})",
                  fontsize=11, labelpad=10)
    upper = max(float(np.max(means + errors)), max(point["overlap"]["max"] for point in points))
    ax.set_ylim(0, max(10, upper * 1.2))
    ax.set_xlim(min(ticks) - 0.5, max(ticks) + 0.5)
    ax.grid(axis="y", color="#e2e8f0")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
    figure.suptitle("k-mer length sensitivity: centralized vs federated SLS agreement", fontsize=15, y=0.975)
    ax.set_title(
        f"{config['data']['global_collection_size']:,} unique DNA sequences · {config['partition']['num_clients']} clients · "
        f"local G1={config['selection']['client_selection_size']} · final G1={config['selection']['final_selection_size']}\n"
        f"gamma={config['kernel']['gamma']} at all stages · patience={config['search']['patience']} · "
        f"{config['experiment']['repetitions']} trials/length · error bars: sample SD",
        fontsize=10, color="#475569", pad=16,
    )
    if skipped:
        lengths = ", ".join(str(value) for value in sorted(skipped))
        figure.text(0.5, 0.04, f"Infeasible lengths: {lengths}. Unique sequence capacity is too small; settings were not changed.",
                    ha="center", fontsize=8.5, color="#92400e")
    figure.text(0.5, 0.015,
                "Non-private pooled-context baseline. Fixed G0 excluded from overlap; agreement is not utility or optimality.",
                ha="center", fontsize=8, color="#64748b")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.19, top=0.80)
    png, svg = directory / "kmer_length_vs_final_overlap.png", directory / "kmer_length_vs_final_overlap.svg"
    figure.savefig(png, dpi=180, bbox_inches="tight")
    figure.savefig(svg, bbox_inches="tight")
    return png, svg
