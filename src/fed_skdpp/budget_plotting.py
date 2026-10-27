"""Headless plots of measured SLS traces and exhaustive checkpoint diagnostics."""

from pathlib import Path

import numpy as np


def _groups(result: dict) -> tuple[list[dict], list[dict]]:
    return ([trial["centralized"] for trial in result["trials"]],
            [stage for trial in result["trials"] for stage in trial["clients"]])


def plot_budget_curves(result: dict, directory: Path) -> tuple[Path, Path]:
    """Plot actual trajectories only; never extend a certified stop as new data."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    config, budget = result["config"], result["budget_config"]
    groups = _groups(result)
    figure = Figure(figsize=(13.8, 5.8), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3, gridspec_kw={"width_ratios": [1.15, 1.15, 0.75]})
    colors = ("#2563eb", "#7c3aed", "#0f766e")
    for group_index, stages in enumerate(groups):
        ax = axes[group_index]
        for index, stage in enumerate(stages):
            with np.load(stage["trace_file"], allow_pickle=False) as trace:
                scores = trace["logdet"]
            stride = max(1, len(scores) // 1500)
            x = np.unique(np.r_[np.arange(0, len(scores), stride), len(scores) - 1])
            color = colors[index % len(colors)] if group_index == 0 else "#64748b"
            ax.plot(x, scores[x], color=color, linewidth=1.2 if group_index == 0 else 0.7,
                    alpha=0.95 if group_index == 0 else 0.35,
                    label=f"Trial {index + 1}" if group_index == 0 else "Client trajectories" if index == 0 else None)
            certified = stage["certified"]
            ax.scatter(stage["iterations"], stage["logdet"], s=35 if group_index == 0 else 18,
                       marker="o" if certified else "x", color="#16a34a" if certified else "#ea580c", zorder=3)
            if group_index == 0 and stage["reference_patience_stop_iteration"] is not None:
                ax.scatter(stage["reference_patience_stop_iteration"], stage["reference_patience_logdet"],
                           color="#475569", marker="x", s=25, zorder=3,
                           label=f"Patience {budget['reference_patience']} stop" if index == 0 else None)
        ax.set_xlabel("Attempted swap proposals")
        ax.set_ylabel("Joint logdet (fixed G0 + G1)")
        ax.set_title("Centralized: G0=" + str(config["partition"]["num_clients"] * config["selection"]["forced_set_size"])
                     + f", G1={config['selection']['final_selection_size']}" if group_index == 0 else
                     f"Clients: G0={config['selection']['forced_set_size']}, G1={config['selection']['client_selection_size']}", fontsize=10)
        ax.set_xlim(0, max(1, max(stage["iterations"] for stage in stages)) * 1.03)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        ax.grid(color="#e2e8f0", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    for index, stages in enumerate(groups):
        offsets = np.linspace(-0.13, 0.13, len(stages))
        for offset, stage in zip(offsets, stages):
            ax.scatter(index + offset, stage["iterations"],
                       marker="o" if stage["certified"] else "x", s=25,
                       color="#16a34a" if stage["certified"] else "#ea580c", alpha=0.8)
    ax.scatter([], [], color="#16a34a", s=25, label="Certified checkpoint")
    ax.scatter([], [], color="#ea580c", marker="x", s=25, label="Uncertified cap")
    ax.axhline(budget["max_proposals"], color="#ea580c", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_ylim(max(1, min(stage["iterations"] for stages in groups for stage in stages)) * 0.6,
                budget["max_proposals"] * 1.5)
    ax.set_xticks([0, 1], [f"Centralized\nn={len(groups[0])}", f"Client-local\nn={len(groups[1])}"])
    ax.set_ylabel("Observed stopping proposal count")
    ax.set_title("Checkpoint certification / cap", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax.legend(fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.86))
    figure.suptitle("SLS search-budget calibration: centralized first, then smaller clients", fontsize=14, y=0.98)
    figure.text(0.5, 0.895,
                f"{config['data']['kmer_length']}-mers · gamma={config['kernel']['gamma']} · "
                f"{config['experiment']['repetitions']} paired datasets · cap={budget['max_proposals']:,} · "
                f"audit every {budget['checkpoint_interval']:,} proposals · tolerance={config['search']['improvement_tolerance']:.0e}",
                ha="center", fontsize=9, color="#475569")
    figure.text(0.5, 0.045,
                "Curves stop where measured; axes and joint cardinalities differ. Client runs are nested within the paired datasets.",
                ha="center", fontsize=8, color="#475569")
    figure.text(0.5, 0.018,
                "Non-private reference study. A certificate means no improving one-item swap beyond tolerance, not a global optimum.",
                ha="center", fontsize=8, color="#64748b")
    figure.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.20, wspace=0.34)
    paths = directory / "search_budget_logdet_curves.png", directory / "search_budget_logdet_curves.svg"
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return paths


def plot_budget_audits(result: dict, directory: Path) -> tuple[Path, Path]:
    """Show residual best swap gains, not an applied greedy/search trajectory."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    config, budget = result["config"], result["budget_config"]
    figure = Figure(figsize=(11.5, 5.7), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    for index, (ax, stages) in enumerate(zip(axes, _groups(result))):
        for stage_index, stage in enumerate(stages):
            rows = stage["checkpoints"]
            x = [row["iteration"] for row in rows]
            y = [row["max_logdet_gain"] if row["max_logdet_gain"] is not None else np.nan for row in rows]
            color = ("#2563eb", "#7c3aed", "#0f766e")[stage_index % 3] if index == 0 else "#64748b"
            ax.plot(x, y, marker="o" if index == 0 else None, markersize=2.5,
                    color=color, alpha=0.9 if index == 0 else 0.35, linewidth=1 if index == 0 else 0.7,
                    label=f"Trial {stage_index + 1}" if index == 0 else "Client audits" if stage_index == 0 else None)
        ax.axhline(config["search"]["improvement_tolerance"], linestyle="--", color="#ea580c", linewidth=0.8,
                   label="Certification tolerance")
        ax.set_yscale("symlog", linthresh=max(config["search"]["improvement_tolerance"], 1e-4))
        ax.set_xlabel("Attempted swap proposals")
        ax.set_ylabel("Best remaining one-swap logdet gain")
        ax.set_title("Centralized full neighborhoods" if index == 0 else "Client-local full neighborhoods", fontsize=11)
        ax.grid(color="#e2e8f0", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
        ax.legend(fontsize=8, frameon=False)
    figure.suptitle("A flat trace alone is not a convergence certificate", fontsize=14, y=0.98)
    figure.text(0.5, 0.89,
                f"Audits every {budget['checkpoint_interval']:,} proposals; all incoming/outgoing pairs checked without changing G1.",
                ha="center", fontsize=9, color="#475569")
    figure.text(0.5, 0.03,
                "Positive: improving swaps remain. Negative: all swaps are worse. First certified checkpoint is not exact convergence time or global optimality.",
                ha="center", fontsize=8, color="#64748b")
    figure.subplots_adjust(left=0.09, right=0.98, top=0.80, bottom=0.16, wspace=0.30)
    paths = directory / "search_budget_remaining_swap_gains.png", directory / "search_budget_remaining_swap_gains.svg"
    for path in paths:
        figure.savefig(path, dpi=180, bbox_inches="tight")
    return paths
