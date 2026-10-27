"""Thin command-line orchestration for client selection sweeps and plotting."""

import argparse
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys

from .cli import _metadata, allocate_run_directory
from .config import load_config
from .metrics import HAMMING_METRIC
from .plotting import (plot_client_selection_overlap, plot_pool_retention, plot_sweep_metrics,
                       plot_mcmc_trace, plot_nearest_hamming_sweep)
from .pool_retention import analyze_pool_retention, write_pool_retention
from .sweep import load_sweep_config, run_client_selection_sweep, write_sweep_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep client_selection_size and compare final G1 selections")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/client_selection_sweep.yml"))
    parser.add_argument("--backend", choices=("numpy", "dppy"), default="numpy")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        # Check the optional dependency before starting a long experiment.
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Install plotting dependencies with python -m pip install -e '.[plot]'") from exc
        config_path = args.config.resolve()
        base = load_config(config_path)
        sweep = load_sweep_config(args.sweep_config)
        sampling = base.search.algorithm == "dppy_mcmc"
        exact = base.search.algorithm == "dppy_exact"
        fixed = base.search.local_proposal_budget is not None or base.search.global_proposal_budget is not None
        if (sampling or exact) and args.backend != "dppy":
            raise ValueError(f"{base.search.algorithm} requires backend=dppy")
        if sampling:
            from dppy.mcmc_sampling import basis_exchange_sampler  # noqa: F401
        print(f"Backend={args.backend}; sizes={list(sweep.client_selection_sizes)}; trials={base.experiment.repetitions}", flush=True)
        print(f"Algorithm={base.search.algorithm}; pooled-context mode is NON-PRIVATE.", flush=True)
        if sampling:
            print(f"{base.search.chain_steps} transitions/stage; terminal state; no convergence guarantee.", flush=True)
        if fixed:
            print(f"Fixed proposed swaps: local={base.search.local_proposal_budget}, global={base.search.global_proposal_budget}; patience disabled for fixed stages.", flush=True)
        output = args.output_dir.resolve() if args.output_dir else config_path.parent / base.output.directory
        run_directory = allocate_run_directory(output, f"{base.search.algorithm}_client_selection_sweep") if sampling or exact else None
        if fixed:
            run_directory = allocate_run_directory(output, f"{args.backend}_sls_fixed_budget_client_selection_sweep")
        if run_directory and sampling:
            print(f"Streaming chain artifacts: {run_directory / 'chains'}", flush=True)

        def progress(size: int, row: dict) -> None:
            comparison = (f"nearest Hamming={row[HAMMING_METRIC]:.4f}, " if not sampling and not exact else
                          f"overlap={row['intersection_size']}/{base.selection.final_selection_size}, Jaccard={row['jaccard']:.4f}, ")
            print(
                f"size={size}, trial={row['trial'] + 1}/{base.experiment.repetitions}: "
                + comparison + f"gap={row['logdet_gap_fed_minus_central']:+.6f}",
                flush=True,
            )

        def point_done(point: dict) -> None:
            value = point["hamming_distance"] if not sampling and not exact else point["overlap"]
            label = "mean nearest Hamming" if not sampling and not exact else "mean overlap"
            print(f"size={point['client_selection_size']}: {label}={value['mean']:.4f} ± {value['std']:.4f}", flush=True)

        result = run_client_selection_sweep(
            base, sweep, args.backend, progress, point_done,
            run_directory / "chains" if run_directory else None,
        )
        result["metadata"] = _metadata(base.selection.forced_set_size, base.selection.final_objective == "pooled_context")
        result["metadata"]["dependency_versions"]["matplotlib"] = version("matplotlib")
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["metadata"]["process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        except ImportError:
            result["metadata"]["process_peak_rss_bytes"] = None
        result["metadata"]["peak_memory_scope"] = "entire sweep process, not separate approach peaks"
        directory = write_sweep_results(result, output, run_directory)
        png, svg = (plot_client_selection_overlap(result, directory) if sampling or exact else
                    plot_nearest_hamming_sweep(result, directory))
        if exact:
            metrics_png, _ = plot_sweep_metrics(result, directory)
            print(f"Metrics plot: {metrics_png}")
        if fixed:
            metrics_png, _ = plot_sweep_metrics(result, directory)
            print(f"Metrics plot: {metrics_png}")
        if sampling:
            metrics_png, _ = plot_sweep_metrics(result, directory)
            analysis = analyze_pool_retention(result)
            analysis["source_results_path"] = str((directory / "results.json").resolve())
            analysis["source_results_sha256"] = hashlib.sha256((directory / "results.json").read_bytes()).hexdigest()
            analysis_directory = write_pool_retention(analysis, directory)
            retention_png, _ = plot_pool_retention(analysis, analysis_directory)
            print(f"Metrics plot: {metrics_png}\nPool retention plot: {retention_png}")
            if base.output.save_chain:
                trace_png, _ = plot_mcmc_trace(result, directory)
                print(f"Chain trace: {trace_png}")
        print(f"Results: {directory}")
        print(f"Plot: {png}\nVector plot: {svg}")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
