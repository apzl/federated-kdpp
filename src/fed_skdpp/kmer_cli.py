"""Thin CLI for fixed-gamma k-mer length sweeps and comparison plots."""

import argparse
from importlib.metadata import version
from pathlib import Path
import sys

from .cli import _metadata
from .config import load_config
from .metrics import HAMMING_METRIC
from .kmer_sweep import load_kmer_sweep, run_kmer_sweep, write_kmer_results
from .plotting import plot_nearest_hamming_sweep, plot_sweep_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare centralized and two-stage SLS across k-mer lengths")
    parser.add_argument("--config", type=Path, default=Path("configs/kmer_sls.yml"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/kmer_sweep.yml"))
    parser.add_argument("--backend", choices=("numpy", "dppy"), default="dppy")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        import matplotlib  # noqa: F401
        if args.backend == "dppy":
            from dppy.finite_dpps import FiniteDPP  # noqa: F401
        path = args.config.resolve()
        base = load_config(path)
        sweep = load_kmer_sweep(args.sweep_config)
        print(f"Backend={args.backend}; SLS; kmer_lengths={list(sweep.kmer_lengths)}; "
              f"trials={base.experiment.repetitions}; patience={base.search.patience}", flush=True)
        print(f"N={base.data.global_collection_size}; clients={base.partition.num_clients}; "
              f"local G1={base.selection.client_selection_size}; final G1={base.selection.final_selection_size}; "
              f"gamma={base.kernel.gamma} at all stages; seed={base.experiment.seed}; no privacy guarantee.", flush=True)

        def progress(length: int, row: dict) -> None:
            print(f"length={length}, trial={row['trial'] + 1}/{base.experiment.repetitions}: "
                  f"nearest Hamming={row[HAMMING_METRIC]:.4f}, logdet gap={row['logdet_gap_fed_minus_central']:.4f}, "
                  f"proposals central={row['centralized_proposals']}, server={row['server_proposals']}", flush=True)

        def point_done(point: dict) -> None:
            value = point["hamming_distance"]
            print(f"length={point['kmer_length']}: mean nearest Hamming={value['mean']:.4f} ± {value['std']:.4f}", flush=True)

        def skipped(point: dict) -> None:
            print(f"SKIPPED length={point['kmer_length']}: {point['reason']} "
                  f"(unique capacity={point['unique_kmer_capacity']}, requested N={point['required_collection_size']})", flush=True)

        result = run_kmer_sweep(base, sweep, args.backend, progress, point_done, skipped)
        result["metadata"] = _metadata(base.selection.forced_set_size, base.selection.final_objective == "pooled_context")
        result["metadata"]["dependency_versions"]["matplotlib"] = version("matplotlib")
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["metadata"]["process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        except ImportError:
            result["metadata"]["process_peak_rss_bytes"] = None
        result["metadata"]["peak_memory_scope"] = "entire length sweep process, not separate approach peaks"
        output = args.output_dir.resolve() if args.output_dir else (path.parent / base.output.directory).resolve()
        directory = write_kmer_results(result, output)
        overlap_png, overlap_svg = plot_nearest_hamming_sweep(result, directory, "kmer_length")
        metrics_png, metrics_svg = plot_sweep_metrics(result, directory, parameter="kmer_length")
        print(f"Results: {directory}\nHamming distance: {overlap_png}\nMetrics: {metrics_png}\n"
              f"Vector plots: {overlap_svg}, {metrics_svg}")
        return 0
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
