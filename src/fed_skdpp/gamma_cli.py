"""Thin CLI for gamma sweeps ranked by nearest-neighbor Hamming distance."""

import argparse
from importlib.metadata import version
from pathlib import Path
import sys

from .cli import _metadata
from .config import load_config
from .metrics import HAMMING_METRIC
from .gamma_sweep import load_gamma_sweep, run_gamma_sweep, write_gamma_results
from .plotting import plot_nearest_hamming_sweep


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank gamma by lowest symmetric nearest-neighbor Hamming distance")
    parser.add_argument("--config", type=Path, default=Path("configs/gamma_sls.yml"))
    parser.add_argument("--sweep-config", type=Path, default=Path("configs/gamma_sweep.yml"))
    parser.add_argument("--backend", choices=("numpy", "dppy"), default="dppy")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        import matplotlib  # noqa: F401
        if args.backend == "dppy":
            from dppy.finite_dpps import FiniteDPP  # noqa: F401
        path = args.config.resolve()
        base = load_config(path)
        sweep = load_gamma_sweep(args.sweep_config)
        print(f"Backend={args.backend}; SLS; gammas={list(sweep.gammas)}; trials={base.experiment.repetitions}", flush=True)
        print(f"kmer_length={base.data.kmer_length}; local G1={base.selection.client_selection_size}; "
              f"final G1={base.selection.final_selection_size}; seed={base.experiment.seed}; NON-PRIVATE baseline.", flush=True)
        print("Same gamma at every stage; minimize symmetric nearest-neighbor Hamming distance; no pool-retention analysis.", flush=True)

        def progress(gamma: float, row: dict) -> None:
            print(f"gamma={gamma:.2f}, trial={row['trial'] + 1}/{base.experiment.repetitions}: "
                  f"nearest Hamming={row[HAMMING_METRIC]:.4f}, "
                  f"proposals central={row['centralized_proposals']}, server={row['server_proposals']}", flush=True)

        def point_done(point: dict) -> None:
            value = point["hamming_distance"]
            print(f"gamma={point['gamma']:.2f}: mean nearest Hamming={value['mean']:.4f} ± {value['std']:.4f}", flush=True)

        result = run_gamma_sweep(base, sweep, args.backend, progress, point_done)
        result["metadata"] = _metadata(base.selection.forced_set_size, base.selection.final_objective == "pooled_context")
        result["metadata"]["dependency_versions"]["matplotlib"] = version("matplotlib")
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["metadata"]["process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        except ImportError:
            result["metadata"]["process_peak_rss_bytes"] = None
        result["metadata"]["peak_memory_scope"] = "entire gamma sweep process, not separate approach peaks"
        output = args.output_dir.resolve() if args.output_dir else (path.parent / base.output.directory).resolve()
        directory = write_gamma_results(result, output)
        png, svg = plot_nearest_hamming_sweep(result, directory, "gamma")
        print(f"Best observed gamma(s): {result['best_gammas']}; mean nearest Hamming={result['best_mean_nearest_hamming']:.4f}")
        print(f"Results: {directory}\nPlot: {png}\nVector plot: {svg}")
        return 0
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
