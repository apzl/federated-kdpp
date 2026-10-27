"""Run the centralized size sweep using datasets from a saved distributed run."""

import argparse
from pathlib import Path
import resource
import sys

from .centralized_pool import load_centralized_pool_config, run_centralized_pool_experiment
from .centralized_pool_reporting import (
    plot_centralized_pool, write_centralized_pool_report, write_centralized_pool_results,
)
from .cli import _metadata, allocate_run_directory


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/centralized_pool_baseline.yml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        path = args.config.resolve()
        config = load_centralized_pool_config(path)
        root = args.output_dir.resolve() if args.output_dir else (path.parent / config.output.directory).resolve()
        print(f"Saved global-data centralized exact k-DPP: {config.source.results}", flush=True)

        def progress(row):
            if row["stage"] == "centralized_start":
                print(f"Trial {row['trial'] + 1}: all {row['population_size']} global items, "
                      f"{row['num_sizes']} sizes; constructing dense centralized kernel", flush=True)
            elif row["stage"] == "centralized_complete":
                print(f"Trial {row['trial'] + 1} complete: final logdet={row['last_logdet']:.9f}; "
                      f"{row['seconds']:.1f}s", flush=True)
            elif row["round"] <= 3 or row["round"] % 10 == 0 or not row["remaining"]:
                print(f"  Trial {row['trial'] + 1} exact {row['branch']} rejection round "
                      f"{row['round']}: accepted {row['accepted']}, remaining sizes={row['remaining']}", flush=True)

        result = run_centralized_pool_experiment(config, progress)
        result["metadata"] = _metadata()
        result["metadata"].update({
            "execution": "non-private centralized dense-kernel oracle, no client/local-sampling stage",
            "server_observes": "saved public global collection; no private contexts",
            "communication": "none; zero rounds and bytes",
        })
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["metadata"]["process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        result["metadata"]["peak_memory_scope"] = "numerical process through completed centralized sweep"
        directory = allocate_run_directory(root, "centralized_global_pool_kdpp")
        write_centralized_pool_results(result, directory)
        plot_centralized_pool(result, directory)
        write_centralized_pool_report(result, directory)
        print(f"Completed in {result['total_seconds']:.1f}s; results: {directory}", flush=True)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
