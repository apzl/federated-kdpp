"""CLI for ordinary local DPPs and an exact k-DPP server cardinality sweep."""

import argparse
from pathlib import Path
import resource
import sys

from .cli import _metadata, allocate_run_directory
from .variable_pool import load_variable_pool_config, run_variable_pool_experiment
from .variable_pool_reporting import plot_variable_pool, write_variable_pool_report, write_variable_pool_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/variable_pool_baseline.yml"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = load_variable_pool_config(config_path)
        root = args.output_dir.resolve() if args.output_dir else config_path.parent / config.output.directory
        print(f"Ordinary local DPP → exact server k-DPP; gamma={config.kernel.gamma}; "
              f"{config.data.global_collection_size} items / {config.partition.num_clients} clients; "
              f"{config.experiment.repetitions} independent trials. No G0, no privacy guarantee.", flush=True)

        def progress(row):
            stage = row["stage"]
            if stage == "local_dpp":
                print(f"Trial {row['trial'] + 1}, client {row['client'] + 1}: "
                      f"ordinary DPP size={row['selection_size']}, logdet={row['logdet']:.8f}", flush=True)
            elif stage == "pool":
                print(f"Trial {row['trial'] + 1}: merged m={row['pool_size']}; "
                      f"{row['num_sizes']} server sizes", flush=True)
            elif row["round"] <= 3 or not row["remaining"] or row["round"] % 10 == 0:
                print(f"  Exact {row['branch']} rejection round {row['round']}: "
                      f"accepted {row['accepted']}, remaining sizes={row['remaining']}", flush=True)

        result = run_variable_pool_experiment(config, progress)
        result["metadata"] = _metadata()
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["metadata"]["process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        result["metadata"]["peak_memory_scope"] = "entire numerical experiment process"
        directory = allocate_run_directory(root, "variable_pool_dpp_baseline")
        write_variable_pool_results(result, directory)
        plot_variable_pool(result, directory)
        write_variable_pool_report(result, directory)
        print(f"Completed in {result['total_seconds']:.1f}s; results: {directory}", flush=True)
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
