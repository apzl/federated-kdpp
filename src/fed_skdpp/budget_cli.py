"""Thin command for centralized-then-client SLS budget calibration."""

import argparse
from importlib.metadata import version
from pathlib import Path
import sys

from .budget import load_budget_config, run_budget_experiment, write_budget_results
from .budget_plotting import plot_budget_curves, plot_budget_audits
from .cli import _metadata, allocate_run_directory
from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Calibrate SLS proposals with full one-swap convergence checks")
    parser.add_argument("--config", type=Path, default=Path("configs/kmer_sls.yml"))
    parser.add_argument("--budget-config", type=Path, default=Path("configs/search_budget.yml"))
    parser.add_argument("--backend", choices=("numpy", "dppy"), default="dppy")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        import matplotlib  # noqa: F401
        if args.backend == "dppy":
            from dppy.finite_dpps import FiniteDPP  # noqa: F401
        path = args.config.resolve()
        config = load_config(path)
        budget = load_budget_config(args.budget_config)
        output = args.output_dir.resolve() if args.output_dir else (path.parent / config.output.directory).resolve()
        directory = allocate_run_directory(output, f"{args.backend}_search_budget")
        print(f"NON-PRIVATE SLS budget study: {config.experiment.repetitions} trials; "
              f"{config.data.kmer_length}-mers; gamma={config.kernel.gamma}; "
              f"cap={budget.max_proposals:,}; full swap audit every {budget.checkpoint_interval:,} proposals.", flush=True)
        print(f"First central: N={config.data.global_collection_size}, "
              f"G0={config.partition.num_clients * config.selection.forced_set_size}, "
              f"G1={config.selection.final_selection_size}; then all {config.partition.num_clients} clients/trial: "
              f"N={config.client_collection_size}, G0={config.selection.forced_set_size}, G1={config.selection.client_selection_size}.", flush=True)
        print("Patience is disabled; no audit-selected moves applied; central initialization independent of local suggestions.", flush=True)
        print(f"Output directory: {directory}", flush=True)

        def checkpoint(kind: str, trial: int, client: int | None, row: dict) -> None:
            cadence = 10000 if kind == "centralized" else 5000
            if row["certified"] or row["iteration"] % cadence == 0 or row["iteration"] == budget.max_proposals:
                suffix = f", client={client + 1}" if client is not None else ""
                print(f"{kind}, trial={trial + 1}{suffix}: proposals={row['iteration']:,}, "
                      f"logdet={row['logdet']:.6f}, best remaining swap={row['max_logdet_gain']}, "
                      f"improving pairs={row['improving_swaps']}, certified={row['certified']}", flush=True)

        def stage_done(kind: str, trial: int, client: int | None, stage: dict) -> None:
            suffix = f", client={client + 1}" if client is not None else ""
            gain = stage["gain_after_reference_patience"]
            print(f"DONE {kind}, trial={trial + 1}{suffix}: {stage['stop_reason']}; "
                  f"{stage['iterations']:,} proposals; additional logdet after patience {budget.reference_patience}: {gain}", flush=True)

        result = run_budget_experiment(config, budget, args.backend, directory, stage_done, checkpoint)
        metadata = _metadata(config.selection.forced_set_size, True)
        metadata.update({
            "server_observes": "not applicable; no server protocol, only trusted non-private optimizer measurements",
            "communication": "no messages or network; convergence study only",
            "peak_memory_scope": "entire sequential budget-study process, not separate stage peaks",
        })
        metadata["dependency_versions"]["matplotlib"] = version("matplotlib")
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            metadata["process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        except ImportError:
            metadata["process_peak_rss_bytes"] = None
        result["metadata"] = metadata
        write_budget_results(result, directory)
        curves = plot_budget_curves(result, directory)
        audits = plot_budget_audits(result, directory)
        for name, summary in result["summary"].items():
            print(f"{name}: {summary}")
        print(f"Transfer assessment: {result['transfer']}")
        print(f"Results: {directory}\nCurves: {curves[0]}\nAudit plot: {audits[0]}")
        return 0
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
