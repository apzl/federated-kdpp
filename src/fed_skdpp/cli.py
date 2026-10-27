"""Shared CLI, output serialization, and reproducibility metadata."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import uuid

import yaml

from .config import load_config
from .experiment import run_experiment
from .metrics import HAMMING_METRIC, NORMALIZED_HAMMING_METRIC


def _metadata(forced_set_size: int = 0, pooled_context: bool = False) -> dict:
    versions = {}
    for package in ("fed-skdpp", "numpy", "PyYAML", "dppy", "scipy"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    digest = hashlib.sha256()
    for source in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return {
        "python": sys.version, "platform": platform.platform(),
        "dependency_versions": versions, "source_sha256": digest.hexdigest(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dtype": "float64", "quality_scores": "all one",
        "server_observes": (
            "public global collection, client-selected item IDs, and every client's disclosed G0 membership"
            if pooled_context else "public global collection and client-selected item IDs"
        ),
        "private_forced_sets": forced_set_size > 0 and not pooled_context,
        "client_contexts_disclosed": pooled_context,
        "privacy_guarantee": False,
        "log_visibility": "trusted local evaluator only; exact client scores/history are not server-visible protocol messages",
        "private_membership_logged": False,
        "disclosed_context_membership_logged": pooled_context,
        "execution": "sequential local simulation, not a deployed federated system",
        "communication": "estimated int64-ID payload only; no network or framing overhead measured",
        "thread_environment": {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        )},
    }


def allocate_run_directory(directory: Path, prefix: str) -> Path:
    """Reserve a fresh directory before streaming large generated artifacts."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = directory / f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=False)
    return target


def write_results(result: dict, directory: Path, run_directory: Path | None = None) -> Path:
    """Create a unique run directory; never overwrite a prior experiment."""
    if run_directory is None:
        run_directory = allocate_run_directory(directory, result["backend"])
    elif any((run_directory / name).exists() for name in (
        "results.json", "config.yml", "metrics.csv", "search_history.csv", "report.md",
    )):
        raise FileExistsError("Refusing to overwrite existing experiment artifacts")
    with (run_directory / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with (run_directory / "config.yml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(result["config"], handle, sort_keys=False)
    rows = [trial["metrics"] for trial in result["trials"]]
    with (run_directory / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if result["config"]["output"]["save_history"]:
        with (run_directory / "search_history.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = ["trial", "stage", "iteration", "logdet", "accepted", "improved", "stalled_iterations"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for trial in result["trials"]:
                stages = [("centralized", trial["centralized"]), ("server", trial["server"])]
                stages.extend((f"client_{index}", stage) for index, stage in enumerate(trial["clients"]))
                for name, stage in stages:
                    for step in stage["history"]:
                        writer.writerow({"trial": trial["trial"], "stage": name, **step})
    if result["config"]["search"]["algorithm"] == "stochastic_local_search":
        from .hamming_reporting import write_hamming_report
        write_hamming_report(result, run_directory / "report.md")
    return run_directory


def main(backend: str, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{backend} centralized/distributed k-mer comparison")
    parser.add_argument("--config", type=Path, default=Path("config.yml"))
    parser.add_argument("--output-dir", type=Path, help="Override the configured results directory")
    args = parser.parse_args(argv)
    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        pooled = config.selection.final_objective == "pooled_context"
        budget = (f"chain steps={config.search.chain_steps}" if config.search.algorithm == "dppy_mcmc"
                  else f"one exact draw per stage; mode={config.search.exact_mode}" if config.search.algorithm == "dppy_exact"
                  else f"fixed proposals: local={config.search.local_proposal_budget}, global={config.search.global_proposal_budget}"
                  if config.search.local_proposal_budget is not None or config.search.global_proposal_budget is not None
                  else f"patience={config.search.patience}")
        print(f"Backend: {backend}; algorithm={config.search.algorithm}; gamma={config.kernel.gamma}; {budget}", flush=True)
        print(
            f"Local G0 size={config.selection.forced_set_size}; "
            f"local G1 size={config.selection.client_selection_size}; "
            f"final G1 size={config.selection.final_selection_size}. "
            f"Final objective={config.selection.final_objective}; no privacy guarantee.", flush=True,
        )
        if pooled:
            print(
                f"NON-PRIVATE baseline: all {config.partition.num_clients * config.selection.forced_set_size} "
                "context items disclosed to server; fixed in both final joint-logdet selections.", flush=True,
            )

        def progress(row: dict) -> None:
            comparison = (f"nearest Hamming={row[HAMMING_METRIC]:.4f} (lower is closer)"
                          if config.search.algorithm == "stochastic_local_search" else f"Jaccard={row['jaccard']:.4f}")
            print(
                f"Trial {row['trial'] + 1}/{config.experiment.repetitions}: "
                f"central={row['centralized_logdet']:.6f}, fed={row['federated_logdet']:.6f}, "
                f"gap={row['logdet_gap_fed_minus_central']:+.6f}, {comparison}; "
                f"proposals central={row['centralized_proposals']}, fed={row['federated_total_proposals']}",
                flush=True,
            )

        output = args.output_dir.resolve() if args.output_dir else config_path.parent / config.output.directory
        run_directory = None
        if config.search.algorithm == "dppy_exact" or (config.search.algorithm == "dppy_mcmc" and config.output.save_chain):
            if backend != "dppy":
                raise ValueError(f"{config.search.algorithm} requires backend=dppy")
            run_directory = allocate_run_directory(output, config.search.algorithm)
        elif config.search.local_proposal_budget is not None or config.search.global_proposal_budget is not None:
            run_directory = allocate_run_directory(output, f"{backend}_sls_fixed_budget")
        result = run_experiment(
            config, backend, progress,
            run_directory / "chains" if run_directory else None,
        )
        result["metadata"] = _metadata(config.selection.forced_set_size, pooled)
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["metadata"]["process_peak_rss_bytes"] = int(peak if sys.platform == "darwin" else peak * 1024)
        except ImportError:
            result["metadata"]["process_peak_rss_bytes"] = None
        result["metadata"]["peak_memory_scope"] = "entire script process, not separate approach peaks"
        run_directory = write_results(result, output, run_directory)
        if config.search.algorithm == "dppy_exact":
            from .exact_plotting import plot_exact_comparison
            png, _ = plot_exact_comparison(result, run_directory)
            print(f"Plot: {png}")
        elif config.search.algorithm == "stochastic_local_search":
            from .exact_plotting import plot_sls_comparison
            try:
                png, _ = plot_sls_comparison(result, run_directory)
            except ImportError:
                print("Optional Matplotlib unavailable; numerical results and report saved.")
            else:
                print(f"Plot: {png}")
        print("Mean ± sample standard deviation:")
        comparisons = ((HAMMING_METRIC, NORMALIZED_HAMMING_METRIC) if config.search.algorithm == "stochastic_local_search"
                       else ("intersection_size", "jaccard"))
        for metric in ("centralized_logdet", "federated_logdet", "logdet_gap_fed_minus_central") + comparisons:
            value = result["summary"][metric]
            print(f"  {metric}: {value['mean']:.6f} ± {value['std']:.6f}")
        print(f"Results: {run_directory}")
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


def numpy_main() -> int:
    return main("numpy")


def dppy_main() -> int:
    return main("dppy")
