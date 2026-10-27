"""Analyze existing sweep selections without re-running stochastic search."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .cli import _metadata
from .plotting import plot_pool_retention
from .pool_retention import analyze_pool_retention, write_pool_retention


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot local-pool retention of centralized G1 items")
    parser.add_argument("--results", required=True, type=Path, help="Saved client-selection sweep results.json")
    parser.add_argument("--output-dir", type=Path, help="Default: source sweep's directory")
    args = parser.parse_args(argv)
    try:
        import matplotlib  # noqa: F401

        source = args.results.resolve()
        content = source.read_bytes()
        result = analyze_pool_retention(json.loads(content))
        result["source_results"] = {"path": str(source), "sha256": hashlib.sha256(content).hexdigest()}
        config = result["base_config"]
        result["analysis_metadata"] = _metadata(
            config["selection"]["forced_set_size"], config["selection"]["final_objective"] == "pooled_context",
        )
        result["analysis_metadata"]["dependency_versions"]["matplotlib"] = version("matplotlib")
        result["analysis_metadata"]["execution"] = "postprocessing saved selections; no optimization re-run"
        for point in result["points"]:
            summary = point["summary"]
            print(
                f"size={point['client_selection_size']}: "
                f"retained={summary['pool_centralized_overlap']['mean']:.1f}, "
                f"excluded locally={summary['excluded_locally']['mean']:.1f}, "
                f"final overlap={summary['final_overlap']['mean']:.1f}", flush=True,
            )
        output = args.output_dir.resolve() if args.output_dir else source.parent
        target = write_pool_retention(result, output)
        png, svg = plot_pool_retention(result, target)
        print(f"Analysis: {target}\nPlot: {png}\nVector plot: {svg}")
        return 0
    except ImportError as exc:
        parser.exit(2, f"error: plotting requires Matplotlib (install '.[plot]'): {exc}\n")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
