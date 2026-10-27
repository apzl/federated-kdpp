"""Centralized exact k-DPP sweep on saved, unfiltered global datasets.

The target is Kulesza--Taskar (2011), Section 3 Eq. (12), on all of S,
not on the realized local-suggestion pool. Reuses the tested exact rejection
extension in variable_dpp; does not modify/re-run the distributed experiment.
"""

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from .config import DataConfig, KernelConfig, NumericsConfig, OutputConfig
from .kernels import hamming_rbf, stable_logdet
from .variable_dpp import exact_rejection_sweep, selected_indices


@dataclass(frozen=True)
class SavedPoolSource:
    results: str

    def __post_init__(self):
        if not isinstance(self.results, str) or not self.results.strip():
            raise ValueError("source.results must name a saved distributed results.json")


@dataclass(frozen=True)
class CentralizedSizeSweep:
    minimum_selection_size: int = 10
    maximum_selection_size: int = 5000
    selection_size_interval: int = 5

    def __post_init__(self):
        for name in ("minimum_selection_size", "maximum_selection_size", "selection_size_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_selection_size > self.maximum_selection_size:
            raise ValueError("minimum_selection_size exceeds maximum_selection_size")

    def sizes(self) -> list[int]:
        sizes = list(range(self.minimum_selection_size, self.maximum_selection_size + 1,
                           self.selection_size_interval))
        if sizes[-1] != self.maximum_selection_size:
            sizes.append(self.maximum_selection_size)
        return sizes


@dataclass(frozen=True)
class CentralizedSampling:
    seed: int = 42

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass(frozen=True)
class CentralizedPoolConfig:
    source: SavedPoolSource
    sweep: CentralizedSizeSweep = CentralizedSizeSweep()
    experiment: CentralizedSampling = CentralizedSampling()
    output: OutputConfig = OutputConfig(directory="../results", save_history=False)

    def __post_init__(self):
        if self.output.save_chain or self.output.save_history:
            raise ValueError("Exact sampling has no optimizer history or MCMC chain")

    def as_dict(self):
        return asdict(self)


def load_centralized_pool_config(path: str | Path) -> CentralizedPoolConfig:
    """Strict YAML; source paths resolve relative to the config, never cwd."""
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    kinds = {"source": SavedPoolSource, "sweep": CentralizedSizeSweep,
             "experiment": CentralizedSampling, "output": OutputConfig}
    if not isinstance(raw, dict) or set(raw) - set(kinds):
        raise ValueError("Unknown centralized-pool config section or non-mapping config")
    if not isinstance(raw.get("source"), dict) or "results" not in raw["source"]:
        raise ValueError("source.results is required; new datasets must not be generated")
    parsed = {}
    for name, kind in kinds.items():
        values = raw.get(name, {})
        if not isinstance(values, dict) or set(values) - {field.name for field in fields(kind)}:
            raise ValueError(f"Unknown or invalid {name} settings")
        if name == "source":
            source = SavedPoolSource(**values)
            parsed[name] = SavedPoolSource(str((path.parent / source.results).resolve()))
        elif name == "output":
            parsed[name] = kind(**{**asdict(OutputConfig(directory="../results", save_history=False)), **values})
        else:
            parsed[name] = kind(**values)
    return CentralizedPoolConfig(**parsed)


def read_distributed_source(path: str | Path) -> tuple[dict, str]:
    """Validate public datasets and hashes from the prior context-free baseline.

    Saved row order defines global IDs. No new words or partitions are drawn.
    Data/kernel/numerical choices are inherited to prevent accidental changes
    to a paired comparison. Reject malformed, duplicate or tampered fixtures.
    """
    payload = Path(path).read_bytes()
    source = json.loads(payload)
    if source.get("schema") != "ordinary_local_dpp_exact_server_size_sweep_v1":
        raise ValueError("Expected the saved variable-pool distributed baseline schema")
    data = DataConfig(**source["config"]["data"])
    KernelConfig(**source["config"]["kernel"])
    NumericsConfig(**source["config"]["numerics"])
    if not source.get("trials"):
        raise ValueError("Distributed source contains no trials")
    seen = set()
    for trial in source["trials"]:
        index = trial["trial"]
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("Source trial IDs must be unique nonnegative integers")
        seen.add(index)
        words = trial["global_collection"]
        if (len(words) != data.global_collection_size
                or any(not isinstance(word, str) or len(word) != data.kmer_length for word in words)
                or len(set(words)) != len(words)):
            raise ValueError("Source dataset has wrong cardinality, duplicates or sequence lengths")
        try:
            encoded = np.asarray([[data.alphabet.index(symbol) for symbol in word] for word in words], dtype=np.int64)
        except ValueError as exc:
            raise ValueError("Source sequence contains a symbol outside its configured alphabet") from exc
        if hashlib.sha256(encoded.tobytes()).hexdigest() != trial["collection_sha256"]:
            raise ValueError("Source collection hash mismatch: refusing to regenerate or reorder data")
        pool = trial["pool_ids"]
        if (len(pool) != trial["pool_size"] or len(pool) != len(set(pool))
                or any(type(item) is not int or not 0 <= item < len(words) for item in pool)):
            raise ValueError("Source suggestion pool IDs are invalid")
        sizes = set()
        for row in trial["sweep"]["records"]:
            size = row["selection_size"]
            if type(size) is not int or not 0 <= size <= len(pool) or size in sizes or not np.isfinite(row["logdet"]):
                raise ValueError("Source distributed sweep has invalid sizes or scores")
            sizes.add(size)
    return source, hashlib.sha256(payload).hexdigest()


def run_centralized_pool_experiment(config: CentralizedPoolConfig, progress=None) -> dict:
    """One exact global draw per size for EVERY saved trial, no local stage.

    Uses a dense (N,N) float64 L as a deliberately centralized oracle. Each
    output is an exact ideal-arithmetic k-DPP marginal, not an optimum. Shared
    rejection proposals correlate draws across sizes. Source distributed RNG
    streams are not reused: SeedSequence([seed, trial_id, 4, 0]) is reserved for
    this centralized baseline. Data pairing is not sample/ID pairing.
    """
    started = perf_counter()
    source, source_hash = read_distributed_source(config.source.results)
    data = DataConfig(**source["config"]["data"])
    kernel = KernelConfig(**source["config"]["kernel"])
    numerics = NumericsConfig(**source["config"]["numerics"])
    sizes = config.sweep.sizes()
    if max(sizes) > data.global_collection_size:
        raise ValueError("maximum_selection_size exceeds the saved global dataset size")
    trials = []
    for saved in source["trials"]:
        trial_started = perf_counter()
        number = saved["trial"]
        if progress is not None:
            progress({"stage": "centralized_start", "trial": number,
                      "population_size": data.global_collection_size, "num_sizes": len(sizes)})
        encoded = np.asarray([[data.alphabet.index(symbol) for symbol in word]
                              for word in saved["global_collection"]], dtype=np.int64)
        matrix = hamming_rbf(encoded, kernel.gamma, kernel.jitter)
        rng_components = [config.experiment.seed, number, 4, 0]
        rng = np.random.default_rng(np.random.SeedSequence(rng_components))
        def trial_progress(row):
            if progress is not None:
                progress({**row, "trial": number})
        sweep = exact_rejection_sweep(matrix, sizes, rng, numerics, trial_progress)
        # Every saved set is checked for cardinality, uniqueness and global
        # index bounds; direct Cholesky independently spot-checks its score.
        for row in sweep["records"]:
            ids = selected_indices(row, sweep["permutations"])
            if (len(ids) != row["selection_size"] or len(np.unique(ids)) != len(ids)
                    or np.any(ids < 0) or np.any(ids >= len(encoded))):
                raise RuntimeError("Centralized sample violated cardinality, uniqueness or eligibility")
        checks = []
        probes = np.unique(np.linspace(0, len(sizes) - 1, min(len(sizes), 5), dtype=int))
        for probe in probes:
            row = sweep["records"][probe]
            ids = selected_indices(row, sweep["permutations"])
            direct = stable_logdet(matrix[np.ix_(ids, ids)])
            error = abs(direct - row["logdet"])
            if not np.isfinite(direct) or error > numerics.psd_tolerance:
                raise ValueError("Centralized direct logdet check failed")
            checks.append({"selection_size": row["selection_size"], "absolute_logdet_error": error})
        distributed = {row["selection_size"]: row["logdet"] for row in saved["sweep"]["records"]}
        comparison = [{"selection_size": row["selection_size"], "centralized_logdet": row["logdet"],
                       "distributed_logdet": distributed[row["selection_size"]],
                       "logdet_gap_central_minus_distributed": row["logdet"] - distributed[row["selection_size"]]}
                      for row in sweep["records"] if row["selection_size"] in distributed]
        trials.append({
            "trial": number, "collection_sha256": saved["collection_sha256"],
            "global_collection": saved["global_collection"], "population_size": len(encoded),
            "distributed_pool_size": saved["pool_size"], "rng_seed_components": rng_components,
            "sweep": sweep, "comparison": comparison,
            "validation": {"all_sample_invariants_passed": True, "sample_count": len(sizes),
                           "direct_logdet_checks": checks},
            "dense_kernel_bytes": matrix.nbytes, "total_seconds": perf_counter() - trial_started,
        })
        del matrix
        if progress is not None:
            progress({"stage": "centralized_complete", "trial": number,
                      "last_logdet": sweep["records"][-1]["logdet"],
                      "seconds": trials[-1]["total_seconds"]})
    return {
        "schema": "centralized_saved_global_pool_exact_size_sweep_v1",
        "config": config.as_dict(),
        "inherited_config": {name: source["config"][name] for name in ("data", "kernel", "numerics")},
        "source": {"results": str(Path(config.source.results).resolve()), "sha256": source_hash,
                   "schema": source["schema"], "trial_count": len(source["trials"])},
        "method": {
            "target": "P(Y | |Y|=b) = det(L_Y) / e_b(eigenvalues(L_S)), Y subset of all saved global items",
            "sampler": "shared exact rejection extension, not the DPPy spectral API",
            "context": "all G0_i empty; context-free non-private centralized reference",
            "dataset_policy": "reuse exact saved words and row/global-ID order; verify collection and source hashes",
            "kernel_policy": "inherit gamma, jitter, dtype and tolerances from distributed source",
            "draw_policy": "one draw per size/trial; coupled across sizes; separate RNG stream from distributed run",
            "comparison": "paired datasets, common sizes only; independent central and distributed draws",
            "communication_rounds": 0, "communication_bytes": 0,
            "privacy_guarantee": False, "optimization_guarantee": False,
            "dense_kernel_scope": "full global dense kernel allowed only in this centralized oracle",
        }, "trials": trials, "total_seconds": perf_counter() - started,
    }
