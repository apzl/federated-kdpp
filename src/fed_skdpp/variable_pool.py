"""Non-private ordinary-local-DPP / exact-server-k-DPP baseline experiment."""

from dataclasses import asdict, dataclass, fields
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

from .config import DataConfig, KernelConfig, NumericsConfig, OutputConfig, PartitionConfig, TrialConfig
from .data import decode_kmers, partition_indices, sample_kmers
from .kernels import hamming_rbf, stable_logdet
from .variable_dpp import OrdinaryDPPSampler, exact_rejection_sweep, prefix_logdets, selected_indices, sweep_sizes


@dataclass(frozen=True)
class PoolSweepConfig:
    minimum_selection_size: int = 10
    selection_size_interval: int = 5

    def __post_init__(self):
        for name in ("minimum_selection_size", "selection_size_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class VariablePoolConfig:
    data: DataConfig = DataConfig(kmer_length=20, global_collection_size=10000)
    partition: PartitionConfig = PartitionConfig(num_clients=10)
    kernel: KernelConfig = KernelConfig(gamma=0.6)
    sweep: PoolSweepConfig = PoolSweepConfig()
    numerics: NumericsConfig = NumericsConfig()
    experiment: TrialConfig = TrialConfig(seed=42, repetitions=3)
    output: OutputConfig = OutputConfig(directory="../results", save_history=False)

    def __post_init__(self):
        if self.data.global_collection_size % self.partition.num_clients:
            raise ValueError("global_collection_size must be divisible by num_clients")
        if self.output.save_chain or self.output.save_history:
            raise ValueError("This exact-sampling experiment has no optimization history or MCMC chain")

    def as_dict(self):
        return asdict(self)


def load_variable_pool_config(path: str | Path) -> VariablePoolConfig:
    """Strict YAML config, separate from the older fixed-local-size experiments."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    section_types = {
        "data": DataConfig, "partition": PartitionConfig, "kernel": KernelConfig,
        "sweep": PoolSweepConfig, "numerics": NumericsConfig,
        "experiment": TrialConfig, "output": OutputConfig,
    }
    if not isinstance(raw, dict) or set(raw) - set(section_types):
        raise ValueError("Expected a mapping of known variable-pool config sections")
    defaults = VariablePoolConfig()
    parsed = {}
    for name, kind in section_types.items():
        section = raw.get(name, {})
        if not isinstance(section, dict) or set(section) - {field.name for field in fields(kind)}:
            raise ValueError(f"Invalid or unknown {name} settings")
        parsed[name] = kind(**{**asdict(getattr(defaults, name)), **section})
    return VariablePoolConfig(**parsed)


def _rng(seed: int, trial: int, stage: int, client: int = 0) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, trial, stage, client]))


def attach_client_diagnostics(
    matrix: np.ndarray, pool_owner: np.ndarray, num_clients: int, sweep: dict,
) -> None:
    """Add logdet(L_[Y intersect S_i]) and cardinality for EVERY client and k.

    matrix is the public float64 (m,m) pool L; pool_owner is an int64 (m,)
    mapping into [0,num_clients). Empty intersections have logdet zero. This
    is a contribution diagnostic, not a context-aware gain or client utility
    for the full shared Y. Scores of unequal-size intersections are confounded
    by size; counts are therefore retained and plotted alongside logdet.
    """
    if (pool_owner.shape != (len(matrix),) or pool_owner.dtype.kind not in "iu"
            or np.any(pool_owner < 0) or np.any(pool_owner >= num_clients)):
        raise ValueError("pool_owner must map every matrix row to a valid client")
    by_permutation: dict[tuple[int, bool], list[dict]] = {}
    for record in sweep["records"]:
        reverse = record["method"] == "complement"
        by_permutation.setdefault((record["permutation_index"], reverse), []).append(record)
    for (index, reverse), rows in by_permutation.items():
        order = np.asarray(sweep["permutations"][index], dtype=np.int64)
        if reverse:
            order = order[::-1]
        owners = pool_owner[order]
        # Reuse one client-prefix Cholesky per accepted permutation/direction.
        prefixes = []
        for client in range(num_clients):
            ids = order[owners == client]
            prefixes.append(prefix_logdets(matrix, ids, len(ids)))
        for row in rows:
            ids = order[:row["selection_size"]]
            counts = np.bincount(pool_owner[ids], minlength=num_clients)
            row["client_selection_sizes"] = counts.tolist()
            row["client_logdets"] = [float(prefixes[client][count]) for client, count in enumerate(counts)]
            if sum(row["client_selection_sizes"]) != row["selection_size"]:
                raise RuntimeError("Client contribution cardinalities do not sum to server size")


def run_variable_pool_trial(config: VariablePoolConfig, trial: int, progress=None) -> dict:
    """One fresh IID dataset and partition; fixed local pool across the k sweep."""
    started = perf_counter()
    seed = config.experiment.seed
    encoded = sample_kmers(config.data, _rng(seed, trial, 0))
    partitions = partition_indices(len(encoded), config.partition.num_clients, _rng(seed, trial, 1))
    clients = []
    for client, ids in enumerate(partitions):
        local_kernel = hamming_rbf(encoded[ids], config.kernel.gamma, config.kernel.jitter)
        sampler = OrdinaryDPPSampler(local_kernel, config.numerics)
        selected, score, diagnostics = sampler.sample(_rng(seed, trial, 2, client))
        clients.append({"client": client, "population_size": len(ids),
                        "selected_ids": ids[selected].tolist(), "logdet": score, **diagnostics})
        if progress is not None:
            progress({"stage": "local_dpp", "trial": trial, "client": client,
                      "selection_size": len(selected), "logdet": score})
        del sampler, local_kernel
    pool_ids = np.sort(np.asarray([item for client in clients for item in client["selected_ids"]], dtype=np.int64))
    if len(np.unique(pool_ids)) != len(pool_ids):
        raise RuntimeError("Disjoint client partitions unexpectedly produced duplicate pool IDs")
    owner = np.empty(len(encoded), dtype=np.int64)
    for client, ids in enumerate(partitions):
        owner[ids] = client
    pool_owner = owner[pool_ids]
    sizes = sweep_sizes(len(pool_ids), config.sweep.minimum_selection_size, config.sweep.selection_size_interval)
    if progress is not None:
        progress({"stage": "pool", "trial": trial, "pool_size": len(pool_ids), "num_sizes": len(sizes)})
    matrix = hamming_rbf(encoded[pool_ids], config.kernel.gamma, config.kernel.jitter)
    sweep = exact_rejection_sweep(matrix, sizes, _rng(seed, trial, 3), config.numerics, progress)
    scoring_started = perf_counter()
    attach_client_diagnostics(matrix, pool_owner, config.partition.num_clients, sweep)
    # Independent direct-Cholesky checks of both rejection branches and the
    # endpoint. Keep numerical exactness separate from ideal-law exactness.
    indices = np.unique(np.linspace(0, len(sizes) - 1, min(len(sizes), 5), dtype=int))
    errors = []
    for index in indices:
        row = sweep["records"][index]
        selected = selected_indices(row, sweep["permutations"])
        direct = stable_logdet(matrix[np.ix_(selected, selected)])
        error = abs(direct - row["logdet"])
        if not np.isfinite(direct) or error > config.numerics.psd_tolerance:
            raise ValueError("Server direct and complementary logdet disagree beyond tolerance")
        errors.append(error)
    sweep["diagnostics"]["direct_cholesky_validation_count"] = len(errors)
    sweep["diagnostics"]["direct_cholesky_max_error"] = max(errors, default=0.)
    sweep["diagnostics"]["client_scoring_and_validation_seconds"] = perf_counter() - scoring_started
    # IDs are logical int64 payloads; actual transport/framing is not measured.
    upload_bytes = int(pool_ids.nbytes)
    broadcast_bytes = config.partition.num_clients * 8 * sum(sizes)
    return {
        "trial": trial, "collection_sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
        "global_collection": decode_kmers(encoded, config.data.alphabet),
        "client_partitions": [ids.tolist() for ids in partitions], "clients": clients,
        "pool_ids": pool_ids.tolist(), "pool_owner": pool_owner.tolist(),
        "pool_size": len(pool_ids), "sweep": sweep,
        "communication": {"logical_rounds": 2, "upload_id_payload_bytes": upload_bytes,
                          "broadcast_all_sizes_id_payload_bytes": broadcast_bytes,
                          "total_id_payload_bytes": upload_bytes + broadcast_bytes,
                          "scope": "one upload and one batched broadcast; ID payload only, no network run"},
        "total_seconds": perf_counter() - started,
    }


def run_variable_pool_experiment(config: VariablePoolConfig, progress=None) -> dict:
    """Three (configurable) independent trials, one exact draw per k per trial."""
    started = perf_counter()
    trials = [run_variable_pool_trial(config, trial, progress) for trial in range(config.experiment.repetitions)]
    return {
        "schema": "ordinary_local_dpp_exact_server_size_sweep_v1", "config": config.as_dict(),
        "method": {
            "local_law": "independent ordinary L-ensemble DPP per client: det(L_A)/det(I+L)",
            "server_law": "conditional on realized pool P, det(L_Y)/e_b(eigenvalues(L_P)), |Y|=b",
            "local_sampler": "native DPPy ordinary DPP, GS spectral/projection sampler",
            "server_sampler": "exact coupled rejection; not DPPy's spectral sampler",
            "contexts": "all G0_i are empty; context-free non-private aggregation baseline",
            "pool_size": "m is random: sum of local ordinary-DPP sample sizes",
            "data": "unique uniform DNA strings; reverse complements distinct; random disjoint IID partition",
            "trial_policy": "fresh data, partitions and local draws per trial; same pool across all sizes within trial",
            "draw_policy": "one exact server draw per size per trial; coupled, not independent or necessarily nested across sizes",
            "size_grid": "minimum, minimum+interval, ... <=m; append full-pool endpoint m if off-grid",
            "small_pool_policy": "if m<minimum, report endpoint m only; never redraw or drop a trial",
            "client_score": "logdet(L_[Y intersect original client partition]); empty set score is zero",
            "boxplot": "ten client contribution scores at each size in each trial, not ten repeated local DPP draws",
            "privacy": "none: public synthetic data and plaintext client-selected IDs; no cryptography or hidden G0",
            "server_observes": "public features, client-labeled local selections and every shared server selection",
            "participation": "all clients participate; no failures or non-IID model",
            "warning": "pool-conditioned draws are not claimed to follow the full-10000-item k-DPP law",
        }, "trials": trials, "total_seconds": perf_counter() - started,
    }
