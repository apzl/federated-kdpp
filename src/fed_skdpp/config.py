"""Typed, validated settings for a single experiment configuration."""

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path
from typing import Any

import yaml


def _positive_int(name: str, value: int) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


@dataclass(frozen=True)
class DataConfig:
    alphabet: str = "ACGT"
    kmer_length: int = 6
    global_collection_size: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.alphabet, str) or len(self.alphabet) < 2:
            raise ValueError("alphabet must contain at least two distinct characters")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet characters must be unique")
        _positive_int("kmer_length", self.kmer_length)
        _positive_int("global_collection_size", self.global_collection_size)
        if self.global_collection_size > len(self.alphabet) ** self.kmer_length:
            raise ValueError("global_collection_size exceeds the number of unique k-mers")


@dataclass(frozen=True)
class PartitionConfig:
    num_clients: int = 10

    def __post_init__(self) -> None:
        _positive_int("num_clients", self.num_clients)


@dataclass(frozen=True)
class SelectionConfig:
    client_selection_size: int = 50
    final_selection_size: int = 50
    forced_set_size: int = 0
    final_objective: str = "regular"

    def __post_init__(self) -> None:
        _positive_int("client_selection_size", self.client_selection_size)
        _positive_int("final_selection_size", self.final_selection_size)
        if type(self.forced_set_size) is not int or self.forced_set_size < 0:
            raise ValueError("forced_set_size must be a nonnegative integer")
        if not isinstance(self.final_objective, str) or self.final_objective not in {"regular", "pooled_context"}:
            raise ValueError("final_objective must be regular or pooled_context")


@dataclass(frozen=True)
class KernelConfig:
    name: str = "hamming_rbf"
    gamma: float = 0.5
    jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.name != "hamming_rbf":
            raise ValueError("Only hamming_rbf is implemented in this first experiment")
        _nonnegative_float("gamma", self.gamma)
        if self.gamma == 0:
            raise ValueError("gamma must be strictly positive")
        _nonnegative_float("jitter", self.jitter)


@dataclass(frozen=True)
class SearchConfig:
    algorithm: str = "stochastic_local_search"
    patience: int = 50
    improvement_tolerance: float = 1e-10
    max_iterations: int | None = None
    initialization: str = "paired_uniform"
    chain_steps: int = 10000
    burn_in: int = 2000
    exact_mode: str = "KuTa12"
    local_proposal_budget: int | None = None
    global_proposal_budget: int | None = None

    def __post_init__(self) -> None:
        if self.algorithm not in {"stochastic_local_search", "dppy_mcmc", "dppy_exact"}:
            raise ValueError("Use stochastic_local_search, dppy_mcmc (MCMC), or dppy_exact (spectral sampling)")
        if self.exact_mode not in {"KuTa12", "GS", "GS_bis"}:
            raise ValueError("exact_mode must be KuTa12, GS, or GS_bis")
        _positive_int("patience", self.patience)
        _nonnegative_float("improvement_tolerance", self.improvement_tolerance)
        if self.max_iterations is not None:
            _positive_int("max_iterations", self.max_iterations)
        for name in ("local_proposal_budget", "global_proposal_budget"):
            budget = getattr(self, name)
            if budget is not None:
                _positive_int(name, budget)
                if self.algorithm != "stochastic_local_search":
                    raise ValueError(f"{name} applies only to stochastic_local_search")
                if self.max_iterations is not None and budget > self.max_iterations:
                    raise ValueError(f"{name} exceeds the max_iterations safety cap")
        if self.initialization not in {"paired_uniform", "independent_uniform"}:
            raise ValueError("initialization must be paired_uniform or independent_uniform")
        _positive_int("chain_steps", self.chain_steps)
        if type(self.burn_in) is not int or self.burn_in < 0:
            raise ValueError("burn_in must be a nonnegative integer")
        if self.algorithm == "dppy_mcmc" and self.burn_in >= self.chain_steps:
            raise ValueError("burn_in must be smaller than chain_steps")


@dataclass(frozen=True)
class NumericsConfig:
    symmetry_tolerance: float = 1e-12
    psd_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        _nonnegative_float("symmetry_tolerance", self.symmetry_tolerance)
        _nonnegative_float("psd_tolerance", self.psd_tolerance)


@dataclass(frozen=True)
class TrialConfig:
    seed: int = 42
    repetitions: int = 5

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        _positive_int("repetitions", self.repetitions)


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "results"
    save_history: bool = True
    save_chain: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.directory, str) or not self.directory.strip():
            raise ValueError("output directory must be a nonempty path")
        if type(self.save_history) is not bool:
            raise ValueError("save_history must be true or false")
        if type(self.save_chain) is not bool:
            raise ValueError("save_chain must be true or false")


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = DataConfig()
    partition: PartitionConfig = PartitionConfig()
    selection: SelectionConfig = SelectionConfig()
    kernel: KernelConfig = KernelConfig()
    search: SearchConfig = SearchConfig()
    numerics: NumericsConfig = NumericsConfig()
    experiment: TrialConfig = TrialConfig()
    output: OutputConfig = OutputConfig()

    def __post_init__(self) -> None:
        if self.search.algorithm == "dppy_exact" and self.output.save_chain:
            raise ValueError("dppy_exact produces one sample per stage; save_chain must be false")
        if self.data.global_collection_size % self.partition.num_clients:
            raise ValueError("global_collection_size must be divisible by num_clients")
        if self.selection.client_selection_size > self.client_collection_size:
            raise ValueError("client_selection_size exceeds a client's collection size")
        if self.selection.client_selection_size + self.selection.forced_set_size > self.client_collection_size:
            raise ValueError("client_selection_size plus forced_set_size exceeds a client's collection size")
        if self.selection.final_selection_size > self.data.global_collection_size:
            raise ValueError("final_selection_size exceeds the global collection size")
        if self.selection.final_selection_size > self.server_pool_size:
            raise ValueError("final_selection_size exceeds the server candidate pool size")
        eligible_size = self.data.global_collection_size - self.partition.num_clients * self.selection.forced_set_size
        if self.selection.final_selection_size > eligible_size:
            raise ValueError("final_selection_size exceeds the globally eligible collection size")

    @property
    def client_collection_size(self) -> int:
        return self.data.global_collection_size // self.partition.num_clients

    @property
    def server_pool_size(self) -> int:
        return self.partition.num_clients * self.selection.client_selection_size

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML; reject unknown keys rather than silently ignoring typos."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config.yml must contain a mapping")
    section_types = {
        "data": DataConfig, "partition": PartitionConfig,
        "selection": SelectionConfig, "kernel": KernelConfig,
        "search": SearchConfig, "numerics": NumericsConfig,
        "experiment": TrialConfig, "output": OutputConfig,
    }
    unknown = set(raw) - set(section_types)
    if unknown:
        raise ValueError(f"Unknown config sections: {sorted(unknown)}")
    parsed = {}
    for name, section_type in section_types.items():
        section = raw.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping")
        unknown = set(section) - {field.name for field in fields(section_type)}
        if unknown:
            raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
        parsed[name] = section_type(**section)
    return ExperimentConfig(**parsed)
