"""SLS proposal-budget calibration, first centralized and then client-local.

References: expansion paper Section III Algorithm 1 (uniform nondecreasing
swaps), Section V-A (proposal-budget/trajectory study); k-DPP paper Section 3
(fixed-cardinality logdet MAP vs sampling); MCMC paper Section 1 (sampling
accepts downhill transitions, unlike SLS). Exhaustive checkpoint stopping is
our explicitly labelled extension, not a paper replication or global guarantee.
"""

import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import yaml

from .backends import KernelObjective
from .config import ExperimentConfig, NumericsConfig, SearchConfig, _positive_int
from .data import decode_kmers, partition_indices, sample_kmers, split_context
from .experiment import _rng
from .objectives import ForcedSetObjective
from .search import SearchResult, SearchStep, stochastic_local_search
from .swap_audit import audit_one_swap_neighborhood


@dataclass(frozen=True)
class BudgetConfig:
    max_proposals: int = 100000
    checkpoint_interval: int = 1000
    reference_patience: int = 50

    def __post_init__(self) -> None:
        for name in ("max_proposals", "checkpoint_interval", "reference_patience"):
            _positive_int(name, getattr(self, name))
        if self.checkpoint_interval > self.max_proposals:
            raise ValueError("checkpoint_interval must fit max_proposals")

    def as_dict(self) -> dict:
        return asdict(self)


def load_budget_config(path: str | Path) -> BudgetConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or set(raw) - set(BudgetConfig.__dataclass_fields__):
        raise ValueError("budget config contains unknown settings or is not a mapping")
    return BudgetConfig(**raw)


def audited_stochastic_local_search(
    objective: ForcedSetObjective, selection_size: int, rng: np.random.Generator,
    settings: SearchConfig, budget: BudgetConfig,
    numerics: NumericsConfig = NumericsConfig(),
    initial_selection: np.ndarray | None = None,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> tuple[SearchResult, list[dict]]:
    """Same SLS proposal/acceptance path; disable patience and audit checkpoints.

    initial_selection and result.selected use eligible-local indices, not L's
    indices. Checkpoints contain those same local G1 indices, never modify G0,
    apply an audit-selected swap, or consume RNG. At the final cap, audit also
    runs if it is not an interval multiple. History is retained for this study.
    Earliest CHECKED certificate is reported, not the exact time it first became
    true, a universal iteration budget, or a globally optimal selection.
    """
    if settings.algorithm != "stochastic_local_search":
        raise ValueError("Budget study requires stochastic_local_search, not MCMC")
    active = replace(settings, patience=budget.max_proposals + 1, max_iterations=budget.max_proposals)
    checkpoints = []

    def check(step: SearchStep, selected: np.ndarray) -> str | None:
        audit = audit_one_swap_neighborhood(
            objective.matrix, objective.forced_indices, objective.selected_indices(selected),
            settings.improvement_tolerance, numerics=numerics,
        )
        record = {"iteration": step.iteration, "logdet": step.logdet, **asdict(audit),
                  "selected_ids": selected.tolist()}
        checkpoints.append(record)
        if on_checkpoint is not None:
            on_checkpoint(record)
        return "one_swap_local_optimum" if audit.certified else None

    def observe(step: SearchStep, selected: np.ndarray) -> str | None:
        return check(step, selected) if step.iteration % budget.checkpoint_interval == 0 else None

    search = stochastic_local_search(
        len(objective.eligible_indices), selection_size, objective, rng, active,
        initial_selection=initial_selection, save_history=True, on_step=observe,
    )
    if not checkpoints or checkpoints[-1]["iteration"] != search.iterations:
        last = search.history[-1]
        reason = check(last, np.asarray(search.selected, dtype=np.int64))
        if reason is not None:
            search = replace(search, stop_reason=reason)
    return search, checkpoints


def _run_budget_stage(
    encoded: np.ndarray, candidate_ids: np.ndarray, context_ids: np.ndarray,
    selection_size: int, config: ExperimentConfig, budget: BudgetConfig,
    backend: str, rng: np.random.Generator, initial_ids: np.ndarray | None,
    trace_path: Path,
    on_checkpoint: Callable[[dict], None] | None = None,
) -> dict:
    started = perf_counter()
    candidates, context = np.sort(candidate_ids), np.sort(context_ids)
    if np.intersect1d(candidates, context).size:
        raise ValueError("G0 must be disjoint from stage candidates")
    item_ids = np.concatenate((context, candidates))
    kernel = KernelObjective(encoded[item_ids], config.kernel, config.numerics, backend)
    objective = ForcedSetObjective(kernel.matrix, np.arange(len(context), dtype=np.int64),
                                   config.numerics.symmetry_tolerance, config.numerics.psd_tolerance)
    initial = None
    if initial_ids is not None:
        if not np.isin(initial_ids, candidates).all():
            raise ValueError("Initial G1 must belong to candidates")
        initial = np.searchsorted(candidates, initial_ids)
    ready = perf_counter()
    search, checkpoints = audited_stochastic_local_search(
        objective, selection_size, rng, config.search, budget, config.numerics, initial, on_checkpoint,
    )
    finished = perf_counter()
    scores = np.asarray([step.logdet for step in search.history], dtype=np.float64)
    accepted = np.asarray([step.accepted for step in search.history], dtype=np.bool_)
    improved = np.asarray([step.improved for step in search.history], dtype=np.bool_)
    stalled = np.asarray([step.stalled_iterations for step in search.history], dtype=np.int64)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("xb") as handle:
        np.savez_compressed(handle, logdet=scores, accepted=accepted, improved=improved,
                            stalled_iterations=stalled)
    # The unchanged history prefix identifies the counterfactual patience stop.
    stopping = np.flatnonzero(stalled >= budget.reference_patience)
    baseline_iteration = int(stopping[0]) if len(stopping) else None
    within = np.flatnonzero(scores >= search.logdet - config.search.improvement_tolerance)
    improved_indices = np.flatnonzero(improved)
    audit_seconds = sum(row["audit_seconds"] for row in checkpoints)
    for row in checkpoints:
        row["selected_ids"] = candidates[row["selected_ids"]].tolist()
        for name in ("best_in", "best_out"):
            if row[name] is not None:
                row[name] = int(item_ids[row[name]])
    certified = checkpoints[-1]["certified"]
    return {
        "selected_ids": candidates[list(search.selected)].tolist(),
        "initial_ids": candidates[list(search.initial_selection)].tolist(),
        "selection_size": selection_size, "context_size": len(context),
        "candidate_count": len(candidates), "joint_selection_size": len(context) + selection_size,
        "objective": "joint_logdet(G0 union G1)",
        "initial_logdet": search.initial_logdet, "logdet": search.logdet,
        "context_logdet": objective.context_logdet,
        "conditional_gain": search.logdet - objective.context_logdet,
        "iterations": search.iterations, "accepted_swaps": search.accepted_swaps,
        "strict_improvements": search.strict_improvements, "stop_reason": search.stop_reason,
        "certified": certified,
        "first_certified_checkpoint": search.iterations if certified else None,
        "last_meaningful_improvement_iteration": int(improved_indices[-1]) if len(improved_indices) else 0,
        "first_within_tolerance_of_terminal_iteration": int(within[0]),
        "reference_patience": budget.reference_patience,
        "reference_patience_stop_iteration": baseline_iteration,
        "reference_patience_logdet": float(scores[baseline_iteration]) if baseline_iteration is not None else None,
        "gain_after_reference_patience": float(search.logdet - scores[baseline_iteration]) if baseline_iteration is not None else None,
        "final_max_swap_gain": checkpoints[-1]["max_logdet_gain"],
        "final_improving_swaps": checkpoints[-1]["improving_swaps"],
        "kernel_setup_seconds": ready - started, "search_and_audits_seconds": finished - ready,
        "audit_seconds": audit_seconds,
        "num_audits": len(checkpoints),
        "audited_swap_pairs": sum(row["total_swaps"] for row in checkpoints),
        "optimizer_seconds_estimate": max(0., finished - ready - audit_seconds),
        "dense_candidate_kernel_bytes": kernel.matrix.nbytes,
        "trace_file": str(trace_path.resolve()),
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "trace_length": len(scores), "checkpoints": checkpoints,
    }


def stage_summary(stages: list[dict]) -> dict:
    """Descriptive dispersion; client states are nested in paired data trials."""
    iterations = np.asarray([stage["iterations"] for stage in stages], dtype=np.float64)
    gains = np.asarray([stage["gain_after_reference_patience"] for stage in stages
                        if stage["gain_after_reference_patience"] is not None])
    return {
        "num_states": len(stages), "certified_states": sum(stage["certified"] for stage in stages),
        "capped_states": sum(not stage["certified"] for stage in stages),
        "proposal_count": {"mean": float(iterations.mean()), "median": float(np.median(iterations)),
                           "std": float(iterations.std(ddof=1)) if len(stages) > 1 else 0.,
                           "min": int(iterations.min()), "max": int(iterations.max())},
        "mean_gain_after_reference_patience": float(gains.mean()) if len(gains) else None,
    }


def run_budget_experiment(
    config: ExperimentConfig, budget: BudgetConfig, backend: str, directory: Path,
    on_stage: Callable[[str, int, int | None, dict], None] | None = None,
    on_checkpoint: Callable[[str, int, int | None, dict], None] | None = None,
) -> dict:
    """Run all centralized stages FIRST, then clients from the same trial data.

    No client suggestions/server selection/transport is executed. Centralized
    optimizer accesses pooled G0 only as a named NON-PRIVATE reference; each
    local run uses its own fixed context. Central G1 initialization is uniform
    over the FULL eligible global universe, independent of local preprocessing.
    Data/context/proposal stream identifiers follow the existing experiment;
    local SLS prefixes match earlier seed-paired runs. Exact diagnostic outputs
    and synthetic context memberships are trusted non-private research artifacts.
    """
    if backend not in {"numpy", "dppy"}:
        raise ValueError("backend must be numpy or dppy")
    if config.search.algorithm != "stochastic_local_search":
        raise ValueError("Budget experiment requires stochastic_local_search")
    if config.selection.final_objective != "pooled_context":
        raise ValueError("This budget study requires the declared non-private pooled_context reference")
    trials, data = [], []
    seed = config.experiment.seed
    for trial in range(config.experiment.repetitions):
        encoded = sample_kmers(config.data, _rng(seed, trial, 0))
        partitions = partition_indices(len(encoded), config.partition.num_clients, _rng(seed, trial, 1))
        local = [split_context(part, config.selection.forced_set_size, _rng(seed, trial, 5, index))
                 for index, part in enumerate(partitions)]
        pooled = np.sort(np.concatenate([context for context, _ in local]))
        eligible = np.sort(np.concatenate([candidates for _, candidates in local]))
        kmers = decode_kmers(encoded, config.data.alphabet)
        trials.append({
            "trial": trial, "global_collection": kmers,
            "collection_sha256": hashlib.sha256("\n".join(kmers).encode()).hexdigest(),
            "client_partitions": [part.tolist() for part in partitions],
            "client_disclosed_context_ids": [context.tolist() for context, _ in local],
            "pooled_context_ids": pooled.tolist(), "clients": [],
        })
        data.append((encoded, pooled, eligible, local))
    for trial, (encoded, pooled, eligible, _) in enumerate(data):
        initial = np.sort(_rng(seed, trial, 2).choice(eligible, config.selection.final_selection_size, replace=False))
        callback = None if on_checkpoint is None else lambda row, t=trial: on_checkpoint("centralized", t, None, row)
        stage = _run_budget_stage(encoded, eligible, pooled, config.selection.final_selection_size,
                                  config, budget, backend, _rng(seed, trial, 3), initial,
                                  directory / "traces" / f"trial_{trial}" / "centralized.npz", callback)
        trials[trial]["centralized"] = stage
        if on_stage is not None:
            on_stage("centralized", trial, None, stage)
    for trial, (encoded, _, _, local) in enumerate(data):
        for client, (context, candidates) in enumerate(local):
            callback = None if on_checkpoint is None else lambda row, t=trial, c=client: on_checkpoint("client", t, c, row)
            stage = _run_budget_stage(encoded, candidates, context, config.selection.client_selection_size,
                                      config, budget, backend, _rng(seed, trial, 10, client), None,
                                      directory / "traces" / f"trial_{trial}" / f"client_{client}.npz", callback)
            trials[trial]["clients"].append(stage)
            if on_stage is not None:
                on_stage("client", trial, client, stage)
    centralized = [trial["centralized"] for trial in trials]
    clients = [stage for trial in trials for stage in trial["clients"]]
    all_central_certified = all(stage["certified"] for stage in centralized)
    tested_transfer_budget = max(stage["iterations"] for stage in centralized)
    return {
        "backend": backend, "config": config.as_dict(), "budget_config": budget.as_dict(),
        "effective_search_config": asdict(replace(config.search, patience=budget.max_proposals + 1,
                                                  max_iterations=budget.max_proposals)),
        "scope": "NON-PRIVATE centralized and client-local SLS convergence study; no federated/server protocol run",
        "reference": "expansion paper Section III Algorithm 1 and Section V-A; checkpoint stopping is an extension",
        "algorithm": "unchanged uniform proposals and nondecreasing acceptance; patience disabled; no audit moves applied",
        "convergence": "first checked checkpoint with no one-swap logdet increase > configured tolerance, not global optimality",
        "initialization": "central G1 uniform from all globally eligible items; not coupled to a client suggestion pool",
        "pairing": "same trial data, partitions and contexts as prior seed-paired experiment; all centralized runs precede client runs",
        "comparison": "joint logdet uses different cardinalities/context sizes; compare proposal requirements, not absolute scores across scales",
        "communication_rounds": 0, "transmitted_bytes": 0,
        "privacy_guarantee": False, "private_protocol_evaluated": False,
        "dispersion": "descriptive sample SD; 30 client states nested within three datasets when configured, not 30 independent data trials",
        "trials": trials, "summary": {"centralized": stage_summary(centralized), "clients": stage_summary(clients)},
        "transfer": {
            "centralized_budget_certified_for_all_trials": all_central_certified,
            "max_observed_centralized_stop_budget": tested_transfer_budget,
            "recommended_empirical_centralized_budget": tested_transfer_budget if all_central_certified else None,
            "clients_certified_by_observed_centralized_budget": sum(stage["certified"] and stage["iterations"] <= tested_transfer_budget for stage in clients),
            "caveat": "sample-specific tested budget, not a universal optimum; capped central runs cannot establish a sufficient centralized budget",
        },
    }


def write_budget_results(result: dict, directory: Path) -> None:
    """Write summary/checkpoints only after traces, refusing existing outputs."""
    names = ("results.json", "config.yml", "budget.yml", "stages.csv", "checkpoints.csv")
    if any((directory / name).exists() for name in names):
        raise FileExistsError("Refusing to overwrite existing budget-study results")
    with (directory / "results.json").open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    for name, payload in (("config.yml", result["config"]), ("budget.yml", result["budget_config"])):
        with (directory / name).open("x", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
    stages, checkpoints = [], []
    for trial in result["trials"]:
        entries = [("centralized", None, trial["centralized"])]
        entries.extend(("client", client, stage) for client, stage in enumerate(trial["clients"]))
        for kind, client, stage in entries:
            prefix = {"stage": kind, "trial": trial["trial"], "client": client}
            stages.append({**prefix, **{key: value for key, value in stage.items()
                                        if key not in {"selected_ids", "initial_ids", "checkpoints"}}})
            checkpoints.extend({**prefix, **{key: value for key, value in row.items() if key != "selected_ids"}}
                               for row in stage["checkpoints"])
    for name, rows in (("stages.csv", stages), ("checkpoints.csv", checkpoints)):
        with (directory / name).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
