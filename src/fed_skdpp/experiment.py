"""Forced-set stochastic search and two-stage suggestion baselines.

``pooled_context`` deliberately shares every client context with the server
and optimizes joint logdet with that same fixed pooled context in both final
searches. This is a NON-PRIVATE pooled-context baseline, not the repository's
private federated objective and not an aggregation of individual client gains.
``regular`` retains the earlier G1-only final objectives without disclosure.
"""

from dataclasses import asdict
import hashlib
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from .backends import KernelObjective
from .config import ExperimentConfig
from .data import decode_kmers, partition_indices, sample_kmers, split_context
from .exact import sample_conditional_exact_k_dpp
from .kernels import hamming_rbf, stable_logdet
from .metrics import hamming_comparison_definition, jaccard, nearest_neighbor_hamming, summarize
from .mcmc import sample_conditional_k_dpp
from .objectives import ForcedSetObjective
from .search import stochastic_local_search


def _rng(seed: int, trial: int, stream: int, client: int = 0) -> np.random.Generator:
    """Independent deterministic streams; neither backend consumes shared RNG."""
    return np.random.default_rng(np.random.SeedSequence([seed, trial, stream, client]))


def _run_stage(
    encoded: np.ndarray, candidate_ids: np.ndarray, selection_size: int,
    config: ExperimentConfig, backend: str, rng: np.random.Generator,
    initial_ids: np.ndarray | None = None,
    forced_ids: np.ndarray | None = None,
    chain_path: Path | None = None,
    proposal_budget: int | None = None,
) -> dict:
    """Search candidate-local indices, then map all reported indices to S."""
    started = perf_counter()
    candidate_ids = np.sort(candidate_ids)
    context = np.empty(0, dtype=np.int64) if forced_ids is None else np.sort(forced_ids)
    if len(np.unique(context)) != len(context) or np.intersect1d(context, candidate_ids).size:
        raise ValueError("forced IDs must be unique and disjoint from eligible candidate IDs")
    item_ids = np.concatenate((context, candidate_ids))
    kernel = KernelObjective(encoded[item_ids], config.kernel, config.numerics, backend)
    objective = kernel if forced_ids is None else ForcedSetObjective(
        kernel.matrix, np.arange(len(context), dtype=np.int64),
        config.numerics.symmetry_tolerance, config.numerics.psd_tolerance,
    )
    kernel_ready = perf_counter()
    initial_local = None
    if initial_ids is not None:
        if not np.isin(initial_ids, candidate_ids).all():
            raise ValueError("initial IDs must belong to the stage's candidate collection")
        initial_local = np.searchsorted(candidate_ids, initial_ids)
    sampling = config.search.algorithm == "dppy_mcmc"
    exact = config.search.algorithm == "dppy_exact"
    diagnostics = {}
    if exact:
        if initial_ids is not None:
            raise ValueError("Exact sampling has no initial selection")
        selected, score, diagnostics = sample_conditional_exact_k_dpp(
            kernel.matrix, np.arange(len(context), dtype=np.int64), selection_size,
            rng, config.search.exact_mode, config.numerics,
        )
        record = {
            "selected": selected.tolist(), "initial_selection": [],
            "initial_logdet": None, "logdet": score, "iterations": 0,
            "accepted_swaps": 0, "strict_improvements": 0, "stalled_iterations": 0,
            "stop_reason": "exact_sample", "history": [],
            "search_seconds": diagnostics["sampling_seconds"],
        }
    elif sampling:
        result, chain, chain_scores, diagnostics = sample_conditional_k_dpp(
            kernel.matrix, np.arange(len(context), dtype=np.int64), selection_size,
            rng, config.search, initial_selection=initial_local,
            save_history=config.output.save_history, numerics=config.numerics,
        )
        if config.output.save_chain:
            if chain_path is None:
                raise ValueError("Saving MCMC chains requires an artifact_directory")
            chain_path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation protects any existing generated artifact.
            with chain_path.open("xb") as handle:
                np.savez_compressed(handle, selected_ids=candidate_ids[chain], joint_logdet=chain_scores)
            diagnostics["chain_file"] = str(chain_path.resolve())
            diagnostics["chain_file_sha256"] = hashlib.sha256(chain_path.read_bytes()).hexdigest()
        del chain, chain_scores
    else:
        result = stochastic_local_search(
            len(candidate_ids), selection_size, objective, rng, config.search,
            initial_selection=initial_local, save_history=config.output.save_history,
            proposal_budget=proposal_budget,
        )
    if not exact:
        record = asdict(result)
    record.update(diagnostics)
    record["selected_ids"] = [int(candidate_ids[position]) for position in record.pop("selected")]
    record["initial_ids"] = [int(candidate_ids[position]) for position in record.pop("initial_selection")]
    if exact:
        record["initial_ids"] = None
    record["candidate_count"] = len(candidate_ids)
    record["selection_size"] = selection_size
    record["kernel_item_count"] = len(item_ids)
    record["objective"] = "joint_logdet" if forced_ids is not None else "regular_logdet"
    record["kernel_setup_seconds"] = kernel_ready - started
    record["total_seconds"] = perf_counter() - started
    record["dense_candidate_kernel_bytes"] = kernel.matrix.nbytes
    if config.search.algorithm == "stochastic_local_search":
        record["configured_proposal_budget"] = proposal_budget
        record["stopping_rule"] = "fixed_proposal_budget" if proposal_budget is not None else "patience_or_safety_cap"
    if forced_ids is not None:
        # Diagnostics contain counts/scores, never private context membership.
        record["forced_set_size"] = len(context)
        record["joint_selection_size"] = len(context) + selection_size
        record["context_logdet"] = objective.context_logdet
        record["conditional_gain"] = record["logdet"] - objective.context_logdet
    return record


def run_trial(
    config: ExperimentConfig, backend: str, trial: int,
    artifact_directory: Path | None = None,
) -> dict:
    """Run one paired trial on the same public universe and kernel parameters.

    Reserve forced_set_size random G0_i items locally from each disjoint
    client partition. Each client optimizes joint logdet(G0_i union G1_i),
    always uploads its G1_i suggestion IDs. Since partitions are disjoint,
    local exclusion also guarantees that suggestions exclude every G0_i.

    For pooled_context, clients ALSO disclose G0_i to the server. The union
    of those contexts is forced in both the centralized and server searches;
    only G1 is swappable. Thus both optimize and are evaluated with the SAME
    pooled-context joint logdet. This explicitly non-private baseline is not
    the private federated problem or an aggregate of individual client gains.

    For regular, contexts are retained locally and both final searches still
    optimize ordinary logdet(G1). The trusted centralized reference respects
    the same exclusion policy without giving its eligibility mask to server.

    For paired_uniform, run clients first, draw a common random initial final
    set from their union, and start both final searches from it. Centralized
    proposals remain eligible over all non-context global items. This coupling makes the
    starting set comparable, but means the central initialization uses the
    compressed pool; it is not independent of client preprocessing.

    Exact sampling has no initialization: central and server use independent
    RNG streams 3 and 4, even when their candidate collections coincide. One
    client with no compression therefore recovers the SAME conditional
    distribution, while its two independent realizations may differ.
    """
    if backend not in {"numpy", "dppy"}:
        raise ValueError("backend must be numpy or dppy")
    if config.search.algorithm in {"dppy_mcmc", "dppy_exact"} and backend != "dppy":
        raise ValueError(f"{config.search.algorithm} requires backend=dppy")
    if config.search.algorithm == "dppy_mcmc" and config.output.save_chain and artifact_directory is None:
        raise ValueError("Saving MCMC chains requires an artifact_directory")
    if type(trial) is not int or trial < 0:
        raise ValueError("trial must be a nonnegative integer")
    seed = config.experiment.seed
    pooled = config.selection.final_objective == "pooled_context"
    started = perf_counter()
    encoded = sample_kmers(config.data, _rng(seed, trial, 0))
    partitions = partition_indices(len(encoded), config.partition.num_clients, _rng(seed, trial, 1))
    local_sets = [
        split_context(partition, config.selection.forced_set_size, _rng(seed, trial, 5, client))
        for client, partition in enumerate(partitions)
    ]
    data_seconds = perf_counter() - started
    fed_started = perf_counter()
    clients = [
        _run_stage(
            encoded, candidates, config.selection.client_selection_size,
            config, backend, _rng(seed, trial, 10, client),
            forced_ids=context if config.selection.forced_set_size else None,
            chain_path=artifact_directory / f"trial_{trial}" / f"client_{client}.npz" if artifact_directory else None,
            proposal_budget=config.search.local_proposal_budget,
        )
        for client, (context, candidates) in enumerate(local_sets)
    ]
    pool_started = perf_counter()
    server_pool = np.array(sorted({item for client in clients for item in client["selected_ids"]}), dtype=np.int64)
    # User-requested disclosure is isolated to this named non-private baseline.
    # Regular/private-local mode never constructs this union at the server.
    shared_context = np.sort(np.concatenate([context for context, _ in local_sets])) if pooled else np.empty(0, dtype=np.int64)
    if pooled and len(shared_context) != config.partition.num_clients * config.selection.forced_set_size:
        raise AssertionError("Disjoint contexts must produce the configured pooled context size")
    if np.intersect1d(shared_context, server_pool).size:
        raise AssertionError("Suggested new items must exclude the shared fixed context")
    if len(server_pool) != config.server_pool_size:
        raise AssertionError("Disjoint clients must contribute the configured number of unique items")
    initial_ids = None
    if config.search.initialization == "paired_uniform" and config.search.algorithm != "dppy_exact":
        initial_ids = np.sort(_rng(seed, trial, 2).choice(
            server_pool, config.selection.final_selection_size, replace=False,
        ))
    preprocessing_seconds = perf_counter() - fed_started
    pool_seconds = perf_counter() - pool_started
    # The centralized reference can access all eligible global items. The
    # server sees only suggestions, plus explicitly disclosed pooled G0 when
    # enabled; it is never given this full global eligibility collection.
    central_candidates = np.sort(np.concatenate([eligible for _, eligible in local_sets]))
    centralized = _run_stage(
        encoded, central_candidates,
        config.selection.final_selection_size, config, backend,
        _rng(seed, trial, 3), initial_ids,
        forced_ids=shared_context if pooled else None,
        chain_path=artifact_directory / f"trial_{trial}" / "centralized.npz" if artifact_directory else None,
        proposal_budget=config.search.global_proposal_budget,
    )
    # Identical RNG stream for paired final searches; candidates still differ.
    server_stream = 3 if initial_ids is not None else 4
    server = _run_stage(
        encoded, server_pool, config.selection.final_selection_size,
        config, backend, _rng(seed, trial, server_stream), initial_ids,
        forced_ids=shared_context if pooled else None,
        chain_path=artifact_directory / f"trial_{trial}" / "server.npz" if artifact_directory else None,
        proposal_budget=config.search.global_proposal_budget,
    )
    a, b = tuple(centralized["selected_ids"]), tuple(server["selected_ids"])
    # Evaluate BOTH outputs with the same public kernel function and jitter,
    # independent of the backend/container used during selection.
    central_joint = np.union1d(shared_context, a).astype(np.int64)
    fed_joint = np.union1d(shared_context, b).astype(np.int64)
    central_utility = stable_logdet(hamming_rbf(encoded[central_joint], config.kernel.gamma, config.kernel.jitter))
    fed_utility = stable_logdet(hamming_rbf(encoded[fed_joint], config.kernel.gamma, config.kernel.jitter))
    context_logdet = stable_logdet(hamming_rbf(encoded[shared_context], config.kernel.gamma, config.kernel.jitter))
    central_regular = stable_logdet(hamming_rbf(encoded[list(a)], config.kernel.gamma, config.kernel.jitter))
    fed_regular = stable_logdet(hamming_rbf(encoded[list(b)], config.kernel.gamma, config.kernel.jitter))
    for stage, utility in ((centralized, central_utility), (server, fed_utility)):
        if not np.isfinite(utility) or not np.isclose(stage["logdet"], utility, atol=config.numerics.psd_tolerance, rtol=0):
            raise AssertionError("Stage and shared public-kernel utilities must agree")
    if len(set(a)) != config.selection.final_selection_size or len(set(b)) != config.selection.final_selection_size:
        raise AssertionError("Final selections must contain the configured number of unique items")
    if not set(b) <= set(server_pool):
        raise AssertionError("The server must select only from client suggestions")
    if np.intersect1d(shared_context, a).size or np.intersect1d(shared_context, b).size:
        raise AssertionError("Final new-item selections must exclude the fixed pooled context")
    evaluation = []
    for client, (context, _) in enumerate(local_sets):
        # Trusted evaluation only: exact per-client gains are not protocol
        # messages and these diagnostics must not be shared with the server.
        base = stable_logdet(hamming_rbf(encoded[context], config.kernel.gamma, config.kernel.jitter))
        if not np.isfinite(base):
            raise ValueError("Forced-set kernel has non-finite logdet; conditional gain is undefined")
        scores = {}
        for name, selection in (("centralized", a), ("federated", b)):
            if np.intersect1d(context, selection).size:
                raise AssertionError("A public selection must not overlap any private context")
            joint = np.union1d(context, selection).astype(np.int64)
            score = stable_logdet(hamming_rbf(encoded[joint], config.kernel.gamma, config.kernel.jitter))
            if not np.isfinite(score):
                raise ValueError("Final augmented-set kernel has non-finite logdet")
            scores[f"{name}_joint_logdet"] = score
            scores[f"{name}_conditional_gain"] = score - base
        evaluation.append({"client": client, "forced_set_size": len(context), **scores})
    uploaded_items = sum(client["selection_size"] for client in clients)
    id_bytes = np.dtype(np.int64).itemsize
    metrics = {
        "trial": trial,
        "centralized_logdet": central_utility,
        "federated_logdet": fed_utility,
        "centralized_regular_logdet": central_regular,
        "federated_regular_logdet": fed_regular,
        "shared_context_logdet": context_logdet,
        "centralized_conditional_gain": central_utility - context_logdet,
        "federated_conditional_gain": fed_utility - context_logdet,
        "logdet_gap_fed_minus_central": fed_utility - central_utility,
        "jaccard": jaccard(a, b),
        "intersection_size": len(set(a) & set(b)),
        "centralized_selection_size": len(a), "federated_selection_size": len(b),
        "server_pool_size": len(server_pool),
        "forced_set_size_per_client": config.selection.forced_set_size,
        "total_forced_items": config.partition.num_clients * config.selection.forced_set_size,
        "shared_context_size": len(shared_context),
        "centralized_joint_selection_size": len(central_joint),
        "federated_joint_selection_size": len(fed_joint),
        "centralized_overlap_violations": 0, "federated_overlap_violations": 0,
        "centralized_proposals": centralized.get("proposed_swaps", centralized["iterations"]),
        "client_proposals": sum(client.get("proposed_swaps", client["iterations"]) for client in clients),
        "server_proposals": server.get("proposed_swaps", server["iterations"]),
        "federated_total_proposals": sum(client.get("proposed_swaps", client["iterations"]) for client in clients)
        + server.get("proposed_swaps", server["iterations"]),
        "centralized_total_seconds": centralized["total_seconds"],
        "federated_total_seconds": preprocessing_seconds + server["total_seconds"],
        "federated_ideal_parallel_seconds_estimate": max(client["total_seconds"] for client in clients) + pool_seconds + server["total_seconds"],
        "data_generation_seconds": data_seconds,
        "estimated_suggestion_upload_bytes": uploaded_items * id_bytes,
        "estimated_context_upload_bytes": len(shared_context) * id_bytes,
        "estimated_client_upload_bytes": (uploaded_items + len(shared_context)) * id_bytes,
        "estimated_final_broadcast_bytes": config.partition.num_clients * len(b) * id_bytes,
        "estimated_total_communication_bytes": (uploaded_items + len(shared_context) + config.partition.num_clients * len(b)) * id_bytes,
        "logical_communication_rounds": 2,
    }
    if config.search.algorithm == "stochastic_local_search":
        metrics.update(nearest_neighbor_hamming(encoded[list(a)], encoded[list(b)]))
    if config.search.algorithm == "dppy_mcmc":
        metrics.update({
            "centralized_chain_steps": centralized["chain_steps"],
            "server_chain_steps": server["chain_steps"],
            "client_chain_steps": sum(client["chain_steps"] for client in clients),
            "centralized_downhill_swaps": centralized["downhill_swaps"],
            "server_downhill_swaps": server["downhill_swaps"],
            "centralized_acceptance_rate": centralized["acceptance_rate_per_proposal"],
            "server_acceptance_rate": server["acceptance_rate_per_proposal"],
        })
    if config.search.algorithm == "dppy_exact":
        metrics.update({
            "centralized_samples": centralized["sample_count"],
            "client_samples": sum(client["sample_count"] for client in clients),
            "server_samples": server["sample_count"],
        })
    kmers = decode_kmers(encoded, config.data.alphabet)
    record = {
        "trial": trial, "metrics": metrics,
        "collection_sha256": hashlib.sha256("\n".join(kmers).encode()).hexdigest(),
        "global_collection": kmers,
        "client_partitions": [part.tolist() for part in partitions],
        "clients": clients, "server_pool_ids": server_pool.tolist(),
        "trusted_evaluation_client_scores": evaluation,
        "centralized": centralized, "server": server,
        "centralized_kmers": [kmers[index] for index in a],
        "federated_kmers": [kmers[index] for index in b],
    }
    if pooled:
        # Membership is intentionally server-visible in this NON-PRIVATE mode.
        record["server_shared_context_ids"] = shared_context.tolist()
        record["client_disclosed_context_ids"] = [context.tolist() for context, _ in local_sets]
    return record


def run_experiment(
    config: ExperimentConfig, backend: str,
    on_trial: Callable[[dict], None] | None = None,
    artifact_directory: Path | None = None,
) -> dict:
    """Run configured paired trials; return JSON-serializable results."""
    pooled = config.selection.final_objective == "pooled_context"
    trials = []
    for trial in range(config.experiment.repetitions):
        result = run_trial(config, backend, trial, artifact_directory)
        trials.append(result)
        if on_trial is not None:
            on_trial(result["metrics"])
    return {
        "backend": backend,
        **({"selection_comparison": hamming_comparison_definition()}
           if config.search.algorithm == "stochastic_local_search" else {}),
        "scope": (
            "NON-PRIVATE two-stage pooled-context semi-supervised baseline; all client contexts disclosed to server"
            if pooled else
            "two-stage suggestion-aggregation baseline with local semi-supervised search; "
            "ordinary centralized/server objectives; not a global semi-supervised protocol"
            if config.selection.forced_set_size else
            "non-private two-stage distributed ordinary-DPP selection baseline"
        ),
        "optimizer": (
            "native DPPy exact spectral k-DPP sampling; one draw per stage; independent central/server RNG"
            if config.search.algorithm == "dppy_exact" else
            "native DPPy exchange MCMC with isolated log-weight arithmetic; terminal sample, not MAP"
            if config.search.algorithm == "dppy_mcmc" else
            "shared stochastic local search; not MCMC or guaranteed MAP"
        ),
        "final_objective": config.selection.final_objective,
        "utility": (
            "logdet(L_[G0_pooled union G1]); SAME fixed pooled G0 and cardinality for both final searches"
            if pooled else
            "final regular diversity logdet(L_G1); local forced-set search uses logdet(L_[G0_i union G1_i])"
        ),
        "global_client_gain_aggregation": "not used; pooled-context logdet is NOT an aggregation of per-client conditional gains",
        "forced_set_construction": {
            "size_per_client": config.selection.forced_set_size,
            "sampling": "uniform without replacement from each disjoint client partition",
            "rng_stream": 5, "overlap_between_contexts": "none (disjoint partitions)",
            "membership_in_outputs": pooled,
            "disclosed_to_server": pooled,
            "pooled_size": config.partition.num_clients * config.selection.forced_set_size if pooled else 0,
        },
        "observations": {
            "server": (
                "public universe, kernel parameters, suggested G1_i IDs, EVERY disclosed G0_i membership, pooled G0, final G1"
                if pooled else
                "public universe, kernel parameters, suggested G1_i IDs, and final G1 only; no G0_i or eligibility masks"
            ),
            "trusted_evaluator": "private local contexts and exact scores; diagnostics are not server-visible",
            "privacy_guarantee": False,
        },
        "gap_definition": "federated_logdet - centralized_logdet; positive favors federated",
        "config": config.as_dict(), "trials": trials,
        "summary": summarize([trial["metrics"] for trial in trials]),
    }
