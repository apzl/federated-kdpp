"""Public-selection comparisons, separate from the logdet search objective."""

import numpy as np


HAMMING_METRIC = "symmetric_mean_nearest_hamming"
NORMALIZED_HAMMING_METRIC = HAMMING_METRIC + "_normalized"
HAMMING_KEYS = (
    "centralized_to_federated_mean_nearest_hamming",
    "federated_to_centralized_mean_nearest_hamming",
    HAMMING_METRIC, NORMALIZED_HAMMING_METRIC,
)


def nearest_neighbor_hamming(centralized: np.ndarray, federated: np.ndarray) -> dict:
    """Compare nonempty (n, length)/(m, length) arrays of aligned symbols.

    Raw Hamming distance counts mismatching positions, without squaring it.
    For each item, find its nearest item in the OTHER set, allowing many-to-one
    matches and exact matches of distance zero. Average each direction, then
    give those two directed means equal weight, even if cardinalities differ.
    Fixed context items must not be included by callers. This is an evaluation
    dissimilarity (lower is better agreement), not a diversity objective or a
    one-to-one assignment. It need not satisfy the triangle inequality.

    Arrays may contain finite numeric symbol encodings or strings. Invalid
    shapes, empty inputs, different lengths, and nonfinite numeric values fail.
    Uses an (n,m) integer work array, not an (n,m,length) broadcast tensor.
    """
    left, right = np.asarray(centralized), np.asarray(federated)
    for values in (left, right):
        if values.ndim != 2 or not values.shape[0] or not values.shape[1]:
            raise ValueError("Hamming inputs must be nonempty (items, positive length) arrays")
        if values.dtype.kind not in "biufUS":
            raise ValueError("Hamming symbols must be finite numeric values or strings")
        if values.dtype.kind in "biuf" and not np.isfinite(values).all():
            raise ValueError("Hamming symbols must be finite")
    if left.shape[1] != right.shape[1]:
        raise ValueError("Hamming inputs must have the same sequence length")
    distances = np.zeros((len(left), len(right)), dtype=np.int64)
    for position in range(left.shape[1]):
        distances += left[:, position, None] != right[None, :, position]
    forward = float(distances.min(axis=1).mean())
    backward = float(distances.min(axis=0).mean())
    symmetric = (forward + backward) / 2
    return dict(zip(HAMMING_KEYS, (forward, backward, symmetric, symmetric / left.shape[1])))


def hamming_comparison_definition() -> dict:
    """Machine-readable semantics for SLS reports; no optimizer changes."""
    return {
        "primary_metric": HAMMING_METRIC, "direction": "minimize",
        "definition": "0.5 * (mean_a min_b d_H(a,b) + mean_b min_a d_H(a,b))",
        "sets": "centralized and federated FINAL G1 only; all fixed G0 excluded",
        "units": "mean number of mismatching positions; raw range [0, kmer_length]",
        "pairing": "nearest neighbor across sets; many-to-one permitted; not bipartite matching",
        "exact_matches": "included with distance zero",
        "normalized_metric": NORMALIZED_HAMMING_METRIC,
        "normalization": "raw symmetric distance divided by kmer_length; range [0,1]",
        "interpretation": "lower is closer to the centralized selection, not necessarily higher diversity or utility",
        "optimization": "unchanged joint/regular logdet objective and SLS acceptance rule",
        "legacy_diagnostics": "intersection_size and jaccard are retained, but not the primary SLS comparison",
    }


def scalar_summary(values) -> dict:
    """Mean/sample SD and observed range over all configured paired trials."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Summary requires a nonempty finite vector")
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max()), "num_trials": len(values)}


def hamming_sweep_summaries(experiment: dict) -> dict:
    """Additional point fields for SLS; keep sampler/legacy schemas unchanged."""
    if experiment["config"]["search"]["algorithm"] != "stochastic_local_search":
        return {}
    rows = [trial["metrics"] for trial in experiment["trials"]]
    return {"hamming_distance": scalar_summary([row[HAMMING_METRIC] for row in rows]),
            "hamming_distance_normalized": scalar_summary([row[NORMALIZED_HAMMING_METRIC] for row in rows])}


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    """Set Jaccard on public item IDs; two empty sets have similarity 1."""
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 1.0


def summarize(rows: list[dict]) -> dict:
    """Mean and sample standard deviation of metrics over paired trials."""
    keys = (
        "centralized_logdet", "federated_logdet", "logdet_gap_fed_minus_central",
        "jaccard", "intersection_size", "centralized_total_seconds", "federated_total_seconds",
        "centralized_proposals", "federated_total_proposals",
        "centralized_conditional_gain", "federated_conditional_gain",
    )
    optional = tuple(key for key in HAMMING_KEYS if rows and all(key in row for row in rows))
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0,
        }
        for key in keys + optional
    }
