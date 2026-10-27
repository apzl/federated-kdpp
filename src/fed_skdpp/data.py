"""Uniform unique k-mers and balanced random, disjoint client partitions."""

import numpy as np

from .config import DataConfig


def sample_kmers(config: DataConfig, rng: np.random.Generator) -> np.ndarray:
    """Return an int64 (N, kmer_length) array of alphabet indices.

    Sample distinct strings uniformly without replacement; reverse complements
    are not collapsed. Avoid enumerating the entire alphabet**length universe.
    A rejection path supports universes too large for int64 integer IDs.
    """
    n, length = config.global_collection_size, config.kmer_length
    base = len(config.alphabet)
    universe_size = base ** length
    if universe_size <= np.iinfo(np.int64).max:
        codes = rng.choice(universe_size, size=n, replace=False)
        encoded = np.empty((n, length), dtype=np.int64)
        for position in range(length - 1, -1, -1):
            encoded[:, position] = codes % base
            codes = codes // base
        return encoded
    rows: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    while len(rows) < n:
        row = tuple(int(value) for value in rng.integers(base, size=length))
        if row not in seen:
            rows.append(row)
            seen.add(row)
    return np.asarray(rows, dtype=np.int64)


def decode_kmers(encoded: np.ndarray, alphabet: str) -> list[str]:
    """Map (N, length) integer alphabet codes to strings in the same order."""
    return ["".join(alphabet[int(value)] for value in row) for row in encoded]


def partition_indices(
    population_size: int, num_clients: int, rng: np.random.Generator,
) -> list[np.ndarray]:
    """Return sorted global IDs for a random balanced partition of range(N)."""
    if population_size < 1 or num_clients < 1 or population_size % num_clients:
        raise ValueError("A balanced partition requires positive, divisible sizes")
    shuffled = rng.permutation(population_size)
    return [np.sort(part) for part in np.split(shuffled, num_clients)]


def split_context(
    collection_ids: np.ndarray, forced_set_size: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Reserve random immutable G0 IDs; return (G0, eligible new-item IDs).

    Input/output index space is public global item IDs, dtype int64, shape
    (collection_size,), (forced_set_size,), and (collection_size-forced_set_size,).
    G0 is sampled uniformly without replacement and both returned arrays are
    sorted copies. Their disjoint union equals the original client collection.
    This function executes locally; G0 must not be sent to the server.
    """
    collection_ids = np.asarray(collection_ids)
    if collection_ids.ndim != 1 or not np.issubdtype(collection_ids.dtype, np.integer):
        raise ValueError("collection_ids must be a one-dimensional integer array")
    if len(np.unique(collection_ids)) != len(collection_ids) or (collection_ids < 0).any():
        raise ValueError("collection_ids must contain unique nonnegative IDs")
    if type(forced_set_size) is not int or not 0 <= forced_set_size <= len(collection_ids):
        raise ValueError("forced_set_size must fit the client collection")
    # Preserve the original proposal/data streams when G0 is empty.
    context = np.sort(rng.choice(collection_ids, size=forced_set_size, replace=False)).astype(np.int64)
    eligible = np.setdiff1d(collection_ids, context).astype(np.int64)
    return context, eligible
