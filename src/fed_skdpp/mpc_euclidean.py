"""Numerical contract and trusted oracle for the three-party Euclidean PoC.

Source: expansion paper Section III, Algorithm 1 and Equation (4). This is
pooled-context SLS optimization, NOT k-DPP sampling. Coordinates are private
here, a stronger/different input model than public features/private membership.
Only the trusted synthetic evaluator calls the plaintext functions below.
"""

from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path

import numpy as np
import yaml

from .kernels import euclidean_rbf, stable_logdet, validate_kernel


# Both primes are 3 mod 4 (Blum primes), enabling MPyC's fast square-root
# path for the many random-bit protocols. A 1 mod 4 prime is much slower.
FIELD_PRIME = 2**255 - 765
WIDE_FIELD_PRIME = 2**521 - 1
STEP_FIELDS = ("active", "old_logdet", "proposed_logdet", "delta", "ratio",
               "ratio_valid", "accepted", "improved", "stalled", "valid")
SUMMARY_FIELDS = ("valid", "full_logdet", "full_determinant", "context_logdet",
                  "initial_logdet", "final_logdet", "iterations", "accepted_swaps",
                  "stalled", "patience_reached")


def primary_prime(config):
    required = config.bit_length+config.fractional_bits+config.statistical_security_bits+1
    return FIELD_PRIME if required < 255 else WIDE_FIELD_PRIME


def diagnostic_prime(index, config):
    return WIDE_FIELD_PRIME if index == SUMMARY_FIELDS.index("full_determinant") else primary_prime(config)


@dataclass(frozen=True)
class MPCConfig:
    total_points: int = 24
    dimension: int = 4
    private_fraction: float = 0.25
    num_clients: int = 3
    gamma: float = 1.0
    jitter: float = 0.0
    selection_size: int = 6
    patience: int = 50
    max_proposals: int = 200
    improvement_tolerance: float = 1e-10
    repetitions: int = 3
    public_seed: int = 42
    bit_length: int = 128
    fractional_bits: int = 64
    determinant_bit_length: int = 256
    determinant_fractional_bits: int = 128
    statistical_security_bits: int = 40
    exp_degree: int = 12
    log_terms: int = 12
    minimum_pivot: float = 1e-12
    maximum_log_ratio: float = 16.0
    logdet_absolute_tolerance: float = 1e-7
    ratio_relative_tolerance: float = 1e-7
    timeout_seconds: int = 900
    output_directory: str = "../results"

    def __post_init__(self):
        for name in ("total_points", "dimension", "num_clients", "selection_size",
                     "patience", "max_proposals", "repetitions", "bit_length",
                     "fractional_bits", "determinant_bit_length", "determinant_fractional_bits",
                     "statistical_security_bits", "exp_degree",
                     "log_terms", "timeout_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_clients != 3:
            raise ValueError("This PoC implements exactly three MPC parties, threshold one")
        if type(self.public_seed) is not int or self.public_seed < 0:
            raise ValueError("public_seed must be a nonnegative integer")
        for name in ("gamma", "minimum_pivot", "maximum_log_ratio",
                     "logdet_absolute_tolerance", "ratio_relative_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("jitter", "improvement_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.private_fraction < 1:
            raise ValueError("private_fraction must be between zero and one")
        count = self.total_points * self.private_fraction
        if not float(count).is_integer() or int(count) % self.num_clients:
            raise ValueError("private count must be integral and divisible by num_clients")
        if not 0 < self.private_count < self.total_points:
            raise ValueError("need both private and public points")
        if self.selection_size >= self.public_count:
            raise ValueError("selection_size must leave an unused public swap candidate")
        if self.max_proposals < self.patience:
            raise ValueError("max_proposals must be at least patience")
        if self.fractional_bits < 48 or self.bit_length < 2 * self.fractional_bits:
            raise ValueError("use at least 48 fractional bits and bit_length >= 2*f")
        if self.bit_length + self.fractional_bits + self.statistical_security_bits + 1 >= 521:
            raise ValueError("fixed 521-bit prime is too small for requested precision")
        if (self.determinant_fractional_bits < self.fractional_bits
                or self.determinant_bit_length < 2*self.determinant_fractional_bits
                or self.determinant_bit_length+self.determinant_fractional_bits+self.statistical_security_bits+1 >= 521):
            raise ValueError("invalid wide precision for the full determinant product")
        if self.minimum_pivot <= 2.0 ** (-self.fractional_bits + 8):
            raise ValueError("minimum_pivot is below the configured numerical guard")
        if self.exp_degree < 8 or self.log_terms < 12:
            raise ValueError("polynomial approximation orders are too small for this PoC")
        if self.maximum_log_ratio > (self.bit_length-self.fractional_bits-4)*math.log(2):
            raise ValueError("maximum_log_ratio exceeds fixed-point output range")

    @property
    def private_count(self):
        return int(self.total_points * self.private_fraction)

    @property
    def public_count(self):
        return self.total_points - self.private_count

    @property
    def diagnostics_count(self):
        return len(SUMMARY_FIELDS) + self.max_proposals * len(STEP_FIELDS)


def load_mpc_config(path: Path) -> MPCConfig:
    with path.open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("MPC config must be a mapping")
    unknown = set(values) - {field.name for field in fields(MPCConfig)}
    if unknown:
        raise ValueError(f"Unknown MPC settings: {sorted(unknown)}")
    return MPCConfig(**values)


def exp_reduction(bound: float) -> int:
    """Public range reduction ensuring |x / 2**s| <= 1/8."""
    return max(0, math.ceil(math.log2(max(bound, 0.125) / 0.125)))


def polynomial_exp(value, degree: int, reductions: int):
    """Generic array/scalar exp approximation; also operates on secure types."""
    scaled = value / (2**reductions)
    result = scaled * 0 + 1 / math.factorial(degree)
    for order in range(degree - 1, -1, -1):
        result = result * scaled + 1 / math.factorial(order)
    for _ in range(reductions):
        result = result * result
    return result


def generate_fixture(config: MPCConfig, private_rng,
                     public_rng: np.random.Generator):
    """Trusted synthetic fixture: uniformly permute points into private/public.

    Use a cryptographic RNG in a privacy experiment; seeded NumPy generators
    are accepted only for deterministic unit fixtures. Private RNG state must
    never be published or supplied to MPC workers.
    Each party receives only its own points and random coin shares. Coin sums
    modulo their public ranges give uniform joint proposals hidden from any
    one party. Public initialization is independent of the private contexts.
    """
    points = private_rng.uniform(0, 1, (config.total_points, config.dimension))
    points = points[private_rng.permutation(config.total_points)]
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Generated duplicate coordinates; fixture is invalid")
    count = config.private_count // config.num_clients
    owners = []
    ranges = np.array([config.public_count-config.selection_size, config.selection_size])
    for party in range(config.num_clients):
        owners.append({
            "points": points[party*count:(party+1)*count].tolist(),
            "coins": private_rng.integers(0, ranges, (config.max_proposals, 2)).tolist(),
        })
    public = points[config.private_count:]
    initial = public_rng.choice(config.public_count, config.selection_size, replace=False)
    return points, owners, public, initial


def combined_coins(owners, config):
    ranges = np.array([config.public_count-config.selection_size, config.selection_size])
    return np.sum([np.array(owner["coins"], dtype=np.int64) for owner in owners], axis=0) % ranges


def plaintext_reference(points, initial, coins, config: MPCConfig):
    """Independent float64 full-joint-kernel SLS; ordered slots match MPC.

    Input points are (N,d), private rows first; initial and returned IDs are
    public-pool-local. Incoming ranks index the sorted eligible complement.
    Every attempted proposal counts, ties accepted, patience reset only for
    improvements exceeding tolerance, as in the existing SLS implementation.
    """
    kernel = euclidean_rbf(points, config.gamma, config.jitter)
    validate_kernel(kernel)
    private = list(range(config.private_count))
    def score(selected):
        ids = private + [config.private_count + int(index) for index in selected]
        value = stable_logdet(kernel[np.ix_(ids, ids)])
        if not math.isfinite(value):
            raise ValueError("Plaintext joint kernel is singular or numerically invalid")
        return value
    selected = np.array(initial, dtype=np.int64, copy=True)
    current = initial_score = score(selected)
    stalled = accepted_count = 0
    steps = []
    for incoming_rank, outgoing in coins:
        complement = np.setdiff1d(np.arange(config.public_count), selected)
        proposal = selected.copy()
        proposal[int(outgoing)] = complement[int(incoming_rank)]
        proposed = score(proposal)
        delta = proposed-current
        accepted = proposed >= current
        improved = accepted and delta > config.improvement_tolerance
        old = current
        if accepted:
            selected, current = proposal, proposed
            accepted_count += 1
        stalled = 0 if improved else stalled+1
        steps.append(dict(old_logdet=old, proposed_logdet=proposed, delta=delta,
                          ratio=math.exp(delta), accepted=accepted, improved=improved,
                          stalled=stalled))
        if stalled >= config.patience:
            break
    return {
        "full_logdet": stable_logdet(kernel),
        "context_logdet": stable_logdet(kernel[np.ix_(private, private)]),
        "initial_logdet": initial_score, "final_logdet": current,
        "selected": sorted(int(i) for i in selected),
        "iterations": len(steps), "accepted_swaps": accepted_count, "stalled": stalled,
        "patience_reached": stalled >= config.patience,
        "steps": steps, "kernel_condition_number": float(np.linalg.cond(kernel)),
    }


def decode_diagnostics(masked, owners, config):
    """Trusted evaluator only: remove independent one-time pads modulo p.

    Masks cover raw fixed-point field encodings, not bounded float values.
    Each diagnostic has its own pad from every party. Single-party workers
    cannot decode these diagnostics; keeping them out of the public report is
    required in a real deployment. This PoC intentionally publishes aggregate
    validation results obtained by the trusted synthetic evaluator.
    """
    if len(masked) != config.diagnostics_count:
        raise ValueError("Wrong number of masked diagnostic outputs")
    values = []
    for index, encoded in enumerate(masked):
        prime = diagnostic_prime(index, config)
        value = (int(encoded)-sum(int(o["pads"][index]) for o in owners)) % prime
        if value > prime//2:
            value -= prime
        fractional = config.determinant_fractional_bits if index == SUMMARY_FIELDS.index("full_determinant") else config.fractional_bits
        values.append(value / 2**fractional)
    head = dict(zip(SUMMARY_FIELDS, values[:len(SUMMARY_FIELDS)]))
    tail = values[len(SUMMARY_FIELDS):]
    head["steps"] = [dict(zip(STEP_FIELDS, tail[i:i+len(STEP_FIELDS)]))
                     for i in range(0, len(tail), len(STEP_FIELDS))]
    return head
