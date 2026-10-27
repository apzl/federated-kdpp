"""MPyC arithmetic for the Euclidean SLS PoC (optional dependency).

Three semi-honest parties, Shamir threshold one, authenticated/private channels
assumed. The localhost demo uses loopback transport, not production TLS.
All points, cross-client kernel entries, conditional scores, proposal coins,
decisions, and patience counters stay secret-shared. Only final public IDs
are opened. Diagnostic fields are independently one-time-padded modulo their
prime field before opening, for a separate trusted synthetic evaluator.

Algorithmic source: expansion paper Section III Algorithm 1. Secure LDL
elimination and range-reduced polynomial exp/log are proposed numerical
building blocks, not a claimed exact-arithmetic or exact k-DPP protocol.
"""

import math
from time import perf_counter

import numpy as np
from mpyc.runtime import mpc

from .mpc_euclidean import (WIDE_FIELD_PRIME, MPCConfig, exp_reduction,
                            polynomial_exp, primary_prime)


def secure_rbf(points, config: MPCConfig, public_points=None):
    """Secret (N,d) fixed-point array -> secret symmetric (N,N) kernel.

    Coordinates must lie in [0,1]; public shape and gamma bound the polynomial
    interval. No entries or distances are opened. Unit diagonal is set exactly.
    """
    if public_points is None:
        differences = points[:, None, :] - points[None, :, :]
        distances = mpc.np_sum(differences * differences, axis=2)
        kernel = polynomial_exp(-config.gamma * distances, config.exp_degree,
                            exp_reduction(config.gamma * config.dimension))
        upper = kernel * np.triu(np.ones((len(points), len(points)), dtype=np.int64), 1)
        return upper + upper.T + (1+config.jitter) * np.eye(len(points))
    # Evaluate only upper-triangle entries involving a private point. Public
    # entries use the exact float64 RBF, avoiding needless protected exp calls.
    from .kernels import euclidean_rbf
    n = points.shape[0]
    i, j = np.triu_indices(n, 1)
    keep = i < config.private_count
    i, j = i[keep], j[keep]
    differences = points[i] - points[j]
    distances = mpc.np_sum(differences*differences, axis=1)
    values = polynomial_exp(-config.gamma*distances, config.exp_degree,
                            exp_reduction(config.gamma*config.dimension))
    base = np.eye(n)*(1+config.jitter)
    base[config.private_count:, config.private_count:] = euclidean_rbf(public_points, config.gamma, config.jitter)
    kernel = type(points)(base)
    kernel = mpc.np_update(kernel, (i, j), values)
    return mpc.np_update(kernel, (j, i), values)


def secure_eliminate(matrix, count: int, config: MPCConfig):
    """LDL elimination: return safe pivots, Schur remainder, secret validity.

    Shapes and loop bounds are public. No inverse is formed; each pivot's
    reciprocal is reused in a rank-one update. Invalid pivots use dummy 1 to
    keep the circuit defined, with validity false. This is NOT regularization:
    the result must be discarded if the final validity flag is false.
    """
    secfxp = type(matrix).sectype
    remaining, pivots, valid = matrix, [], secfxp(1)
    for _ in range(count):
        pivot = remaining[0, 0]
        good = (pivot >= config.minimum_pivot) * (pivot <= 2*(1+config.jitter))
        valid *= good
        safe = mpc.if_else(good, pivot, secfxp(1))
        pivots.append(safe)
        if remaining.shape[0] > 1:
            column = remaining[1:, 0]
            weighted = column * (1 / safe)
            remaining = remaining[1:, 1:] - weighted[:, None] * column[None, :]
        else:
            remaining = None
    return pivots, remaining, valid


def secure_log_positive(values, config: MPCConfig):
    """Elementwise ln of protected positive LDL pivots, using public bounds.

    Secret comparisons normalize into [1/2,1]; the exponent stays protected.
    Evaluate ln(v)=2*atanh((v-1)/(v+1)) by a configured odd-power series.
    Invalid pivots must already have been replaced by dummy 1 with a false
    validity flag. Approximation and fixed-point error are validated separately.
    """
    # Adapt the leading-bit normalization used by MPyC's reciprocal. Since
    # 0 < pivot <= 2*(1+jitter), inspect only the necessary low encoding bits.
    length = config.fractional_bits + math.ceil(math.log2(2*(1+config.jitter))) + 2
    bits = mpc.np_flip(mpc.np_to_bits(values, l=length), axis=-1)
    leading, multiplier = mpc.np_find(bits, 1, cs_f=lambda b, i: (i+1+b, (b+1)*2**i))
    multiplier *= 2.0**(config.fractional_bits-length)
    value = values*multiplier
    exponent = length+1-config.fractional_bits-leading
    # 1.5 <= value+1 < 2: Newton reciprocal avoids another bit decomposition.
    denominator = value+1
    reciprocal = value*0 + .5
    for _ in range(math.ceil(math.log2(config.fractional_bits))+1):
        reciprocal *= 2-denominator*reciprocal
    t = (value-1)*reciprocal
    squared = t*t
    result = t*0 + 1/(2*config.log_terms-1)
    for order in range(config.log_terms-2, -1, -1):
        result = result*squared + 1/(2*order+1)
    return 2*t*result + exponent*math.log(2)


def secure_logdet(matrix, config):
    pivots, _, valid = secure_eliminate(matrix, matrix.shape[0], config)
    logs = secure_log_positive(mpc.np_fromlist(pivots), config)
    return mpc.np_sum(logs), valid, pivots


@mpc.coroutine
async def _raw_encoding(value, field_type):
    """Reinterpret this party's local fixed-point share without opening it."""
    await mpc.returnType(field_type)
    return await mpc.gather(value)


@mpc.coroutine
async def _cast_exact(value, target, scale):
    """Local linear share rescaling, ONLY for exact integer/Boolean values.

    Source and target use the same prime field. A fixed-point Boolean encodes
    bit b as b*2**f; multiplying local shares by public 2**(-f) converts it
    exactly to an integer bit. Conversely multiply by 2**f to encode an integer
    as fixed point. No rounding, reconstruction, or communication is involved.
    NEVER use this to round a general fractional value to an integer.
    """
    is_array = isinstance(value, mpc.SecureArray)
    if is_array:
        declaration = (target.array, True, value.shape) if target.frac_length else (target.array, value.shape)
    else:
        declaration = (target, True) if target.frac_length else target
    await mpc.returnType(declaration)
    share = await mpc.gather(value)
    return share * target.field(scale)


def _mod_small(value, modulus):
    # Sum of three shares in [0,modulus) is below 3*modulus.
    for _ in range(2):
        value = mpc.if_else(value >= modulus, value-modulus, value)
    return value


async def secure_sls(config: MPCConfig, public: dict, local: dict, progress=None):
    """Execute a fixed-size, fixed-budget circuit with private patience stop.

    `local` contains ONLY this party's points, coin shares, and audit pads.
    Initial/final IDs index the public pool; private rows never enter that
    index space. Returns public IDs, masked audit fields, and public timings.
    Never call this in one-party mode and claim a privacy result.
    """
    if len(mpc.parties) != 3 or mpc.threshold != 1:
        raise ValueError("PoC requires three processes with Shamir threshold one")
    prime = primary_prime(config)
    fxp = mpc.SecFxp(config.bit_length, config.fractional_bits, p=prime)
    integer_bits = max(config.max_proposals+1, 3*config.public_count, config.patience+1).bit_length()+1
    integer = mpc.SecInt(integer_bits, p=prime)
    to_fxp = lambda value: _cast_exact(value, fxp, 2**config.fractional_bits)
    to_int = lambda bit: _cast_exact(bit, integer, pow(2**config.fractional_bits, -1, prime))
    field_type = mpc.SecFld(prime)
    wide_field = mpc.SecFld(WIDE_FIELD_PRIME)
    count = config.private_count // 3
    private_arrays, coin_arrays, pad_arrays = [], [], []
    for owner in range(3):
        own = owner == mpc.pid
        coordinates = np.array(local["points"], dtype=float) if own else np.zeros((count, config.dimension))
        coins = np.array(local["coins"], dtype=np.int64) if own else np.zeros((config.max_proposals, 2), dtype=np.int64)
        pads = [field_type(int(v)) for i, v in enumerate(local["pads"]) if i != 2] if own else [field_type(0)] * (config.diagnostics_count-1)
        private_arrays.append(mpc.input(fxp.array(coordinates), senders=owner))
        coin_arrays.append(mpc.input(integer.array(coins), senders=owner))
        shared_pads = mpc.input(pads, senders=owner)
        shared_pads.insert(2, mpc.input(wide_field(int(local["pads"][2])) if own else wide_field(0), senders=owner))
        pad_arrays.append(shared_pads)
    points = np.vstack(private_arrays + [fxp.array(np.array(public["points"], dtype=float))])
    coins = coin_arrays[0] + coin_arrays[1] + coin_arrays[2]
    ranks = _mod_small(coins[:, 0], config.public_count-config.selection_size)
    outgoing_slots = _mod_small(coins[:, 1], config.selection_size)
    setup_start = perf_counter()
    kernel = secure_rbf(points, config, np.array(public["points"], dtype=float))
    private_pivots, conditional, context_valid = secure_eliminate(kernel, config.private_count, config)
    context_logdet = mpc.np_sum(secure_log_positive(mpc.np_fromlist(private_pivots), config))
    rest_logdet, full_valid, public_pivots = secure_logdet(conditional, config)
    full_logdet = context_logdet + rest_logdet
    wide = mpc.SecFxp(config.determinant_bit_length, config.determinant_fractional_bits, p=WIDE_FIELD_PRIME)
    full_determinant = mpc.prod(mpc.convert(private_pivots + public_pivots, wide))
    selectors = integer.array(np.eye(config.public_count, dtype=np.int64)[public["initial_selection"]])
    def score(select):
        fixed = to_fxp(select)
        selected_kernel = (fixed @ conditional) @ fixed.T
        return secure_logdet(selected_kernel, config)[:2]
    current, initial_valid = score(selectors)
    initial_logdet = context_logdet + current
    valid = context_valid * full_valid * initial_valid
    active = to_int(valid)
    stalled, iterations, accepted_swaps = integer(0), integer(0), integer(0)
    await mpc.barrier()
    setup_seconds = perf_counter()-setup_start
    if progress is not None:
        progress(0)
    search_start = perf_counter()
    diagnostics = []
    for step in range(config.max_proposals):
        available = 1-mpc.np_sum(selectors, axis=0)
        incoming = available * (mpc.np_cumsum(available) == ranks[step]+1)
        outgoing = integer.array(np.arange(config.selection_size, dtype=np.int64)) == outgoing_slots[step]
        old_row = outgoing @ selectors
        proposal = selectors + outgoing[:, None] * (incoming-old_row)[None, :]
        proposed, proposal_valid = score(proposal)
        delta = proposed-current
        proposal_ok = to_int(proposal_valid)
        accepted = active * proposal_ok * to_int(delta >= 0)
        improved = accepted * to_int(delta > config.improvement_tolerance)
        ratio_valid = (delta >= -config.maximum_log_ratio) * (delta <= config.maximum_log_ratio)
        safe_delta = mpc.if_else(ratio_valid, delta, fxp(0))
        ratio = polynomial_exp(safe_delta, config.exp_degree, exp_reduction(config.maximum_log_ratio))
        next_stalled = mpc.if_else(improved, integer(0), stalled+1)
        stalled = mpc.if_else(active, next_stalled, stalled)
        diagnostics.extend([to_fxp(active), context_logdet+current, context_logdet+proposed,
                            delta, ratio, ratio_valid, to_fxp(accepted), to_fxp(improved), to_fxp(stalled), proposal_valid])
        selectors += accepted * (proposal-selectors)
        current = mpc.if_else(to_fxp(accepted), proposed, current)
        iterations += active
        accepted_swaps += accepted
        valid *= to_fxp(1-active*(1-proposal_ok))
        active *= (stalled < config.patience) * proposal_ok
        await mpc.barrier()
        if progress is not None and (step+1) % 25 == 0:
            progress(step+1)
    search_seconds = perf_counter()-search_start
    final_logdet = context_logdet+current
    summary = [valid, full_logdet, full_determinant, context_logdet, initial_logdet,
               final_logdet, to_fxp(iterations), to_fxp(accepted_swaps), to_fxp(stalled), to_fxp(stalled >= config.patience)]
    values = summary+diagnostics
    masked = [_raw_encoding(value, field_type) + pad_arrays[0][i] + pad_arrays[1][i] + pad_arrays[2][i]
              for i, value in enumerate(values) if i != 2]
    encoded = await mpc.output(masked, raw=True)
    masked_determinant = _raw_encoding(values[2], wide_field) + pad_arrays[0][2]+pad_arrays[1][2]+pad_arrays[2][2]
    encoded.insert(2, await mpc.output(masked_determinant, raw=True))
    # On numerical failure publish no selection; the failure flag is declared.
    public_valid = bool(round(await mpc.output(valid)))
    selection = await mpc.output(mpc.np_sum(selectors, axis=0) * to_int(valid))
    selected = np.flatnonzero(selection > 0.5).tolist() if public_valid else []
    return {
        "selected": selected, "numerically_valid": public_valid,
        "masked_diagnostics": [str(value.value) for value in encoded],
        "setup_seconds": setup_seconds, "search_seconds": search_seconds,
        "fixed_executed_proposals": config.max_proposals,
        "explicit_barriers": config.max_proposals+1,
    }
