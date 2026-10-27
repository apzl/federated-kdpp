"""Independent numerical/privacy-contract fixtures for the Euclidean MPC PoC."""

from dataclasses import replace
import math

import numpy as np
import pytest

from fed_skdpp.kernels import euclidean_rbf, stable_logdet, validate_kernel
from fed_skdpp.mpc_euclidean import (FIELD_PRIME, MPCConfig, SUMMARY_FIELDS, STEP_FIELDS,
                                    combined_coins, decode_diagnostics, exp_reduction,
                                    generate_fixture, polynomial_exp, plaintext_reference, diagnostic_prime)


def small_config(**overrides):
    return replace(MPCConfig(total_points=12, selection_size=2, patience=2,
                             max_proposals=4, repetitions=1), **overrides)


def test_euclidean_rbf_has_squared_distance_and_known_two_point_determinant():
    points = np.array([[0, 0, 0, 0], [2, 0, 0, 0]], dtype=float)
    kernel = euclidean_rbf(points, gamma=.25)
    np.testing.assert_allclose(kernel, [[1, math.exp(-1)], [math.exp(-1), 1]])
    assert stable_logdet(kernel) == pytest.approx(math.log1p(-math.exp(-2)))
    validate_kernel(kernel)
    # An RBF kernel does not have the linear 4-dimensional feature rank limit.
    points = np.random.default_rng(2).uniform(size=(12, 4))
    assert math.isfinite(stable_logdet(euclidean_rbf(points, gamma=1)))


@pytest.mark.parametrize("points", [np.array([0, 1]), np.empty((3, 0)), [[float("nan")]]])
def test_euclidean_kernel_rejects_invalid_coordinates(points):
    with pytest.raises(ValueError):
        euclidean_rbf(points, 1)


@pytest.mark.parametrize("changes", [
    {"num_clients": 2}, {"total_points": 25}, {"private_fraction": .2},
    {"selection_size": 18}, {"gamma": 0}, {"jitter": -1}, {"patience": 0},
    {"max_proposals": 49}, {"fractional_bits": 24}, {"bit_length": 96},
    {"fractional_bits": 200, "bit_length": 400}, {"public_seed": -1},
])
def test_mpc_config_rejects_ambiguous_or_unsafe_shapes_and_numerics(changes):
    with pytest.raises(ValueError):
        replace(MPCConfig(), **changes)


def test_public_exp_approximation_on_the_entire_declared_intervals():
    for low, high in ((-4, 0), (-16, 16)):
        x = np.linspace(low, high, 1000)
        values = polynomial_exp(x, MPCConfig().exp_degree, exp_reduction(max(abs(low), abs(high))))
        np.testing.assert_allclose(values, np.exp(x), rtol=5e-13, atol=1e-15)


def test_private_partition_and_random_coins_are_independent_of_public_seed():
    config = small_config()
    points, owners, public, initial = generate_fixture(config, np.random.default_rng(7), np.random.default_rng(42))
    all_private = np.concatenate([owner["points"] for owner in owners])
    assert all_private.shape == (3, 4) and public.shape == (9, 4)
    np.testing.assert_array_equal(np.vstack([all_private, public]), points)
    other = generate_fixture(config, np.random.default_rng(8), np.random.default_rng(42))
    np.testing.assert_array_equal(initial, other[3])
    assert not np.array_equal(points, other[0])
    tape = combined_coins(owners, config)
    assert tape.shape == (4, 2)
    assert (tape[:, 0] < 7).all() and (tape[:, 1] < 2).all()


def test_cross_client_entries_cannot_be_replaced_by_local_determinants():
    points = np.array([[0, 0, 0, 0], [.2, 0, 0, 0], [.4, 0, 0, 0]])
    joint = stable_logdet(euclidean_rbf(points, 1))
    assert joint < -1
    # Every local one-point logdet is zero; their sum loses all interactions.
    assert sum(stable_logdet(euclidean_rbf(points[i:i+1], 1)) for i in range(3)) == 0


def test_plaintext_sls_preserves_context_cardinality_and_monotonicity():
    config = small_config(patience=3, max_proposals=20)
    points, owners, public, initial = generate_fixture(config, np.random.default_rng(11), np.random.default_rng(42))
    original = points.copy()
    ref = plaintext_reference(points, initial, combined_coins(owners, config), config)
    np.testing.assert_array_equal(points, original)
    assert len(ref["selected"]) == len(set(ref["selected"])) == 2
    assert all(0 <= item < len(public) for item in ref["selected"])
    assert ref["final_logdet"] >= ref["initial_logdet"]
    assert ref["patience_reached"] and ref["stalled"] == 3
    for step in ref["steps"]:
        assert step["ratio"] == pytest.approx(math.exp(step["proposed_logdet"]-step["old_logdet"]))
        assert step["accepted"] == (step["delta"] >= 0)
        if step["accepted"]:
            assert step["ratio"] >= 1


def test_singular_context_is_not_silently_regularized():
    config = small_config()
    points = np.zeros((12, 4))
    with pytest.raises(ValueError, match="singular|invalid"):
        plaintext_reference(points, [0, 1], np.zeros((4, 2), dtype=int), config)


def test_field_diagnostic_pads_cover_signed_fixed_point_values():
    config = small_config()
    values = [-4.25, 1.5] + [0.]*(config.diagnostics_count-2)
    values[SUMMARY_FIELDS.index("full_determinant")] = 2.0**-100
    pads = [[(i+1)*(j+123456789012345678901234567890) % diagnostic_prime(j, config)
             for j in range(config.diagnostics_count)] for i in range(3)]
    owners = [{"pads": [str(v) for v in row]} for row in pads]
    masked = [str((round(value*2**(config.determinant_fractional_bits if i == SUMMARY_FIELDS.index("full_determinant") else config.fractional_bits))
                   +sum(row[i] for row in pads)) % diagnostic_prime(i, config)) for i, value in enumerate(values)]
    decoded = decode_diagnostics(masked, owners, config)
    assert decoded[SUMMARY_FIELDS[0]] == -4.25
    assert decoded[SUMMARY_FIELDS[1]] == 1.5
    assert decoded["full_determinant"] == 2.0**-100
    assert len(decoded["steps"]) == config.max_proposals
    assert set(decoded["steps"][0]) == set(STEP_FIELDS)
    with pytest.raises(ValueError, match="Wrong number"):
        decode_diagnostics(masked[:-1], owners, config)
