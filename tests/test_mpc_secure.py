"""Public mathematical fixtures for optional secure arithmetic.

Single-party execution here is only a numerical unit test. Privacy requires
the separate real three-party integration runs performed by the PoC launcher.
"""

import numpy as np
import pytest

pytest.importorskip("mpyc")
from mpyc.runtime import mpc

from fed_skdpp.kernels import euclidean_rbf, stable_logdet
from fed_skdpp.mpc_euclidean import FIELD_PRIME, MPCConfig
from fed_skdpp.mpc_secure import (_cast_exact, secure_eliminate, secure_log_positive,
                                 secure_logdet, secure_rbf)


def test_secure_rbf_schur_and_logdet_match_public_reference():
    async def check():
        await mpc.start()
        config = MPCConfig(total_points=12, selection_size=2, patience=2, max_proposals=4)
        fxp = mpc.SecFxp(config.bit_length, config.fractional_bits, p=FIELD_PRIME)
        points = np.array([[0, 0, 0, 0], [1, 0, 0, 0],
                           [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
        kernel = secure_rbf(fxp.array(points), config, points[3:])
        opened = await mpc.output(kernel)
        reference = euclidean_rbf(points, config.gamma)
        np.testing.assert_allclose(opened, reference, rtol=0, atol=1e-15)
        pivots, schur, valid = secure_eliminate(kernel, 3, config)
        score, valid_full, _ = secure_logdet(kernel, config)
        result = await mpc.output([score, valid, valid_full])
        assert result[1:] == [1, 1]
        assert result[0] == pytest.approx(stable_logdet(reference), abs=1e-11)
        expected = reference[3:, 3:] - reference[3:, :3] @ np.linalg.solve(reference[:3, :3], reference[:3, 3:])
        np.testing.assert_allclose(await mpc.output(schur), expected, atol=1e-15)
        await mpc.shutdown()
    mpc.run(check())


def test_log_normalization_and_singular_guard_do_not_open_private_pivots():
    async def check():
        await mpc.start()
        config = MPCConfig()
        fxp = mpc.SecFxp(config.bit_length, config.fractional_bits, p=FIELD_PRIME)
        values = np.array([1., .5, .125, 1e-9, 1.75])
        np.testing.assert_allclose(await mpc.output(secure_log_positive(fxp.array(values), config)),
                                   np.log(values), rtol=0, atol=1e-9)
        _, valid, _ = secure_logdet(fxp.array(np.ones((3, 3))), config)
        assert await mpc.output(valid) == 0
        await mpc.shutdown()
    mpc.run(check())


def test_small_integer_selector_cast_preserves_exact_bits_and_indices():
    async def check():
        await mpc.start()
        config = MPCConfig()
        fixed = mpc.SecFxp(config.bit_length, config.fractional_bits, p=FIELD_PRIME)
        integer = mpc.SecInt(10, p=FIELD_PRIME)
        values = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
        encoded = _cast_exact(integer.array(values), fixed, 2**config.fractional_bits)
        decoded = _cast_exact(encoded, integer, pow(2**config.fractional_bits, -1, FIELD_PRIME))
        np.testing.assert_array_equal(await mpc.output(encoded), values)
        np.testing.assert_array_equal(await mpc.output(decoded), values)
        bit = _cast_exact(fixed(1), integer, pow(2**config.fractional_bits, -1, FIELD_PRIME))
        assert await mpc.output(bit) == 1
        await mpc.shutdown()
    mpc.run(check())
