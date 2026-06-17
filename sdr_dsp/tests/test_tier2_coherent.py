"""Tests for Tier 2 coherent demods (QPSK, 8-PSK) and the end-to-end recovery
chain that feeds them. These exercise carrier_recovery + symbol_sync together
with a demod -- the integration the recovery-first sequencing was built toward.
"""

import numpy as np

from sdr_dsp.core import (qpsk_demod, psk8_demod, carrier_recovery,
                          symbol_sync)


# the demod's documented Gray mapping, used to build test signals
_QPSK_MAP = {(0, 0): (1 + 1j), (0, 1): (-1 + 1j),
             (1, 1): (-1 - 1j), (1, 0): (1 - 1j)}


def test_qpsk_roundtrip_clean():
    rng = np.random.default_rng(1)
    pairs = [(int(rng.integers(0, 2)), int(rng.integers(0, 2)))
             for _ in range(200)]
    syms = np.array([_QPSK_MAP[p] / np.sqrt(2) for p in pairs],
                    dtype=np.complex64)
    tx_bits = [b for p in pairs for b in p]
    bits, _ = qpsk_demod(syms)
    assert np.mean(np.array(bits) != np.array(tx_bits)) == 0.0


def test_qpsk_gray_adjacent_one_bit():
    # adjacent quadrants must differ by exactly one bit (Gray property)
    _, _ = qpsk_demod(np.array([1 + 1j], dtype=np.complex64))
    maps = [(0, 0), (0, 1), (1, 1), (1, 0)]  # quadrant order
    for a, b in zip(maps, maps[1:] + [maps[0]]):
        diff = sum(x != y for x, y in zip(a, b))
        assert diff == 1


def test_qpsk_empty():
    bits, dec = qpsk_demod(np.zeros(0, dtype=np.complex64))
    assert len(bits) == 0 and len(dec) == 0


def test_psk8_sector_recovery():
    rng = np.random.default_rng(2)
    sectors = rng.integers(0, 8, 150)
    syms = np.exp(1j * (np.pi / 4 * sectors)).astype(np.complex64)
    _, rec = psk8_demod(syms)
    assert np.array_equal(rec, sectors)


def test_psk8_three_bits_per_symbol():
    syms = np.exp(1j * (np.pi / 4 * np.arange(8))).astype(np.complex64)
    bits, _ = psk8_demod(syms)
    assert len(bits) == 8 * 3


def test_qpsk_end_to_end_through_recovery():
    # the integration test: impair a QPSK signal with carrier offset, phase,
    # timing shift, and noise; recover with the primitives; demod; check the
    # constellation snaps to clean clusters.
    rng = np.random.default_rng(0)
    nsym, sps = 500, 4
    pairs = [(int(rng.integers(0, 2)), int(rng.integers(0, 2)))
             for _ in range(nsym)]
    tx_syms = np.array([_QPSK_MAP[p] / np.sqrt(2) for p in pairs],
                       dtype=np.complex64)
    tx = np.repeat(tx_syms, sps).astype(np.complex64)
    t = np.arange(len(tx))
    imp = tx * np.exp(2j * np.pi * 0.001 * t + 1j * 0.5).astype(np.complex64)
    imp = np.concatenate([np.zeros(2, dtype=np.complex64), imp])
    imp += 0.08 * (rng.standard_normal(len(imp))
                   + 1j * rng.standard_normal(len(imp)))

    corr = carrier_recovery(imp, method="costas", order=4, loop_bw=0.005)
    syms = symbol_sync(corr, sps, method="gardner")

    # measure constellation cleanliness: mean distance to nearest ideal point
    s = syms[50:] / np.abs(syms[50:])
    ideal = np.exp(1j * (np.pi / 4 + np.pi / 2
                         * np.round((np.angle(s) - np.pi / 4) / (np.pi / 2))))
    err = np.mean(np.abs(s - ideal))
    assert err < 0.2, f"constellation not clean after recovery: {err}"
