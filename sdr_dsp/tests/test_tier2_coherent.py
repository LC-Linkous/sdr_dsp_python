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
    # Ask the DEMOD for the bit pair at each ideal point, walking the quadrants
    # counter-clockwise (0 -> 90 -> 180 -> 270 degrees); consecutive pairs must
    # differ in exactly one bit. (An earlier version of this test asserted the
    # Gray property of a list typed into the test and discarded the demod
    # output -- it could not detect a wrong mapping in the code.)
    points = [1 + 1j, -1 + 1j, -1 - 1j, 1 - 1j]
    decoded = []
    for p in points:
        bits, _ = qpsk_demod(np.array([p], dtype=np.complex64))
        decoded.append(tuple(int(b) for b in bits))
    assert len(set(decoded)) == 4, f"quadrants not distinct: {decoded}"
    for a, b in zip(decoded, decoded[1:] + [decoded[0]], strict=True):
        diff = sum(x != y for x, y in zip(a, b, strict=True))
        assert diff == 1, f"{a} -> {b} differs in {diff} bits"
    # and it is the documented mapping the modulator-side tests rely on
    inverse = {v: k for k, v in _QPSK_MAP.items()}
    assert decoded == [inverse[p] for p in points]


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

    def cleanliness(symbols):
        # mean distance from each (normalized) symbol to its nearest ideal point
        s = symbols[50:] / np.abs(symbols[50:])
        ideal = np.exp(1j * (np.pi / 4 + np.pi / 2
                             * np.round((np.angle(s) - np.pi / 4) / (np.pi / 2))))
        return float(np.mean(np.abs(s - ideal)))

    err = cleanliness(syms)
    assert err < 0.2, f"constellation not clean after recovery: {err}"
    # The 0.2 bound alone does not show symbol_sync contributes: with
    # rectangular pulses at 4 sps, plain stride-sampling the carrier-corrected
    # signal already scores ~0.06. Timing recovery must do BETTER than the
    # best naive stride, or it isn't recovering anything.
    best_stride = min(cleanliness(corr[off::sps]) for off in range(sps))
    assert err < best_stride, (
        f"symbol_sync ({err:.4f}) no better than naive stride ({best_stride:.4f})")
    # and carrier recovery must matter too: without it the constellation spins
    assert cleanliness(symbol_sync(imp, sps, method="gardner")) > 0.3


def test_psk8_bit_mapping_is_gray_and_sector_consistent():
    # the eight ideal points, walked counter-clockwise: consecutive sectors'
    # 3-bit labels differ in exactly one bit, all eight labels are distinct,
    # and the reported sector matches the point. A sector-swap mutation that
    # broke the Gray property passed every existing test (they checked only
    # the sector index and bit count).
    labels = []
    for k in range(8):
        pt = np.exp(1j * k * np.pi / 4).astype(np.complex64)          # sector k at k*45 deg
        bits, sector = psk8_demod(np.array([pt]))
        assert int(sector[0]) == k
        labels.append(tuple(int(b) for b in bits))
    assert len(set(labels)) == 8
    for a, b in zip(labels, labels[1:] + labels[:1], strict=True):
        assert sum(x != y for x, y in zip(a, b, strict=True)) == 1, f"{a} -> {b}"
