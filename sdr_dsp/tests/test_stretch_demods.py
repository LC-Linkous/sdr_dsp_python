"""Tests for the stretch tier: QAM-16 (demonstrable), DSSS despread (known
code), FHSS hop detection (visualize). Honest scope -- these verify the
algorithms work on good signals, not robust reception.
"""

import numpy as np

from sdr_dsp.core import qam16_demod, dsss_despread, fhss_detect_hops


_QAM_GRAY = {(0, 0): -3, (0, 1): -1, (1, 1): 1, (1, 0): 3}


def test_qam16_roundtrip_clean():
    rng = np.random.default_rng(0)
    nsym = 400
    tx_bits = rng.integers(0, 2, nsym * 4)
    syms = []
    for k in range(nsym):
        i = _QAM_GRAY[(tx_bits[4*k], tx_bits[4*k+1])]
        q = _QAM_GRAY[(tx_bits[4*k+2], tx_bits[4*k+3])]
        syms.append(i + 1j * q)
    syms = np.array(syms, dtype=np.complex64)
    bits, _ = qam16_demod(syms)
    assert np.mean(np.array(bits) != tx_bits) == 0.0


def test_qam16_four_bits_per_symbol():
    syms = np.array([3 + 3j, -1 - 1j], dtype=np.complex64)
    bits, _ = qam16_demod(syms)
    assert len(bits) == 2 * 4


def test_qam16_empty():
    bits, pts = qam16_demod(np.zeros(0, dtype=np.complex64))
    assert len(bits) == 0 and len(pts) == 0


def test_dsss_despread_known_code():
    rng = np.random.default_rng(1)
    code = np.array([1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1],
                    dtype=np.complex64)
    data = np.array([1, -1, -1, 1, -1], dtype=np.complex64)
    spread = np.concatenate([d * code for d in data]).astype(np.complex64)
    spread += 0.5 * (rng.standard_normal(len(spread))
                     + 1j * rng.standard_normal(len(spread)))
    rec = dsss_despread(spread, code)
    assert np.array_equal(np.sign(rec.real), data.real)


def test_dsss_processing_gain():
    # despreading should pull a signal out of noise stronger than the raw chips
    rng = np.random.default_rng(2)
    code = np.array([1, -1, 1, -1, -1, 1, -1], dtype=np.complex64)
    data = np.array([1, 1, -1], dtype=np.complex64)
    spread = np.concatenate([d * code for d in data]).astype(np.complex64)
    noisy = spread + 2.0 * (rng.standard_normal(len(spread))
                            + 1j * rng.standard_normal(len(spread)))
    rec = dsss_despread(noisy, code)
    # even at this noise the signs should mostly survive (processing gain)
    assert np.array_equal(np.sign(rec.real), data.real)


def test_dsss_short_input():
    code = np.ones(10, dtype=np.complex64)
    assert len(dsss_despread(np.ones(5, dtype=np.complex64), code)) == 0


def test_fhss_detects_hops():
    fs = 1e6
    hop_freqs = [-200e3, 100e3, 300e3]
    seg = 2000
    parts = [np.exp(2j * np.pi * f * np.arange(seg) / fs) for f in hop_freqs]
    iq = np.concatenate(parts).astype(np.complex64)
    times, detected = fhss_detect_hops(iq, fs, nfft=256)
    found = sorted(set(np.round(detected / 1e5) * 1e5))
    # the three hop frequencies should be among the detected (within a bin)
    for hf in hop_freqs:
        assert any(abs(f - hf) < 50e3 for f in found), f"missed hop {hf}"


def test_fhss_empty():
    times, hops = fhss_detect_hops(np.zeros(10, dtype=np.complex64), 1e6,
                                   nfft=256)
    assert len(times) == 0 and len(hops) == 0
