"""Tests for correlate/convolve wrappers and the burst/CFO measurements."""

import numpy as np

from sdr_dsp.core import (correlate, convolve, find_bursts, estimate_cfo,
                          matched_filter)


def test_correlate_no_double_conjugation():
    # a complex template correlated against itself peaks at its energy, real.
    rng = np.random.default_rng(0)
    t = (rng.standard_normal(32) + 1j * rng.standard_normal(32))
    c = correlate(t, t, mode="valid")
    # peak magnitude == sum|t|^2 (no double-conjugation destroying it)
    assert abs(np.abs(c)[0] - np.sum(np.abs(t) ** 2)) < 1e-6


def test_convolve_matches_numpy():
    a = np.array([1, 2, 3], dtype=float)
    b = np.array([0, 1, 0.5], dtype=float)
    assert np.allclose(convolve(a, b), np.convolve(a, b))


def test_matched_filter_uses_correlate_correctly():
    # the matched filter and a manual correlate should agree on peak location
    rng = np.random.default_rng(1)
    t = (rng.standard_normal(64) + 1j * rng.standard_normal(64)).astype(
        np.complex64)
    t /= np.linalg.norm(t)
    sig = np.zeros(2000, dtype=np.complex64)
    sig[800:864] = t * 10
    assert abs(int(np.argmax(matched_filter(sig, t))) - 800) <= 2


def test_find_bursts_single():
    sig = np.zeros(10000, dtype=np.complex64)
    sig[3000:5000] = 1.0
    bursts = find_bursts(sig)
    assert len(bursts) == 1
    s, e = bursts[0]
    assert abs(s - 3000) < 50 and abs(e - 5000) < 50


def test_find_bursts_merge_gap():
    sig = np.zeros(10000, dtype=np.complex64)
    sig[1000:2000] = 1.0
    sig[2100:3000] = 1.0     # 100-sample gap
    merged = find_bursts(sig, min_gap=200)
    assert len(merged) == 1   # bridged into one


def test_find_bursts_min_len():
    sig = np.zeros(10000, dtype=np.complex64)
    sig[1000:1005] = 1.0      # tiny blip
    sig[3000:5000] = 1.0      # real burst
    bursts = find_bursts(sig, min_len=100)
    assert len(bursts) == 1   # blip rejected


def test_find_bursts_empty():
    assert find_bursts(np.zeros(100, dtype=np.complex64)) == []


def test_estimate_cfo_finds_offset():
    fs = 1e6
    tone = np.exp(2j * np.pi * 120e3 * np.arange(8192) / fs).astype(np.complex64)
    assert abs(estimate_cfo(tone, fs) - 120e3) < fs / 8192


def test_estimate_cfo_measures_does_not_apply():
    # estimate_cfo returns a number; the signal must be UNCHANGED (no auto-tune)
    fs = 1e6
    tone = np.exp(2j * np.pi * 80e3 * np.arange(4096) / fs).astype(np.complex64)
    before = tone.copy()
    _ = estimate_cfo(tone, fs)
    assert np.array_equal(tone, before)
