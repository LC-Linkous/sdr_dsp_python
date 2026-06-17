"""Tests for the detection module: matched filter finds a known pattern, and
detect_peak honors thresholds.
"""

import numpy as np

from sdr_dsp.core import matched_filter, detect_peak


def _embed(template, total, pos, snr_db, seed):
    rng = np.random.default_rng(seed)
    plen = len(template)
    snr_lin = 10 ** (snr_db / 10)
    scale = np.sqrt(snr_lin * plen * 2.0)
    noise = (rng.standard_normal(total)
             + 1j * rng.standard_normal(total)).astype(np.complex64)
    sig = noise.copy()
    sig[pos:pos + plen] += scale * template
    return sig


def _template(plen, seed=0):
    rng = np.random.default_rng(seed)
    t = (rng.standard_normal(plen) + 1j * rng.standard_normal(plen)
         ).astype(np.complex64)
    return t / np.linalg.norm(t)


def test_matched_filter_finds_pattern_clean():
    t = _template(64)
    sig = _embed(t, 4000, 1500, snr_db=10, seed=1)
    mf = matched_filter(sig, t)
    assert abs(int(np.argmax(mf)) - 1500) <= 2


def test_matched_filter_detects_at_0db_reliably():
    # a matched filter should detect easily at 0 dB input SNR
    hits = 0
    for seed in range(20):
        t = _template(64, seed)
        pos = 100 + seed * 50
        sig = _embed(t, 4000, pos, snr_db=0, seed=seed + 100)
        if abs(int(np.argmax(matched_filter(sig, t))) - pos) <= 2:
            hits += 1
    assert hits >= 18    # robust at 0 dB


def test_matched_filter_rejects_bad_args():
    import pytest
    with pytest.raises(ValueError):
        matched_filter(np.ones(10), np.array([]))
    with pytest.raises(ValueError):
        matched_filter(np.ones(10), np.ones(20))   # template longer


def test_detect_peak_threshold():
    t = _template(64)
    sig = _embed(t, 4000, 1500, snr_db=10, seed=2)
    idx, val = detect_peak(sig, t)
    assert abs(idx - 1500) <= 2
    # an absurdly high threshold => no detection
    idx2, _ = detect_peak(sig, t, threshold=val * 10)
    assert idx2 is None
