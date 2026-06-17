"""Resample tests: our resampler verified against scipy.signal.resample_poly
(oracle) in steady state, plus rate-change and tone-preservation checks.
"""

import numpy as np
from scipy import signal as sig

from sdr_dsp.core import resample
from helpers.signals import tone


def test_resample_matches_scipy_steady_state():
    # our resampler should track scipy's in the middle (edges differ by design)
    fs = 48_000
    x = tone(1000, fs, 4800)
    ours = resample.resample_poly(x, 3, 2)
    ref = sig.resample_poly(x, 3, 2)
    m = min(len(ours), len(ref))
    a, b = ours[100:m - 100], ref[100:m - 100]
    corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert corr > 0.999


def test_decimate_reduces_rate():
    fs = 48_000
    x = tone(1000, fs, 4800)
    out = resample.decimate(x, 4)
    # ~1/4 the samples
    assert abs(len(out) - len(x) // 4) < 20


def test_interpolate_increases_rate():
    fs = 48_000
    x = tone(1000, fs, 1200)
    out = resample.interpolate(x, 3)
    assert abs(len(out) - len(x) * 3) < 20


def test_resample_identity():
    x = tone(1000, 48_000, 1000)
    out = resample.resample_poly(x, 1, 1)
    assert np.allclose(out, x)


def test_resample_reduces_ratio():
    # 4/2 should reduce to 2/1 internally and double the rate
    x = tone(1000, 48_000, 1000)
    out = resample.resample_poly(x, 4, 2)
    assert abs(len(out) - len(x) * 2) < 20


def test_resample_preserves_tone_frequency():
    # a 1 kHz tone resampled 2x is still 1 kHz at the new rate
    fs = 48_000
    x = tone(1000, fs, 4800)
    out = resample.resample_poly(x, 2, 1)
    new_fs = fs * 2
    spec = np.abs(np.fft.fft(out * np.hanning(len(out))))
    freqs = np.fft.fftfreq(len(out), 1.0 / new_fs)
    peak = abs(freqs[np.argmax(spec)])
    assert abs(peak - 1000) < 30


def test_resample_rejects_bad_factor():
    import pytest
    with pytest.raises(ValueError):
        resample.resample_poly(np.ones(10), 0, 1)
