"""Filter tests: our FIR application is verified against scipy.signal.lfilter
(the oracle), and the designed filters actually attenuate what they should.
"""

import numpy as np
from scipy import signal as sig

from sdr_dsp.core import filters
from helpers.signals import tone, noise


def test_fir_apply_matches_scipy_lfilter_complex():
    # the core claim: our own convolution == scipy's lfilter on the same taps
    fs = 1_000_000
    taps = filters.design_lowpass(100_000, fs, num_taps=64)
    x = noise(5000, seed=1)
    ours = filters.fir_apply(x, taps)
    ref = sig.lfilter(taps, [1.0], x)
    assert ours.shape == ref.shape
    assert np.allclose(ours, ref, atol=1e-5)


def test_fir_apply_matches_scipy_lfilter_real():
    fs = 1_000_000
    taps = filters.design_lowpass(50_000, fs, num_taps=48)
    x = np.real(noise(4000, seed=2)).astype(np.float64)
    ours = filters.fir_apply(x, taps)
    ref = sig.lfilter(taps, [1.0], x)
    assert np.allclose(ours, ref, atol=1e-6)


def test_lowpass_attenuates_high_freq():
    # a tone above cutoff should be strongly attenuated; below, preserved
    fs = 1_000_000
    taps = filters.design_lowpass(100_000, fs, num_taps=128)
    low = tone(20_000, fs, 8192)       # passband
    high = tone(300_000, fs, 8192)     # stopband
    p_low = np.mean(np.abs(filters.fir_apply(low, taps)) ** 2)
    p_high = np.mean(np.abs(filters.fir_apply(high, taps)) ** 2)
    # stopband tone should be heavily suppressed relative to passband
    assert p_high < 0.05 * p_low


def test_bandpass_selects_band():
    fs = 1_000_000
    taps = filters.design_bandpass(80_000, 120_000, fs, num_taps=128)
    inband = tone(100_000, fs, 8192)
    outband = tone(300_000, fs, 8192)
    p_in = np.mean(np.abs(filters.fir_apply(inband, taps)) ** 2)
    p_out = np.mean(np.abs(filters.fir_apply(outband, taps)) ** 2)
    assert p_out < 0.05 * p_in


def test_highpass_forced_odd_length():
    fs = 1_000_000
    taps = filters.design_highpass(100_000, fs, num_taps=50)  # even -> bumped
    assert len(taps) % 2 == 1


def test_design_rejects_bad_cutoff():
    import pytest
    with pytest.raises(ValueError):
        filters.design_lowpass(600_000, 1_000_000)  # above Nyquist


def test_fir_apply_rejects_empty_taps():
    import pytest
    with pytest.raises(ValueError):
        filters.fir_apply(np.ones(10), np.array([]))
