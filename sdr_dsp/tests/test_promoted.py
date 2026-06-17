"""Tests for functions promoted from examples into the library:
remove_dc and deemphasis.
"""

import numpy as np

from sdr_dsp.core import remove_dc, deemphasis


def test_remove_dc_zeroes_mean():
    x = (np.full(1000, 0.5 + 0.2j)
         + 0.1 * np.exp(2j * np.pi * 0.1 * np.arange(1000))).astype(np.complex64)
    out = remove_dc(x)
    assert abs(np.mean(out)) < 1e-5


def test_remove_dc_empty():
    out = remove_dc(np.array([], dtype=np.complex64))
    assert len(out) == 0


def test_remove_dc_preserves_ac():
    # the AC (non-DC) content should be essentially unchanged
    ac = (0.3 * np.exp(2j * np.pi * 0.05 * np.arange(1000))).astype(np.complex64)
    x = ac + (0.5 + 0.2j)
    out = remove_dc(x)
    assert np.allclose(out, ac, atol=1e-4)


def test_deemphasis_attenuates_highs():
    # de-emphasis is a lowpass: a high-frequency tone should be attenuated
    # more than a low-frequency one.
    fs = 48_000
    n = 10_000
    t = np.arange(n) / fs
    low = np.cos(2 * np.pi * 500 * t)
    high = np.cos(2 * np.pi * 10_000 * t)
    lo_out = deemphasis(low, fs)
    hi_out = deemphasis(high, fs)
    # compare output amplitude (std) -- high should be reduced more
    assert np.std(hi_out) < np.std(lo_out)


def test_deemphasis_length_preserved():
    out = deemphasis(np.ones(500), 48_000)
    assert len(out) == 500
