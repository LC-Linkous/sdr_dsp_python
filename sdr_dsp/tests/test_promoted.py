"""Tests for functions promoted from examples into the library:
remove_dc and deemphasis.
"""

import numpy as np
import pytest

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


@pytest.mark.parametrize("tau_us,corner_hz", [(75.0, 2122.1), (50.0, 3183.1)])
def test_deemphasis_time_constant_is_pinned(tau_us, corner_hz):
    # The -3 dB corner of the one-pole network is 1/(2*pi*tau): 2122 Hz for the
    # 75 us Americas/Korea standard, 3183 Hz for 50 us elsewhere. The other
    # de-emphasis tests only check "highs are attenuated", which any lowpass
    # -- and either constant -- satisfies. Measure a tone AT the corner: it
    # must come out 3 dB down (+/-0.3 dB), and a tone at the OTHER standard's
    # corner must not.
    from sdr_dsp.core import deemphasis
    fs = 48_000
    t = np.arange(int(fs * 0.5)) / fs

    def gain_db(f_hz):
        x = np.sin(2 * np.pi * f_hz * t)
        y = deemphasis(x, fs, tau_us=tau_us)[len(t) // 2:]       # steady state
        return 20 * np.log10(np.sqrt(np.mean(y ** 2)) / np.sqrt(0.5))

    assert gain_db(corner_hz) == pytest.approx(-3.01, abs=0.3)
    assert gain_db(100.0) == pytest.approx(0.0, abs=0.1)       # passband flat
    other = 3183.1 if tau_us == 75.0 else 2122.1
    assert abs(gain_db(other) + 3.01) > 1.0                     # distinguishes the constants
