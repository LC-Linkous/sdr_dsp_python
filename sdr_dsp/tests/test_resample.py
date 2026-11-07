"""Resample tests: our resampler verified against scipy.signal.resample_poly
(oracle) in steady state, plus rate-change and tone-preservation checks.
"""

import numpy as np
import pytest
from scipy import signal as sig

from sdr_dsp.core import resample
from helpers.signals import tone


@pytest.mark.parametrize("up,down", [(3, 2), (1, 8), (5, 1), (48, 125)])
def test_resample_preserves_absolute_amplitude(up, down):
    # The scipy comparison below is a NORMALIZED correlation, so it is blind to
    # gain: a resampler that forgot the `up` gain compensation (output 1/up of
    # the right level) would still score 0.999. Resampling must preserve the
    # amplitude of an in-band tone, and match scipy's level, not just its shape.
    fs = 48_000
    x = tone(1000, fs, 48_000)
    ours = resample.resample_poly(x, up, down)
    ref = sig.resample_poly(x, up, down)
    m = min(len(ours), len(ref))
    a, b = ours[200:m - 200], ref[200:m - 200]
    assert np.mean(np.abs(a)) == pytest.approx(np.mean(np.abs(x)), rel=1e-2)
    assert np.mean(np.abs(a)) == pytest.approx(np.mean(np.abs(b)), rel=1e-2)
    # the samples themselves agree, not just their statistics (0.5%: our
    # default taps are not scipy's, so steady-state differs at the 1e-3 level)
    assert np.max(np.abs(a - b)) < 5e-3 * np.max(np.abs(x))


@pytest.mark.parametrize("up,down", [(2.7, 1), (1, 2.5), (3.0001, 2), (True, 1)])
def test_resample_rejects_non_integer_ratio(up, down):
    # int(2.7) == 2 silently turned a requested 2.7x into 2x; the ratio must be
    # integers (floats that ARE integral, like 2.0, are fine)
    x = tone(1000, 48_000, 4800)
    with pytest.raises(ValueError, match="positive integer"):
        resample.resample_poly(x, up, down)


def test_resample_accepts_integral_floats_and_numpy_ints():
    x = tone(1000, 48_000, 4800)
    a = resample.resample_poly(x, 3, 2)
    assert np.array_equal(resample.resample_poly(x, 3.0, 2.0), a)
    assert np.array_equal(resample.resample_poly(x, np.int64(3), np.int32(2)), a)


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


def test_decimation_rejects_aliasing():
    # THE anti-aliasing test: a tone above the post-decimation Nyquist must be
    # rejected, not folded back. Without a working anti-alias filter it would
    # alias to a spurious low frequency.
    import numpy as np
    from helpers.signals import tone
    fs = 1_000_000
    # decimate by 4 -> new Nyquist is 125 kHz. Put a tone at 300 kHz (above it).
    x = tone(300_000, fs, 40_000)
    out = resample.decimate(x, 4)
    # the aliased tone, if anti-aliasing failed, would appear strongly. Measure
    # residual power: it should be heavily suppressed vs the input power.
    in_power = float(np.mean(np.abs(x) ** 2))
    out_power = float(np.mean(np.abs(out) ** 2))
    assert out_power < 0.05 * in_power   # >13 dB rejection of out-of-band tone


def test_decimation_preserves_in_band():
    # sanity companion: a tone BELOW the new Nyquist survives decimation
    import numpy as np
    from helpers.signals import tone
    fs = 1_000_000
    x = tone(50_000, fs, 40_000)         # 50 kHz, well under 125 kHz Nyquist
    out = resample.decimate(x, 4)
    in_power = float(np.mean(np.abs(x) ** 2))
    out_power = float(np.mean(np.abs(out) ** 2))
    assert out_power > 0.5 * in_power    # in-band tone mostly preserved


def test_resample_empty_and_short():
    import numpy as np
    # empty input -> empty output, no crash
    out = resample.resample_poly(np.array([], dtype=np.complex64), 3, 2)
    assert len(out) == 0


# ---------------------------------------------------------------------------
# The polyphase form must equal the textbook zero-stuff form it replaces.
# The direct form filters `up` times more samples than it keeps (most of them
# zeros): 4.9 s vs 0.4 s for 500k samples at 24/125. Same taps, same math --
# so the two must agree to rounding, on every phase, both dtypes, and inputs
# shorter than the filter.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("up,down", [(1, 4), (3, 2), (24, 125), (125, 24),
                                     (7, 3), (6, 50), (48, 10)])
@pytest.mark.parametrize("n", [1, 3, 777, 20_000])
@pytest.mark.parametrize("cplx", [False, True])
def test_polyphase_equals_direct_form(up, down, n, cplx):
    from math import gcd
    rng = np.random.default_rng(n * up + down)
    x = rng.standard_normal(n)
    if cplx:
        x = (x + 1j * rng.standard_normal(n)).astype(np.complex64)
    g = gcd(up, down)
    taps = resample._antialias_taps(up // g, down // g)
    ref = resample._resample_poly_direct(x, up // g, down // g, taps)
    got = resample.resample_poly(x, up, down)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-5 if cplx else 1e-12)
