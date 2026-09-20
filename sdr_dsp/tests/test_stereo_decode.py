"""fm_stereo_decode: pilot-locked L/R recovery, checked against known truth.

The synthetic multiplex has exact L and R content, so channel separation is
measurable to the dB; the real 103.7 MHz capture has genuine (unknown) stereo,
so we can only assert that real L-R content is recovered. Both matter: the
synthetic test proves the algorithm, the real test proves it survives a real
capture's SNR, CFO, and quantization.
"""

from pathlib import Path

import numpy as np
import pytest

from sdr_dsp.core import fm_stereo_decode

COMP_FS = 250_000          # composite/demod rate the receivers use
PILOT_HZ = 19_000.0
SAMPLE = (Path(__file__).resolve().parent.parent / "sample_data"
          / "fm_2Msps.iq")


def _multiplex(left, right, fs, pilot_level=0.09):
    """Build the textbook stereo MPX from known L/R (the exact form the
    synthetic generator and every FM stereo transmitter use)."""
    t = np.arange(len(left)) / fs
    pilot_ph = 2 * np.pi * PILOT_HZ * t
    return ((left + right)
            + (left - right) * np.cos(2 * pilot_ph)
            + pilot_level * np.cos(pilot_ph))


def _tone(freq, n, fs, amp=0.8):
    return amp * np.sin(2 * np.pi * freq * np.arange(n) / fs)


def _tone_power_db(x, freq, fs=COMP_FS):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    m = (f > freq - 40) & (f < freq + 40)
    return 20.0 * np.log10(spec[m].max() + 1e-20)


def _rms(x):
    return np.sqrt(np.mean(x ** 2))


# ---------------------------------------------------------------------------
# synthetic: exact ground truth
# ---------------------------------------------------------------------------
def test_separation_distinct_tones_per_channel():
    """L carries 700 Hz, R carries 1500 Hz. After decode each channel must
    show its own tone far above the other -- that gap IS stereo separation."""
    n = int(COMP_FS * 0.5)
    left_in = _tone(700, n, COMP_FS)
    right_in = _tone(1500, n, COMP_FS)
    mpx = _multiplex(left_in, right_in, COMP_FS)
    left, right = fm_stereo_decode(mpx, COMP_FS)
    s = 1500
    left, right = left[s:-s], right[s:-s]

    left_sep = _tone_power_db(left, 700) - _tone_power_db(left, 1500)
    right_sep = _tone_power_db(right, 1500) - _tone_power_db(right, 700)
    assert left_sep > 20, f"left separation only {left_sep:.1f} dB"
    assert right_sep > 20, f"right separation only {right_sep:.1f} dB"


def test_mono_signal_reconstructs_equally():
    """L == R (a mono program sent through the stereo MPX) must come back with
    both channels equal and at the input amplitude -- the matrix and scale
    are correct only if this holds."""
    n = int(COMP_FS * 0.5)
    mono = _tone(700, n, COMP_FS, amp=0.8)
    mpx = _multiplex(mono, mono, COMP_FS)
    left, right = fm_stereo_decode(mpx, COMP_FS)
    s = 1500
    left, right = left[s:-s], right[s:-s]
    # channels equal
    assert _rms(left - right) / _rms(left) < 0.1, "mono did not reconstruct L==R"
    # amplitude preserved (unnormalized, like fm_demod): peak ~ 0.8
    assert abs(_rms(left) * np.sqrt(2) - 0.8) < 0.1, (
        f"amplitude off: {_rms(left) * np.sqrt(2):.3f} vs 0.8")


def test_left_only_stays_left():
    """Signal on L, silence on R -> recovered R must be far below L."""
    n = int(COMP_FS * 0.5)
    left_in = _tone(700, n, COMP_FS, amp=0.8)
    right_in = np.zeros(n)
    mpx = _multiplex(left_in, right_in, COMP_FS)
    left, right = fm_stereo_decode(mpx, COMP_FS)
    s = 1500
    left, right = left[s:-s], right[s:-s]
    sep = 20 * np.log10(_rms(left) / (_rms(right) + 1e-12))
    assert sep > 15, f"L-only bled into R: only {sep:.1f} dB apart"


def test_sum_of_channels_is_mono_compatible():
    """(L+R)/2 from the decoder must equal what a mono receiver (just the
    lowpassed composite) would produce -- stereo decoding must not change the
    mono sum."""
    n = int(COMP_FS * 0.5)
    left_in = _tone(700, n, COMP_FS, amp=0.7)
    right_in = _tone(1500, n, COMP_FS, amp=0.5)
    mpx = _multiplex(left_in, right_in, COMP_FS)
    left, right = fm_stereo_decode(mpx, COMP_FS)
    # A mono receiver is just the centered-lowpassed composite -- the SAME
    # filtering the decoder's sum path uses, so they align in time.
    from sdr_dsp.core import design_lowpass, fir_apply_centered
    mono_ref = fir_apply_centered(
        mpx, design_lowpass(15_000, COMP_FS, num_taps=int(COMP_FS / 800) | 1))
    s = 1500
    stereo_sum = (left + right)[s:-s]      # == sum path == mono
    mono_ref = mono_ref[s:-s]
    err = _rms(stereo_sum - mono_ref) / _rms(mono_ref)
    assert err < 0.05, f"stereo sum diverges from mono by {err:.3f}"


def test_short_input_returns_empty_pair():
    l, r = fm_stereo_decode(np.zeros(2), COMP_FS)
    assert len(l) == 2 and len(r) == 2


# ---------------------------------------------------------------------------
# real capture: genuine stereo must be recovered
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SAMPLE.exists(), reason="real sample not present")
def test_real_capture_has_recoverable_stereo():
    """The 103.7 MHz capture is a real stereo broadcast. Decoding it must
    yield genuine L-R (side) content -- a mono station would give side ~ 0."""
    from sdr_dsp.io import load_iq
    from sdr_dsp.core import design_lowpass, fir_apply, fm_demod

    iq, meta = load_iq(str(SAMPLE))
    fs = float(meta["global"]["core:sample_rate"])
    ch = fir_apply(iq, design_lowpass(100e3, fs, num_taps=201))[201:]
    ch = ch[::int(fs) // COMP_FS]
    composite = fm_demod(ch, deviation_hz=75e3, sample_rate=COMP_FS)

    left, right = fm_stereo_decode(composite, COMP_FS)
    s = 3000
    left, right = left[s:-s], right[s:-s]
    mono = (left + right) / 2
    side = (left - right) / 2
    ratio = _rms(side) / _rms(mono)
    # real music has meaningful but not dominant side energy; a mono station
    # would sit near 0. Measured ~0.4 on this capture.
    assert ratio > 0.05, (
        f"no stereo content recovered (side/mono {ratio:.3f}); either the "
        f"decoder failed or the pilot lock was lost")
    assert ratio < 1.5, f"implausible side/mono {ratio:.3f} -- decode unstable"
