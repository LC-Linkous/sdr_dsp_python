"""End-to-end demodulation against the REAL capture, not just synthesis.

Every demod unit test elsewhere runs on synthetic IQ: a signal we generated,
demodulated, and checked against the bits/tone we put in. That proves the math
is self-consistent, but it cannot catch the ways a real capture differs from
a clean synthesis -- finite SNR, the LO/DC offset, a real (non-zero) CFO, the
anti-alias rolloff, quantization to int8. `sample_data/fm_2Msps.iq` is a real,
ear-verified 103.7 MHz broadcast (pilot +27.3 dB); these tests run the actual
`fm_receiver` demod chain on it and assert on properties of real broadcast
audio that hold for a genuine station and fail for noise.

They are deliberately tolerance-generous: the point is "does the chain recover
real program audio from a real capture", not bit-exactness (there are no known
bits in a music broadcast). The margins asserted here were measured on the
shipped capture with ~20 dB of headroom, so they pin the behavior without being
brittle to a future re-import of the same station.
"""

from pathlib import Path

import numpy as np
import pytest

from sdr_dsp.core import (deemphasis, design_lowpass, fir_apply, fm_demod,
                          resample_poly)
from sdr_dsp.io import load_iq

SAMPLE = (Path(__file__).resolve().parent.parent / "sample_data"
          / "fm_2Msps.iq")
pytestmark = pytest.mark.skipif(not SAMPLE.exists(),
                                reason="real sample_data not present")

FM_RATE = 250_000
AUDIO_RATE = 48_000


def _demod_chain(iq, fs):
    """The fm_receiver.py chain, condensed: channel filter -> decimate to
    FM_RATE -> discriminator -> resample to 48k -> de-emphasis -> 15 kHz
    audio lowpass. Kept in step with the example on purpose."""
    ch = fir_apply(iq, design_lowpass(100e3, fs, num_taps=201))[201:]
    decim = int(fs) // FM_RATE
    if decim > 1 and int(fs) % FM_RATE == 0:
        ch = ch[::decim]
        dr = FM_RATE
    else:
        from math import gcd
        g = gcd(FM_RATE, int(fs))
        ch = resample_poly(ch, FM_RATE // g, int(fs) // g)
        dr = FM_RATE
    audio = fm_demod(ch, deviation_hz=75e3, sample_rate=dr)
    from math import gcd
    g = gcd(AUDIO_RATE, dr)
    audio = resample_poly(audio, AUDIO_RATE // g, dr // g)
    audio = deemphasis(audio, AUDIO_RATE)
    audio = fir_apply(audio, design_lowpass(15_000, AUDIO_RATE,
                                            num_taps=101))[200:]
    return audio


def _band_power_db(audio, lo, hi, fs=AUDIO_RATE):
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    f = np.fft.rfftfreq(len(audio), 1.0 / fs)
    m = (f > lo) & (f < hi)
    return 10.0 * np.log10(np.mean(spec[m] ** 2) + 1e-20)


@pytest.fixture(scope="module")
def real_audio():
    iq, meta = load_iq(str(SAMPLE))
    fs = float(meta["global"]["core:sample_rate"])
    return _demod_chain(iq, fs)


def test_real_capture_demodulates_to_program_audio(real_audio):
    """The whole point: a real capture, through the real chain, yields real
    audio -- not silence, not clipping, at a sane level."""
    rms = np.sqrt(np.mean(real_audio ** 2))
    peak = np.max(np.abs(real_audio))
    assert rms > 0.05, f"near silence (rms {rms:.4f}) -- chain produced no audio"
    assert peak <= 1.0 + 1e-6, f"audio clips (peak {peak:.3f})"
    assert len(real_audio) > 40_000, "too little audio to judge"


def test_real_audio_has_broadcast_spectral_tilt(real_audio):
    """Broadcast program audio concentrates in the voice/music band and rolls
    off toward 15 kHz; demodulated noise is comparatively flat. The tilt is
    the signature that this is a station, not hiss."""
    low = _band_power_db(real_audio, 100, 5_000)
    high = _band_power_db(real_audio, 10_000, 15_000)
    assert low - high > 6.0, (
        f"program/high-band tilt only {low - high:.1f} dB; a real broadcast "
        f"measured ~12 dB, noise ~0")


def test_stereo_pilot_is_absent_from_the_mono_output(real_audio):
    """The 19 kHz pilot is strong in the composite (+27 dB) but the 15 kHz
    audio lowpass must keep it out of the mono WAV -- else it aliases and
    whines on cheap speakers."""
    program = _band_power_db(real_audio, 100, 5_000)
    pilot = _band_power_db(real_audio, 18_000, 20_000)
    assert program - pilot > 40.0, (
        f"pilot only {program - pilot:.1f} dB below program; the 15 kHz "
        f"audio filter is not suppressing it")


def test_real_demod_is_stable_across_the_capture(real_audio):
    """Audio level must not lurch between the start and end of the capture --
    a stage failing to carry state, or a transient, would show as a big
    half-to-half RMS swing."""
    iq, meta = load_iq(str(SAMPLE))
    fs = float(meta["global"]["core:sample_rate"])
    h1 = _demod_chain(iq[:len(iq) // 2], fs)
    h2 = _demod_chain(iq[len(iq) // 2:], fs)
    r1, r2 = np.sqrt(np.mean(h1 ** 2)), np.sqrt(np.mean(h2 ** 2))
    assert r1 > 0.05 and r2 > 0.05, "a half produced no audio"
    assert 0.5 < r1 / r2 < 2.0, (
        f"level lurched between halves (rms {r1:.3f} vs {r2:.3f}) -- "
        f"a stage may not be carrying state")


def test_noise_does_not_masquerade_as_program(real_audio):
    """The contrast case: the same chain on pure receiver-level noise must
    NOT show the broadcast tilt. Guards against the tilt test passing on
    something that isn't a station."""
    rng = np.random.default_rng(0)
    n = 2_000_000
    noise = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 2 / 128
             ).astype(np.complex64)
    na = _demod_chain(noise, 2e6)
    tilt = (_band_power_db(na, 100, 5_000)
            - _band_power_db(na, 10_000, 15_000))
    real_tilt = (_band_power_db(real_audio, 100, 5_000)
                 - _band_power_db(real_audio, 10_000, 15_000))
    assert real_tilt > tilt + 6.0, (
        f"real capture tilt ({real_tilt:.1f} dB) not clearly above noise "
        f"tilt ({tilt:.1f} dB) -- the discriminator between station and "
        f"hiss is too weak")
