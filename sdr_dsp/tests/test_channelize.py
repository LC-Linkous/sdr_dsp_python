"""Tests for channelization: single-channel extraction and the polyphase bank.

The headline test is the oracle: the efficient polyphase bank must agree with
the naive single-channel approach (channelize called per channel). Same
discipline as resampler-vs-scipy -- implement the fast version, prove it matches
the obvious one.
"""

import numpy as np
import pytest

from sdr_dsp.core import channelize, channelize_bank


def _tone(freq, fs, n=80000, amp=1.0):
    t = np.arange(n) / fs
    return (amp * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


# -- single channel ---------------------------------------------------------

def test_channelize_extracts_offset_channel():
    fs = 2e6
    # a tone at +300 kHz; extract that channel, it should come to baseband
    sig = _tone(300e3, fs)
    ch, rate = channelize(sig, fs, 300e3, 100e3)
    # after tuning to baseband the tone sits near DC -> low instantaneous freq
    inst = np.angle(ch[1:] * np.conj(ch[:-1]))
    assert abs(np.mean(inst)) < 0.1            # near 0 rad/sample (at baseband)


def test_channelize_reduces_rate():
    fs = 2e6
    sig = _tone(0, fs)
    ch, rate = channelize(sig, fs, 0, 100e3)
    assert rate < fs
    assert len(ch) < len(sig)


# -- polyphase bank: channel routing ---------------------------------------

@pytest.mark.parametrize("freq_mhz", [0, 1, 2, 3, -1, -2, -3])
def test_bank_routes_tone_to_correct_channel(freq_mhz):
    fs = 8e6
    N = 8
    sig = _tone(freq_mhz * 1e6, fs)
    chans, rate, freqs = channelize_bank(sig, fs, N)
    power = np.array([np.mean(np.abs(c) ** 2) for c in chans])
    peak = int(np.argmax(power))
    assert abs(freqs[peak] - freq_mhz * 1e6) < 1.0


def test_bank_shape_and_rate():
    fs = 8e6
    N = 8
    chans, rate, freqs = channelize_bank(_tone(0, fs), fs, N)
    assert chans.shape[0] == N
    assert rate == pytest.approx(fs / N)
    assert len(freqs) == N


def test_bank_center_freqs_ascending():
    fs = 8e6
    N = 8
    _, _, freqs = channelize_bank(_tone(0, fs), fs, N)
    assert np.all(np.diff(freqs) > 0)          # low -> high
    assert freqs[N // 2] == pytest.approx(0.0)  # middle channel at DC


def test_bank_rejects_other_channels():
    fs = 8e6
    N = 8
    sig = _tone(2e6, fs)                        # only one channel occupied
    chans, rate, freqs = channelize_bank(sig, fs, N)
    power = np.array([np.mean(np.abs(c) ** 2) for c in chans])
    peak = int(np.argmax(power))
    others = np.delete(power, peak)
    # the occupied channel should dominate by a wide margin
    assert power[peak] > 100 * others.max()


# -- the oracle: bank == naive loop ----------------------------------------

def test_bank_matches_naive_single_channel():
    fs = 8e6
    N = 8
    t = np.arange(80000) / fs
    # AM-modulated tone in one channel
    msg = 1 + 0.5 * np.cos(2 * np.pi * 5e3 * t)
    sig = (msg * np.exp(2j * np.pi * 2e6 * t)).astype(np.complex64)

    chans, rate, freqs = channelize_bank(sig, fs, N)
    ch_idx = int(np.argmin(np.abs(freqs - 2e6)))
    bank_ch = chans[ch_idx]

    naive, naive_rate = channelize(sig, fs, 2e6, fs / N, decim=N)

    assert rate == pytest.approx(naive_rate)
    m = min(len(bank_ch), len(naive))
    a = bank_ch[:m] / np.abs(bank_ch[:m]).max()
    b = naive[:m] / np.abs(naive[:m]).max()
    corr = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert corr > 0.95                          # same channel content


def test_bank_recovers_modulation():
    # the bank's channel output must carry the actual modulation, not just power
    fs = 8e6
    N = 8
    t = np.arange(80000) / fs
    msg = 1 + 0.5 * np.cos(2 * np.pi * 5e3 * t)
    sig = (msg * np.exp(2j * np.pi * 2e6 * t)).astype(np.complex64)
    chans, rate, freqs = channelize_bank(sig, fs, N)
    ch = chans[int(np.argmin(np.abs(freqs - 2e6)))]
    env = np.abs(ch) - np.mean(np.abs(ch))
    sp = np.abs(np.fft.rfft(env * np.hanning(len(env))))
    f = np.fft.rfftfreq(len(env), 1 / rate)
    assert abs(f[np.argmax(sp)] - 5e3) < 200    # recovers the 5 kHz AM tone


# -- options and edges ------------------------------------------------------

def test_oversampled_doubles_rate():
    fs = 8e6
    N = 8
    crit, rate_c, _ = channelize_bank(_tone(0, fs), fs, N, decim=N)
    over, rate_o, _ = channelize_bank(_tone(0, fs), fs, N, decim=N // 2)
    assert rate_o == pytest.approx(2 * rate_c)
    assert over.shape[1] > crit.shape[1]


def test_bank_invalid_decim_raises():
    with pytest.raises(ValueError):
        channelize_bank(_tone(0, 8e6), 8e6, 8, decim=3)   # 3 doesn't divide 8


def test_bank_invalid_nchannels_raises():
    with pytest.raises(ValueError):
        channelize_bank(_tone(0, 8e6), 8e6, 0)


def test_bank_empty_input():
    chans, rate, freqs = channelize_bank(
        np.zeros(0, dtype=np.complex64), 8e6, 8)
    assert chans.shape == (8, 0)


def test_bank_two_channels_simultaneously():
    # two occupied channels both recovered in the one pass
    fs = 8e6
    N = 8
    sig = (_tone(2e6, fs) + _tone(-3e6, fs, amp=0.7))
    chans, rate, freqs = channelize_bank(sig, fs, N)
    power = np.array([np.mean(np.abs(c) ** 2) for c in chans])
    top2 = sorted(np.argsort(power)[-2:])
    got = sorted(freqs[i] for i in top2)
    assert abs(got[0] - (-3e6)) < 1 and abs(got[1] - 2e6) < 1


# ---------------------------------------------------------------------------
# oversampled (decim = N//2) mode is a real polyphase bank, not interpolation
# ---------------------------------------------------------------------------

def _ctone(f_hz, fs, n):
    return np.exp(2j * np.pi * f_hz * np.arange(n) / fs).astype(np.complex64)


@pytest.mark.parametrize("k_ch", [2, 5])
def test_oversampled_bank_equals_direct_mix_filter_decimate(k_ch):
    # channel k of the oversampled bank must be exactly "mix down by the
    # channel center, lowpass with the prototype, keep every (N/2)-th sample"
    # -- the definition of a channelizer output -- up to a constant phase
    from sdr_dsp.core import design_lowpass, fir_apply, frequency_shift
    fs, N, L = 2e6, 8, 12
    _, _, cf = channelize_bank(np.zeros(N * L * 4, dtype=np.complex64), fs, N)
    x = _ctone(cf[k_ch] + 37e3, fs, 160_000)
    chans, rate, _ = channelize_bank(x, fs, N, decim=N // 2, taps_per_channel=L)
    assert rate == fs / (N // 2)
    y = chans[k_ch]
    proto = design_lowpass(fs / (2 * N), fs, num_taps=N * L)
    ref = fir_apply(frequency_shift(x, -cf[k_ch], fs), proto)[N - 1::N // 2][:len(y)]
    yy, rr = y[50:len(ref) - 50], ref[50:len(ref) - 50]
    corr = abs(np.vdot(rr, yy)) / np.sqrt(np.vdot(rr, rr).real * np.vdot(yy, yy).real)
    assert corr > 0.9999
    assert np.mean(np.abs(yy)) == pytest.approx(np.mean(np.abs(rr)), rel=1e-3)


def test_oversampled_bank_carries_the_transition_band_unaliased():
    # A tone 1.3x beyond the critically-sampled channel Nyquist (in the
    # prototype's transition band) FOLDS in the critically-sampled output and
    # reads at its true offset in the oversampled one. Interpolating the
    # critically-sampled output (what this mode used to do) cannot do this:
    # the fold has already happened.
    fs, N = 2e6, 8
    _, _, cf = channelize_bank(np.zeros(2000, dtype=np.complex64), fs, N)
    off = 1.3 * fs / (2 * N)                       # +162.5 kHz
    x = _ctone(cf[3] + off, fs, 160_000)

    def peak_hz(y, rate):
        y = y[200:]
        spec = np.abs(np.fft.fft(y * np.hanning(len(y))))
        return np.fft.fftfreq(len(y), 1 / rate)[np.argmax(spec)]

    crit, r_crit, _ = channelize_bank(x, fs, N, decim=N)
    over, r_over, _ = channelize_bank(x, fs, N, decim=N // 2)
    assert peak_hz(over[3], r_over) == pytest.approx(off, abs=r_over / 2000)
    folded = peak_hz(crit[3], r_crit)
    assert abs(folded - off) > fs / (2 * N)        # it is NOT at +162.5 kHz
    assert folded == pytest.approx(off - r_crit, abs=r_crit / 2000)   # it is at the fold


def test_oversampled_bank_isolates_neighbours():
    fs, N = 2e6, 8
    _, _, cf = channelize_bank(np.zeros(2000, dtype=np.complex64), fs, N)
    chans, _, _ = channelize_bank(_ctone(cf[4] + 10e3, fs, 160_000), fs, N, decim=N // 2)
    levels = 20 * np.log10(np.mean(np.abs(chans[:, 200:]), axis=1) + 1e-12)
    assert np.argmax(levels) == 4
    others = np.delete(levels, 4)
    leak = np.round(others - levels[4], 1)
    assert np.all(others < levels[4] - 80), f"neighbour leakage: {leak} dB"


def test_critically_sampled_output_unchanged_by_the_general_path():
    # the D = N path must be the bank it always was: same shape, same routing
    fs, N = 1e6, 4
    x = _ctone(130e3, fs, 40_000)
    chans, rate, cf = channelize_bank(x, fs, N)
    assert chans.shape == (N, len(x) // N) and rate == fs / N
    assert np.argmax(np.mean(np.abs(chans), axis=1)) == int(np.argmin(np.abs(cf - 130e3)))
