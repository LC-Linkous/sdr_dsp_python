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
