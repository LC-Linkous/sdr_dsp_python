"""Demod tests: recover known messages/bits from synthetic signals we built,
so correctness is checked against ground truth we control.
"""

import numpy as np

from sdr_dsp.core import demod
from helpers.signals import fm_signal, ook_burst, tone


def test_fm_demod_recovers_message_frequency():
    # FM-modulate a known 1 kHz message; the demod output should contain that
    # tone as its dominant frequency.
    fs = 200_000
    msg_hz = 1000
    iq, message = fm_signal(msg_hz, deviation_hz=10_000, sample_rate=fs,
                            n=20_000)
    out = demod.fm_demod(iq, deviation_hz=10_000, sample_rate=fs)
    # find dominant frequency in the recovered audio
    spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
    freqs = np.fft.rfftfreq(len(out), 1.0 / fs)
    peak = freqs[np.argmax(spec)]
    assert abs(peak - msg_hz) < 50    # recovered tone near 1 kHz


def test_fm_demod_short_input():
    assert len(demod.fm_demod(np.ones(1, dtype=np.complex64))) == 0


def test_am_demod_envelope():
    # amplitude-modulate: carrier scaled by a slow envelope; demod recovers it
    fs = 100_000
    n = 10_000
    t = np.arange(n) / fs
    env = 1.0 + 0.5 * np.cos(2 * np.pi * 500 * t)     # 500 Hz AM
    iq = (env * np.exp(2j * np.pi * 0 * t)).astype(np.complex64)
    out = demod.am_demod(iq, dc_block=True)
    spec = np.abs(np.fft.rfft(out * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    peak = freqs[np.argmax(spec)]
    assert abs(peak - 500) < 20


def test_ook_recovers_bits_clean():
    # build a known bit pattern, recover it from the envelope
    pattern = [1, 0, 1, 1, 0, 0, 1, 0]
    spb = 100
    iq, bits = ook_burst(pattern, spb, sample_rate=100_000, amp=1.0)
    env = demod.ook_envelope(iq)
    sliced = demod.ook_slice(env)
    # downsample the per-sample decisions back to per-bit by majority vote
    recovered = []
    for i in range(len(pattern)):
        chunk = sliced[i * spb:(i + 1) * spb]
        recovered.append(1 if chunk.mean() > 0.5 else 0)
    assert recovered == pattern


def test_ook_recovers_bits_with_noise():
    pattern = [1, 1, 0, 1, 0, 1, 0, 0, 1, 1]
    spb = 200
    iq, bits = ook_burst(pattern, spb, sample_rate=100_000, amp=1.0,
                         noise_sigma=0.1, seed=3)
    env = demod.ook_envelope(iq)
    sliced = demod.ook_slice(env)
    recovered = [1 if sliced[i*spb:(i+1)*spb].mean() > 0.5 else 0
                 for i in range(len(pattern))]
    assert recovered == list(pattern)


def test_ook_slice_custom_threshold():
    env = np.array([0.0, 0.1, 0.9, 1.0])
    out = demod.ook_slice(env, threshold=0.5)
    assert list(out) == [0, 0, 1, 1]


def test_fm_demod_linearity_across_frequencies():
    # the demod must track the message frequency across a RANGE, not just at
    # one lucky point. Sweep several message tones; each must be recovered.
    fs = 200_000
    for msg_hz in (500, 1000, 2500, 5000, 10_000):
        iq, _ = fm_signal(msg_hz, deviation_hz=20_000, sample_rate=fs, n=40_000)
        out = demod.fm_demod(iq, deviation_hz=20_000, sample_rate=fs)
        spec = np.abs(np.fft.rfft(out * np.hanning(len(out))))
        freqs = np.fft.rfftfreq(len(out), 1.0 / fs)
        peak = freqs[np.argmax(spec)]
        assert abs(peak - msg_hz) < 100, f"failed at {msg_hz} Hz (got {peak})"


def test_fm_demod_deviation_scaling():
    # doubling the deviation should double the recovered amplitude (the
    # discriminator output is proportional to instantaneous frequency).
    fs = 200_000
    msg_hz = 1000
    iq1, _ = fm_signal(msg_hz, deviation_hz=10_000, sample_rate=fs, n=40_000)
    iq2, _ = fm_signal(msg_hz, deviation_hz=20_000, sample_rate=fs, n=40_000)
    # demod WITHOUT the deviation normalization to see the raw scaling
    out1 = demod.fm_demod(iq1)
    out2 = demod.fm_demod(iq2)
    amp1 = np.std(out1)
    amp2 = np.std(out2)
    ratio = amp2 / amp1
    assert abs(ratio - 2.0) < 0.2, f"deviation scaling off: ratio {ratio}"
