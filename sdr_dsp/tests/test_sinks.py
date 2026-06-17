"""Tests for sinks: WAV writing and processed-IQ round-trip through io."""

import wave

import numpy as np

from sdr_dsp.sinks import write_wav, write_iq
from sdr_dsp.io import load_iq


def test_write_wav_valid_file(tmp_path):
    audio = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
    path = write_wav(tmp_path / "a.wav", audio, 48000)
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 48000
        assert w.getsampwidth() == 2
        assert w.getnframes() == 48000


def test_write_wav_normalizes(tmp_path):
    # a tiny-amplitude signal should still use most of the int16 range
    audio = 0.001 * np.sin(np.arange(1000) * 0.1)
    path = write_wav(tmp_path / "q.wav", audio, 48000, normalize=True)
    with wave.open(path, "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    assert np.max(np.abs(pcm)) > 20000   # scaled up near full range


def test_write_iq_roundtrip(tmp_path):
    x = (np.random.randn(2000) + 1j * np.random.randn(2000)).astype(np.complex64)
    dp, mp = write_iq(tmp_path / "c.sigmf-data", x, 2e6, center_freq=100e6)
    back, meta = load_iq(mp)
    assert np.max(np.abs(x - back)) == 0.0   # cf32 lossless
    assert meta["global"]["core:sample_rate"] == 2e6
