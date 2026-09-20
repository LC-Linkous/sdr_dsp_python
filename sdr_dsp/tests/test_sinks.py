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
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(2000)
         + 1j * rng.standard_normal(2000)).astype(np.complex64)
    dp, mp = write_iq(tmp_path / "c.sigmf-data", x, 2e6, center_freq=100e6)
    back, meta = load_iq(mp)
    assert np.max(np.abs(x - back)) == 0.0   # cf32 lossless
    assert meta["global"]["core:sample_rate"] == 2e6


def test_write_wav_stereo_two_channels(tmp_path):
    n = 4800
    left = np.sin(2 * np.pi * 440 * np.arange(n) / 48000)
    right = np.sin(2 * np.pi * 660 * np.arange(n) / 48000)
    audio = np.column_stack([left, right])
    path = write_wav(tmp_path / "st.wav", audio, 48000)
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 2
        assert w.getnframes() == n
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    # interleaved L,R,L,R... -> the two channels differ (different tones)
    assert not np.array_equal(pcm[0::2], pcm[1::2])


def test_write_wav_stereo_shared_normalization(tmp_path):
    """One channel loud, one quiet: shared scale must keep their RATIO, not
    normalize each to full scale independently (which would centre the image)."""
    n = 4800
    loud = 0.8 * np.sin(2 * np.pi * 440 * np.arange(n) / 48000)
    quiet = 0.1 * np.sin(2 * np.pi * 440 * np.arange(n) / 48000)
    path = write_wav(tmp_path / "bal.wav", np.column_stack([loud, quiet]),
                     48000, normalize=True)
    with wave.open(path, "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    lr = np.max(np.abs(pcm[0::2])) / np.max(np.abs(pcm[1::2]))
    assert 6 < lr < 10, f"L/R ratio {lr:.1f} not preserved near 8:1"


def test_write_wav_rejects_three_channel(tmp_path):
    import pytest
    bad = np.zeros((100, 3))
    with pytest.raises(ValueError):
        write_wav(tmp_path / "bad.wav", bad, 48000)


def test_write_wav_mono_still_one_channel(tmp_path):
    """Regression: the mono path must be untouched by the stereo addition."""
    audio = np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000)
    path = write_wav(tmp_path / "m.wav", audio, 48000)
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1
        assert w.getnframes() == 4800
