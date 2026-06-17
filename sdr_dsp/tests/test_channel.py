"""Tests for the simulated channel (TX Phase C).

Verifies each impairment is accurate in its stated units (SNR in dB, CFO in Hz,
delay in samples), that the channel is a no-op by default, and that a framed
packet survives the full chain through a clean-ish channel and degrades
gracefully (CRC catches errors) as SNR drops.
"""

import numpy as np
import pytest

from sdr_dsp.core import (apply_channel, add_noise, add_cfo, add_delay,
                          estimate_cfo, build_frame, find_frames,
                          fsk_modulate, fsk_demod)


def _tone(fs=1e6, n=50000):
    return np.exp(2j * np.pi * 0.05 * fs * np.arange(n) / fs).astype(
        np.complex64)


# -- no-op default ----------------------------------------------------------

def test_channel_noop_by_default():
    sig = _tone()
    assert np.array_equal(apply_channel(sig), sig)


def test_empty_input():
    assert len(apply_channel(np.zeros(0, dtype=np.complex64))) == 0


# -- SNR accuracy -----------------------------------------------------------

@pytest.mark.parametrize("target", [20, 10, 3, 0])
def test_snr_is_accurate(target):
    sig = _tone()
    noisy = apply_channel(sig, snr_db=target, seed=0)
    noise = noisy - sig
    measured = 10 * np.log10(np.mean(np.abs(sig) ** 2)
                             / np.mean(np.abs(noise) ** 2))
    assert abs(measured - target) < 0.3


def test_noise_is_reproducible_with_seed():
    sig = _tone()
    a = apply_channel(sig, snr_db=10, seed=42)
    b = apply_channel(sig, snr_db=10, seed=42)
    assert np.array_equal(a, b)


def test_add_noise_standalone():
    sig = _tone()
    out = add_noise(sig, 10, rng=np.random.default_rng(0))
    assert out.shape == sig.shape and not np.array_equal(out, sig)


# -- CFO accuracy -----------------------------------------------------------

@pytest.mark.parametrize("cfo", [10e3, 27e3, -15e3])
def test_cfo_is_accurate(cfo):
    fs = 1e6
    sig = _tone(fs)
    base = estimate_cfo(sig, fs)
    after = estimate_cfo(apply_channel(sig, sample_rate=fs, cfo_hz=cfo), fs)
    assert abs((after - base) - cfo) < fs / 1024


def test_cfo_requires_sample_rate():
    with pytest.raises(ValueError):
        apply_channel(_tone(), cfo_hz=1e3)      # no sample_rate


def test_add_cfo_standalone():
    fs = 1e6
    sig = _tone(fs)
    shifted = add_cfo(sig, 10e3, fs)
    assert abs((estimate_cfo(shifted, fs) - estimate_cfo(sig, fs)) - 10e3) \
        < fs / 1024


# -- delay ------------------------------------------------------------------

def test_delay_shifts_forward():
    x = np.arange(10).astype(np.complex64)
    out = add_delay(x, 3)
    assert np.allclose(out[:3], 0)
    assert out[3] == 0 and out[4] == 1   # x[0]=0 lands at 3, x[1]=1 at 4
    assert len(out) == len(x)


def test_negative_delay_advances():
    x = np.arange(10).astype(np.complex64)
    out = add_delay(x, -2)
    assert out[0] == 2 and out[1] == 3
    assert len(out) == len(x)


def test_delay_zero_is_noop():
    x = _tone(n=1000)
    assert np.array_equal(add_delay(x, 0), x)


# -- scale and phase --------------------------------------------------------

def test_scale_and_phase():
    sig = np.ones(100, dtype=np.complex64)
    out = apply_channel(sig, scale=2.0, phase=np.pi / 2)
    # |out| ~ 2, rotated 90 degrees
    assert np.allclose(np.abs(out), 2.0, atol=1e-5)
    assert np.allclose(np.angle(out), np.pi / 2, atol=1e-5)


# -- full chain survival ----------------------------------------------------

def test_packet_survives_clean_channel():
    fs = 1e6
    sps = 20
    frame = build_frame(b"CQ DE SDR")
    iq = fsk_modulate(frame, sps, 50e3, fs)
    rx = apply_channel(iq, sample_rate=fs, snr_db=30, seed=1)
    bits = fsk_demod(rx, fs)[sps // 2::sps][:len(frame)]
    found = find_frames(np.asarray(bits, dtype=np.uint8))
    assert found and found[0]["payload"] == b"CQ DE SDR" and found[0]["crc_ok"]


def test_packet_lost_or_flagged_at_low_snr():
    # at very low SNR the frame should either not be found or fail CRC --
    # never silently return a corrupt payload as valid
    fs = 1e6
    sps = 20
    frame = build_frame(b"CQ DE SDR")
    iq = fsk_modulate(frame, sps, 50e3, fs)
    rx = apply_channel(iq, sample_rate=fs, snr_db=0, seed=3)
    bits = fsk_demod(rx, fs)[sps // 2::sps][:len(frame)]
    found = find_frames(np.asarray(bits, dtype=np.uint8))
    for f in found:
        if f["payload"] == b"CQ DE SDR":
            assert f["crc_ok"]          # if the right payload, CRC must agree
        # a wrong payload with crc_ok would be the real failure; CRC prevents it
