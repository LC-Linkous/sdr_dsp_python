"""Tests for the phase primitives and the new demods (FSK/SSB/BPSK)."""

import numpy as np

from sdr_dsp.core import (
    instantaneous_frequency, instantaneous_phase,
    fsk_demod, ssb_demod, bpsk_demod, fm_demod,
    estimate_symbol_rate, slice_to_symbols,
)


def _tone(f, fs, n):
    return np.exp(2j * np.pi * f * np.arange(n) / fs).astype(np.complex64)


def test_instantaneous_frequency_constant_for_tone():
    fs = 1e6
    freq = instantaneous_frequency(_tone(100e3, fs, 1000), sample_rate=fs)
    assert abs(np.median(freq) - 100e3) < 100


def test_instantaneous_frequency_short_input():
    assert len(instantaneous_frequency(np.ones(1, dtype=np.complex64))) == 0


def test_instantaneous_phase_unwraps():
    fs = 1e6
    ph = instantaneous_phase(_tone(10e3, fs, 1000), unwrap=True)
    # unwrapped phase of a positive tone increases monotonically
    assert np.all(np.diff(ph) > 0)


def test_fm_demod_uses_primitive_consistently():
    # fm_demod (raw) should equal instantaneous_frequency in rad/sample
    fs = 1e6
    x = _tone(50e3, fs, 1000)
    assert np.allclose(fm_demod(x), instantaneous_frequency(x), atol=1e-6)


def test_fsk_demod_recovers_bits():
    fs = 1e6
    pattern = [1, 0, 1, 1, 0, 0, 1, 0]
    spb = 200
    parts = [_tone(50e3 if b else -50e3, fs, spb) for b in pattern]
    iq = np.concatenate(parts).astype(np.complex64)
    bits = fsk_demod(iq, fs)
    spb_est, _ = estimate_symbol_rate(bits, fs)
    syms = slice_to_symbols(bits, spb_est)
    assert list(syms) == pattern


def test_ssb_usb_recovers_audio():
    fs = 1e6
    audio = ssb_demod(_tone(5000, fs, 20000), fs, sideband="usb")
    spec = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
    peak = np.fft.rfftfreq(len(audio), 1 / fs)[np.argmax(spec)]
    assert abs(peak - 5000) < 100


def test_ssb_lsb_flips():
    # LSB demod conjugates; a +5kHz USB-style tone becomes -5kHz content,
    # which as real audio still shows 5 kHz magnitude but from the lower side
    fs = 1e6
    out_usb = ssb_demod(_tone(5000, fs, 8000), fs, sideband="usb")
    out_lsb = ssb_demod(_tone(5000, fs, 8000), fs, sideband="lsb")
    # they shouldn't be identical (sideband selection matters)
    assert not np.allclose(out_usb, out_lsb)


def test_ssb_bad_sideband():
    import pytest
    with pytest.raises(ValueError):
        ssb_demod(np.ones(100, dtype=np.complex64), 1e6, sideband="xsb")


def test_bpsk_recovers_bits():
    bits_in = [1, 0, 0, 1, 1, 1, 0, 1]
    spb = 100
    parts = [np.exp(1j * (0 if b else np.pi)) * np.ones(spb) for b in bits_in]
    iq = np.concatenate(parts).astype(np.complex64)
    bits, soft = bpsk_demod(iq)
    rec = [1 if bits[i*spb:(i+1)*spb].mean() > 0.5 else 0
           for i in range(len(bits_in))]
    assert rec == bits_in
