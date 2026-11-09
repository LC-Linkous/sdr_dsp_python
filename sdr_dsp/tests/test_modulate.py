"""Tests for the modulators (Phase A of TX).

The defining discipline: each modulator is the inverse of a demod, so the test
is the round-trip -- demod(modulate(x)) recovers x. Analog round-trips are
checked by correlation; digital ones are bit-exact (BER 0). Pulse shaping is
verified by the RRC matched-filter round-trip (root*root = zero-ISI).
"""

import numpy as np
import pytest

from sdr_dsp.core import (
    fm_modulate, am_modulate, ssb_modulate,
    ook_modulate, fsk_modulate, bpsk_modulate, qpsk_modulate,
    rrc_taps, upsample,
    fm_demod, am_demod, ssb_demod, ook_envelope, ook_slice,
    fsk_demod, bpsk_demod, qpsk_demod,
)


def _corr(a, b, trim=200):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return np.corrcoef(a[trim:-trim], b[trim:-trim])[0, 1]


# -- analog round-trips -----------------------------------------------------

def test_fm_roundtrip():
    fs = 1e6
    msg = np.cos(2 * np.pi * 2000 * np.arange(20000) / fs)
    rec = fm_demod(fm_modulate(msg, fs, 75e3), fs, 75e3)
    assert _corr(msg, rec) > 0.99


def test_fm_constant_envelope():
    fs = 1e6
    msg = np.cos(2 * np.pi * 2000 * np.arange(5000) / fs)
    iq = fm_modulate(msg, fs, 75e3)
    assert np.allclose(np.abs(iq), 1.0, atol=1e-5)   # FM is constant-envelope


def test_am_roundtrip():
    fs = 1e6
    msg = np.cos(2 * np.pi * 2000 * np.arange(20000) / fs)
    rec = am_demod(am_modulate(msg, 0.5))
    assert _corr(msg, rec) > 0.99


def test_am_message_in_envelope():
    msg = np.cos(2 * np.pi * 0.01 * np.arange(1000))
    iq = am_modulate(msg, 0.5)
    # the envelope should be 1 + 0.5*msg
    assert np.allclose(np.abs(iq), np.abs(1 + 0.5 * msg), atol=1e-5)


def test_ssb_usb_roundtrip():
    fs = 1e6
    msg = np.cos(2 * np.pi * 2000 * np.arange(20000) / fs)
    rec = ssb_demod(ssb_modulate(msg, "usb"), fs, "usb")
    assert _corr(msg, rec) > 0.95


@pytest.mark.parametrize("n", [8, 9, 4096, 4097])
def test_ssb_modulate_is_the_analytic_signal(n):
    # ssb_modulate builds the analytic signal (message + j*Hilbert(message)),
    # so: a constant message must come back at 1x, not 2x (the DC bin was
    # being doubled along with the positive frequencies); a cosine must come
    # back as exp(j*w*t) -- real part the message, |iq| constant, and the
    # imaginary part the quadrature sine. Even and odd n cover both Nyquist
    # handling branches.
    const = ssb_modulate(np.ones(n), "usb")
    assert np.allclose(const.real, 1.0, atol=1e-5)
    assert np.allclose(const.imag, 0.0, atol=1e-5)

    t = np.arange(n)
    k = max(1, n // 8)                          # whole cycles, below Nyquist
    msg = np.cos(2 * np.pi * k * t / n)
    iq = ssb_modulate(msg, "usb")
    assert np.allclose(iq.real, msg, atol=1e-5)
    assert np.allclose(iq.imag, np.sin(2 * np.pi * k * t / n), atol=1e-5)
    assert np.allclose(np.abs(iq), 1.0, atol=1e-5)
    # lsb is the conjugate: same real part, negated quadrature
    lsb = ssb_modulate(msg, "lsb")
    assert np.allclose(lsb, np.conj(iq), atol=1e-6)


def test_ssb_invalid_sideband():
    with pytest.raises(ValueError):
        ssb_modulate(np.ones(10), "bogus")


# -- digital round-trips (bit-exact) ---------------------------------------

@pytest.fixture
def bits():
    return np.random.default_rng(0).integers(0, 2, 200)


def test_ook_roundtrip(bits):
    sps = 50
    rec = ook_slice(ook_envelope(ook_modulate(bits, sps)))
    rec_bits = rec[::sps][:len(bits)]
    assert np.mean(rec_bits != bits) == 0


def test_fsk_roundtrip(bits):
    fs = 1e6
    sps = 50
    iq = fsk_modulate(bits, sps, 50e3, fs)
    rec = fsk_demod(iq, fs)
    rec_bits = rec[sps // 2::sps][:len(bits)]   # sample mid-symbol
    assert np.mean(rec_bits != bits) == 0


def test_fsk_constant_envelope(bits):
    iq = fsk_modulate(bits, 20, 50e3, 1e6)
    assert np.allclose(np.abs(iq), 1.0, atol=1e-5)


def test_bpsk_roundtrip(bits):
    rec, _ = bpsk_demod(bpsk_modulate(bits, 1))
    assert np.mean(rec[:len(bits)] != bits) == 0


def test_bpsk_convention_matches_demod():
    # +1 must mean bit 1 (the demod's convention)
    iq = bpsk_modulate(np.array([1, 0]), 1)
    assert np.real(iq[0]) > 0 and np.real(iq[1]) < 0


def test_qpsk_roundtrip(bits):
    rec, _ = qpsk_demod(qpsk_modulate(bits, 1))
    assert np.mean(rec[:len(bits)] != bits) == 0


def test_qpsk_refuses_an_odd_bit_count():
    # it used to drop the trailing bit silently -- data the caller thought it sent
    with pytest.raises(ValueError, match="even number of bits"):
        qpsk_modulate(np.array([1, 0, 1]), 1)
    assert len(qpsk_modulate(np.array([1, 0, 1, 1]), 1)) == 2


# -- pulse shaping ----------------------------------------------------------

def test_rrc_taps_normalized():
    taps = rrc_taps(8)
    assert np.sum(taps ** 2) == pytest.approx(1.0, abs=1e-6)


def test_upsample_inserts_zeros():
    up = upsample(np.array([1, 2, 3], dtype=np.complex64), 4)
    assert len(up) == 12
    assert up[0] == 1 and up[4] == 2 and up[8] == 3
    assert up[1] == 0 and up[2] == 0


def test_pulse_shaped_bpsk_roundtrip(bits):
    # the RRC matched-filter round-trip: root*root = raised cosine, zero ISI
    sps = 8
    iq = bpsk_modulate(bits, sps, pulse_shaping=True)
    taps = rrc_taps(sps)
    matched = np.convolve(iq, taps, mode="same")
    syms = matched[::sps][:len(bits)]
    rec, _ = bpsk_demod(syms)
    assert np.mean(rec[:len(bits)] != bits) == 0


def test_pulse_shaped_is_bandlimited(bits):
    # shaped signal should have less high-frequency energy than rectangular
    sps = 8
    rect = bpsk_modulate(bits, sps, pulse_shaping=False)
    shaped = bpsk_modulate(bits, sps, pulse_shaping=True)
    # high-freq energy = energy above half-Nyquist
    def hf_frac(x):
        s = np.abs(np.fft.fft(x))
        return np.sum(s[len(s)//4:3*len(s)//4]) / np.sum(s)
    assert hf_frac(shaped) < hf_frac(rect)


def test_empty_inputs():
    assert len(fm_modulate(np.zeros(0), 1e6, 1e3)) == 0
    assert len(ook_modulate(np.zeros(0, dtype=int), 10)) == 0
    assert len(ssb_modulate(np.zeros(0))) == 0
