"""Analog modulators: the transmit-side inverses of the analog demods.

Each turns a real message signal into complex IQ ready to transmit. They are the
mathematical inverses of the demods in core/demod/analog.py, which is also how
they're tested: demod(modulate(msg)) recovers msg.
"""

from __future__ import annotations

import numpy as np


def fm_modulate(message, deviation_hz, sample_rate):
    """FM-modulate a real message into IQ. Inverse of fm_demod. OUR code.

    Frequency modulation encodes the message in the carrier's instantaneous
    frequency: the IQ phase is the running integral of the message, scaled by
    the deviation. fm_demod recovers it by differentiating the phase.

    message:      real-valued message, expected roughly in [-1, 1].
    deviation_hz: peak frequency deviation (must match the demod's deviation_hz).
    sample_rate:  sample rate in Hz.

    Returns unit-magnitude complex64 IQ (FM is constant-envelope).
    """
    m = np.asarray(message, dtype=np.float64)
    # phase is 2*pi*deviation * integral(message) / fs
    phase = 2 * np.pi * deviation_hz * np.cumsum(m) / float(sample_rate)
    return np.exp(1j * phase).astype(np.complex64)


def am_modulate(message, modulation_index=0.5):
    """AM-modulate a real message into IQ. Inverse of am_demod. OUR code.

    Amplitude modulation rides the message on the carrier envelope:
    (1 + k*m) carrier. am_demod recovers it with an envelope detector. Keep
    modulation_index <= 1 to avoid over-modulation (envelope going negative,
    which the envelope detector can't undo).

    message:          real message, roughly [-1, 1].
    modulation_index: depth k of modulation (0..1). >1 over-modulates.

    Returns complex64 IQ with the message in its magnitude.
    """
    m = np.asarray(message, dtype=np.float64)
    envelope = 1.0 + float(modulation_index) * m
    return envelope.astype(np.complex64)   # baseband carrier = constant 1


def ssb_modulate(message, sideband="usb"):
    """SSB-modulate a real message into IQ. Inverse of ssb_demod. OUR code.

    Single-sideband keeps one sideband of the message's analytic signal and
    suppresses the carrier and the other sideband. We build the analytic signal
    (positive frequencies only) for USB; conjugate for LSB. ssb_demod recovers
    the real message by selecting the matching sideband.

    message:  real message.
    sideband: "usb" (upper) or "lsb" (lower).

    Returns complex64 IQ -- the single-sideband analytic signal.
    """
    if sideband not in ("usb", "lsb"):
        raise ValueError("sideband must be 'usb' or 'lsb'")
    m = np.asarray(message, dtype=np.float64)
    n = len(m)
    if n == 0:
        return np.zeros(0, dtype=np.complex64)
    # analytic signal: zero the negative frequencies, double the positive ones
    spec = np.fft.fft(m)
    spec[n // 2:] = 0
    analytic = 2 * np.fft.ifft(spec)
    iq = analytic.astype(np.complex64)
    if sideband == "lsb":
        iq = np.conj(iq)
    return iq.astype(np.complex64)
