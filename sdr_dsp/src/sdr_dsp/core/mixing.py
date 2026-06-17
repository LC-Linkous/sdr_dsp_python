"""Frequency translation (mixing): shift a signal in frequency. OUR code.

Multiplying by a complex exponential shifts the spectrum -- the operation at the
heart of tuning and channelization. Trivial to implement, fundamental to own.
"""

from __future__ import annotations

import numpy as np


def frequency_shift(iq, shift_hz, sample_rate):
    """Shift a complex signal up (positive) or down (negative) in frequency.

    Multiplies by exp(j*2*pi*shift*t). To bring a signal at offset f to
    baseband (0 Hz), pass shift_hz = -f.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    t = np.arange(n, dtype=np.float64) / float(sample_rate)
    lo = np.exp(2j * np.pi * float(shift_hz) * t).astype(np.complex64)
    return (iq * lo).astype(np.complex64)


def tune_to_baseband(iq, offset_hz, sample_rate):
    """Bring a signal sitting at +offset_hz down to 0 Hz (DC)."""
    return frequency_shift(iq, -float(offset_hz), sample_rate)


def remove_dc(iq):
    """Remove the DC offset / LO leakage: subtract the complex mean. OUR code.

    Direct-conversion SDRs leak their local oscillator into the band center,
    producing a spurious spike at 0 Hz that isn't a real signal. Subtracting the
    mean removes it. Returns complex64.
    """
    iq = np.asarray(iq)
    if len(iq) == 0:
        return iq.astype(np.complex64)
    return (iq - np.mean(iq)).astype(np.complex64)
