"""Phase primitives: instantaneous phase and frequency.

The phase discriminator under FM and FSK demodulation, exposed so both build on
it and so frequency can be analyzed directly (Doppler, drift, chirps).
"""

from __future__ import annotations

import numpy as np


def instantaneous_phase(iq, unwrap=True):
    """The phase angle of each complex sample. OUR code.

    Returns the per-sample phase in radians. With unwrap=True the 2*pi jumps are
    removed so the phase is continuous (useful for seeing accumulated phase /
    measuring frequency as its slope).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    phase = np.angle(iq).astype(np.float64)
    return np.unwrap(phase) if unwrap else phase


def instantaneous_frequency(iq, sample_rate=None):
    """The instantaneous frequency of a complex signal. OUR code.

    Computed by the phase discriminator: the phase change between consecutive
    samples, angle(x[n] * conj(x[n-1])). This is THE primitive under FM and FSK
    demodulation, exposed so both build on it (and so you can analyze frequency
    directly -- Doppler, drift, chirps).

    Returns radians/sample, or Hz if sample_rate is given. Output length is
    len(iq) - 1 (one difference per adjacent pair).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < 2:
        return np.zeros(0, dtype=np.float64)
    prod = iq[1:] * np.conj(iq[:-1])
    rad_per_sample = np.angle(prod).astype(np.float64)
    if sample_rate is not None:
        return rad_per_sample * float(sample_rate) / (2.0 * np.pi)
    return rad_per_sample


