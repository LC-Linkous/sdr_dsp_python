"""Phase-shift-keying demodulation: BPSK (and coming QPSK/8-PSK/differential).

Coherent demods assume the signal is carrier-aligned; use the recovery
primitives in core.sync first for real captures.
"""

from __future__ import annotations

import numpy as np


def bpsk_demod(iq, normalize_phase=True):
    """Demodulate binary phase-shift keying (coherent-ish). OUR code.

    BPSK encodes bits as 0 or pi phase. With the carrier already at baseband and
    roughly phase-aligned, the sign of the real part recovers the bits. This is
    a SIMPLE demod: it assumes the signal is already carrier-aligned (no Costas
    loop / carrier recovery). For captures with a residual carrier offset,
    correct it first (see estimate_cfo / frequency_shift) -- the library does
    not auto-recover the carrier.

    Returns (bits, soft) where bits is uint8 (0/1) and soft is the real-part
    decision statistic (useful for confidence / plotting a constellation).
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if normalize_phase and len(iq):
        # remove a constant phase offset by aligning the dominant axis to real:
        # rotate so the mean squared phase lands on the real axis.
        rot = np.exp(-1j * 0.5 * np.angle(np.mean(iq ** 2)))
        iq = iq * rot
    soft = np.real(iq).astype(np.float64)
    bits = (soft > 0).astype(np.uint8)
    return bits, soft
