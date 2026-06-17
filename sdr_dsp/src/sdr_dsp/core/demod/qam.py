"""Quadrature amplitude modulation (QAM) demodulation.

QAM combines amplitude AND phase to pack more bits per symbol (QAM-16 = 4
bits). It is the most demanding modulation in this library: the 16 points are
close together, so it needs good carrier recovery, good timing recovery,
amplitude normalization, and ideally channel equalization.

HONEST SCOPE: this is DEMONSTRABLE on a clean, strong capture (or synthetic
signal), NOT robust reception. On an 8-bit SDR like the HackRF, with no
equalizer, real over-the-air QAM-16 at low SNR will have errors. We provide it
to show the principle and to work on good captures -- not to claim a QAM modem.
See MODULATIONS.md and HARDWARE.md for the honest status.
"""

from __future__ import annotations

import numpy as np


def qam16_demod(symbols, normalize=True):
    """Demodulate QAM-16 from recovered symbols. OUR code.

    COHERENT and amplitude-sensitive: assumes carrier-aligned, symbol-timed
    input AND a known amplitude scale (QAM decisions depend on absolute level,
    unlike PSK). Recover first, then normalize:

        corr = carrier_recovery(iq, method="costas", order=4)
        syms = symbol_sym(corr, sps)
        bits, pts = qam16_demod(syms)   # normalize=True scales by RMS

    The 16 points sit on a 4x4 grid at I,Q in {-3,-1,+1,+3} (scaled). Each axis
    carries 2 Gray-coded bits. Returns (bits, points) -- 4 bits/symbol and the
    chosen grid points (for plotting the constellation).

    normalize=True scales the input so its RMS matches the standard grid; this
    is the one place QAM needs an amplitude assumption, and it's explicit. Pass
    normalize=False if you've already scaled the signal yourself.
    """
    s = np.asarray(symbols, dtype=np.complex64)
    if len(s) == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.complex64)
    if normalize:
        # standard QAM-16 RMS is sqrt(10) for the {-3,-1,1,3} grid; scale to it
        rms = np.sqrt(np.mean(np.abs(s) ** 2))
        if rms > 0:
            s = s / rms * np.sqrt(10.0)
    # per-axis 4-level decision: thresholds at -2, 0, +2 -> levels -3,-1,1,3
    def level(x):
        lvl = np.where(x < -2, -3, np.where(x < 0, -1, np.where(x < 2, 1, 3)))
        return lvl
    i_lvl = level(np.real(s))
    q_lvl = level(np.imag(s))
    # Gray map for the 4 levels: -3->00, -1->01, +1->11, +3->10
    gray = {-3: (0, 0), -1: (0, 1), 1: (1, 1), 3: (1, 0)}
    bits = []
    for il, ql in zip(i_lvl, q_lvl):
        bits.extend(gray[int(il)])
        bits.extend(gray[int(ql)])
    points = (i_lvl + 1j * q_lvl).astype(np.complex64)
    return np.array(bits, dtype=np.uint8), points
