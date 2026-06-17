"""Pulse shaping: the transmit-side counterpart to matched filtering.

A raw digital signal made of hard rectangular symbols has wide spectral
sidelobes -- it splatters energy into neighboring channels. Pulse shaping
replaces each symbol with a smooth pulse that's bandlimited, so the transmitted
signal is spectrally well-behaved. The classic choice is the raised-cosine (or
root-raised-cosine) pulse. This is the inverse-direction partner of the matched
filter used on receive.
"""

from __future__ import annotations

import numpy as np


def rrc_taps(sps, span_symbols=8, beta=0.35):
    """Root-raised-cosine filter taps. OUR code.

    sps:          samples per symbol (the upsampling factor).
    span_symbols: how many symbols wide the pulse is (longer = sharper spectrum).
    beta:         roll-off factor in [0, 1]; larger = more bandwidth, gentler.

    Returns the normalized tap array. Used on BOTH ends: shape on transmit,
    matched-filter with the same taps on receive (root * root = raised cosine,
    the zero-ISI pulse).
    """
    sps = int(sps)
    N = span_symbols * sps
    t = (np.arange(N + 1) - N / 2) / sps   # in symbol periods
    taps = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            taps[i] = 1.0 - beta + 4 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4 * beta)) < 1e-8:
            taps[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            num = (np.sin(np.pi * ti * (1 - beta))
                   + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta)))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            taps[i] = num / den
    taps /= np.sqrt(np.sum(taps ** 2))
    return taps.astype(np.float64)


def upsample(symbols, sps):
    """Insert sps-1 zeros between symbols (zero-stuffing). OUR code.

    The first step of pulse shaping: place each symbol on the output grid, then
    filter to spread it into a pulse. Returns a complex64 array sps times longer.
    """
    symbols = np.asarray(symbols, dtype=np.complex64)
    out = np.zeros(len(symbols) * int(sps), dtype=np.complex64)
    out[::int(sps)] = symbols
    return out


def pulse_shape(symbols, sps, span_symbols=8, beta=0.35):
    """Upsample symbols and shape them with an RRC pulse. OUR code.

    The standard transmit-side digital chain: take complex symbols (e.g. PSK
    constellation points), upsample by sps, and convolve with a root-raised-cosine
    pulse so the result is bandlimited and ISI-free when matched-filtered on
    receive. Returns the shaped complex64 baseband signal.
    """
    taps = rrc_taps(sps, span_symbols, beta)
    up = upsample(symbols, sps)
    shaped = np.convolve(up, taps, mode="same")
    return shaped.astype(np.complex64)
