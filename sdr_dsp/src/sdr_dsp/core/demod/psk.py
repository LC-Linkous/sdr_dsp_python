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


def dbpsk_demod(symbols):
    """Demodulate differential BPSK. OUR code.

    Differential PSK encodes bits in phase CHANGES between consecutive symbols,
    not absolute phase. That's the whole point: it needs NO carrier recovery,
    because a constant phase offset cancels when you compare adjacent symbols.
    This makes it robust and a good fit for block processing.

    Takes symbol-spaced samples (one per symbol -- use symbol_sync first if you
    have oversampled data). A bit is 0 if the phase barely changed, 1 if it
    flipped by ~pi. Returns (bits, soft) where soft is the real part of the
    differential product (sign gives the bit, magnitude gives confidence).
    """
    s = np.asarray(symbols, dtype=np.complex64)
    if len(s) < 2:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float64)
    diff = s[1:] * np.conj(s[:-1])         # phase difference between neighbors
    soft = np.real(diff).astype(np.float64)
    # phase unchanged -> bit 0; phase flipped (~pi) -> bit 1
    bits = (soft < 0).astype(np.uint8)
    return bits, soft


def dqpsk_demod(symbols):
    """Demodulate differential QPSK. OUR code.

    The QPSK analogue of DBPSK: 2 bits per symbol encoded in the phase CHANGE
    (one of four ~90-degree steps), so it also needs no carrier recovery. Takes
    symbol-spaced samples; returns (bits, phase_diffs) where bits is a uint8
    array (2 per symbol, MSB first) and phase_diffs are the raw differential
    angles for inspection.
    """
    s = np.asarray(symbols, dtype=np.complex64)
    if len(s) < 2:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float64)
    diff = s[1:] * np.conj(s[:-1])
    ang = np.angle(diff)                    # in (-pi, pi]
    # map the four quadrants of phase change to 2-bit symbols (Gray-coded)
    # 0, +pi/2, pi, -pi/2  ->  00, 01, 11, 10
    quad = np.mod(np.round(ang / (np.pi / 2)).astype(int), 4)
    gray = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    bits = []
    for q in quad:
        b = gray[int(q)]
        bits.extend(b)
    return np.array(bits, dtype=np.uint8), ang.astype(np.float64)


def qpsk_demod(symbols, gray=True):
    """Demodulate QPSK from recovered symbols. OUR code.

    QPSK carries 2 bits/symbol in four phase points (the four quadrants of the
    complex plane). This is a COHERENT demod: it assumes the symbols are already
    carrier-aligned and symbol-timed. For a raw capture, recover first:

        from sdr_dsp.core import carrier_recovery, symbol_sync
        corr = carrier_recovery(iq, method="costas", order=4)
        syms = symbol_sync(corr, sps)
        bits, _ = qpsk_demod(syms)

    The library does NOT auto-recover -- you compose the recovery you want, so
    nothing is hidden. Returns (bits, decisions) where bits is uint8 (2 per
    symbol) and decisions are the constellation points chosen (for plotting).

    gray=True uses Gray coding (adjacent quadrants differ by one bit), the
    standard choice that minimizes bit errors.
    """
    s = np.asarray(symbols, dtype=np.complex64)
    if len(s) == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.complex64)
    # Decide the quadrant, then map to 2 bits. We use a consistent Gray mapping
    # where adjacent quadrants (a 90-degree rotation) differ by exactly one bit:
    #   quadrant 0 (I>0,Q>0) -> 00
    #   quadrant 1 (I<0,Q>0) -> 01
    #   quadrant 2 (I<0,Q<0) -> 11
    #   quadrant 3 (I>0,Q<0) -> 10
    # (This is the standard Gray-coded QPSK constellation.)
    i_pos = np.real(s) >= 0
    q_pos = np.imag(s) >= 0
    quadrant = np.where(i_pos & q_pos, 0,
                        np.where(~i_pos & q_pos, 1,
                                 np.where(~i_pos & ~q_pos, 2, 3)))
    gray = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)} if gray else \
           {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
    bits = []
    for q in quadrant:
        bits.extend(gray[int(q)])
    # the chosen constellation points (unit circle, quadrant centers)
    centers = {0: (1 + 1j), 1: (-1 + 1j), 2: (-1 - 1j), 3: (1 - 1j)}
    decisions = np.array([centers[int(q)] for q in quadrant],
                         dtype=np.complex64) / np.sqrt(2)
    return np.array(bits, dtype=np.uint8), decisions


def psk8_demod(symbols):
    """Demodulate 8-PSK from recovered symbols. OUR code.

    8-PSK carries 3 bits/symbol in eight phase points (45-degree spacing).
    COHERENT: assumes carrier-aligned, symbol-timed input (recover first, as in
    qpsk_demod). Higher-order PSK demands more SNR -- the eight points are
    closer together -- so on an 8-bit SDR like the HackRF this needs a strong,
    clean signal. Returns (bits, sector) where bits is uint8 (3 per symbol) and
    sector is the chosen 0..7 phase sector.

    Honest note: at 45-degree spacing, a small residual carrier error rotates
    points across decision boundaries, so good carrier recovery matters more
    here than for QPSK.
    """
    s = np.asarray(symbols, dtype=np.complex64)
    if len(s) == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.uint8)
    # nearest of 8 phases; map sector -> 3 Gray-coded bits
    ang = np.mod(np.angle(s), 2 * np.pi)
    sector = np.mod(np.round(ang / (np.pi / 4)).astype(int), 8)
    gray3 = {0: (0, 0, 0), 1: (0, 0, 1), 2: (0, 1, 1), 3: (0, 1, 0),
             4: (1, 1, 0), 5: (1, 1, 1), 6: (1, 0, 1), 7: (1, 0, 0)}
    bits = []
    for sec in sector:
        bits.extend(gray3[int(sec)])
    return np.array(bits, dtype=np.uint8), sector.astype(np.uint8)
