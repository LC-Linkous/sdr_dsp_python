"""Digital modulators: the transmit-side inverses of the digital demods.

Each turns a bit (or symbol) stream into IQ. OOK and FSK are the robust,
constant-or-simple-envelope schemes -- the best first candidates for real
hardware. BPSK/QPSK carry bits in phase and pair with the recovery layer on
receive. All are verified by round-tripping through their demods.
"""

from __future__ import annotations

import numpy as np

from .shaping import pulse_shape


def ook_modulate(bits, samples_per_symbol, high=1.0, low=0.0):
    """On-off keying: bit 1 -> carrier on, bit 0 -> off. Inverse of ook_slice.

    The simplest digital modulation. Each bit becomes samples_per_symbol samples
    at amplitude `high` (for 1) or `low` (for 0). ook_envelope + ook_slice
    recover the bits by thresholding the magnitude.

    bits:               sequence of 0/1.
    samples_per_symbol: samples per bit.

    Returns complex64 baseband (real-valued amplitude, zero phase).
    """
    bits = np.asarray(bits).astype(np.int8)
    sps = int(samples_per_symbol)
    levels = np.where(bits == 1, high, low).astype(np.float64)
    return np.repeat(levels, sps).astype(np.complex64)


def fsk_modulate(bits, samples_per_symbol, deviation_hz, sample_rate):
    """Binary FSK: bit selects one of two frequencies. Inverse of fsk_demod.

    Bit 1 -> +deviation_hz, bit 0 -> -deviation_hz, encoded as a continuous-phase
    frequency shift (CPFSK -- the phase is integrated so there are no jumps,
    which keeps the spectrum clean). fsk_demod recovers bits from the
    instantaneous frequency's sign.

    bits:               sequence of 0/1.
    samples_per_symbol: samples per bit.
    deviation_hz:       frequency shift magnitude (match the demod's threshold).
    sample_rate:        sample rate in Hz.

    Returns unit-magnitude complex64 IQ (FSK is constant-envelope).
    """
    bits = np.asarray(bits).astype(np.int8)
    sps = int(samples_per_symbol)
    # per-sample frequency: +dev for 1, -dev for 0
    freqs = np.where(bits == 1, deviation_hz, -deviation_hz).astype(np.float64)
    inst_freq = np.repeat(freqs, sps)
    # continuous phase = running integral of frequency
    phase = 2 * np.pi * np.cumsum(inst_freq) / float(sample_rate)
    return np.exp(1j * phase).astype(np.complex64)


def bpsk_modulate(bits, samples_per_symbol=1, pulse_shaping=False,
                  beta=0.35, span_symbols=8):
    """BPSK: bit 0 -> +1, bit 1 -> -1 (phase 0 or pi). Inverse of bpsk_demod.

    Carries one bit per symbol in the carrier phase. With pulse_shaping=False the
    symbols are held rectangular (sps samples each); with pulse_shaping=True they
    are RRC-shaped for a bandlimited spectrum (use a matched RRC filter on
    receive). bpsk_demod recovers bits from the real part's sign.

    bits:               sequence of 0/1.
    samples_per_symbol: samples per symbol (1 = one sample per symbol).
    pulse_shaping:      RRC-shape the symbols if True.

    Returns complex64 baseband.
    """
    bits = np.asarray(bits).astype(np.int8)
    # match bpsk_demod's convention: +1 -> bit 1, -1 -> bit 0
    symbols = np.where(bits == 1, 1.0, -1.0).astype(np.complex64)
    return _emit(symbols, samples_per_symbol, pulse_shaping, beta, span_symbols)


def qpsk_modulate(bits, samples_per_symbol=1, pulse_shaping=False,
                  beta=0.35, span_symbols=8, gray=True):
    """QPSK: 2 bits per symbol, Gray-coded quadrants. Inverse of qpsk_demod.

    Pairs of bits map to the four points (1+1j, -1+1j, -1-1j, 1-1j)/sqrt(2) via
    the same Gray convention qpsk_demod uses, so the round-trip is exact. An odd
    trailing bit is dropped (QPSK consumes bits in pairs).

    bits:               sequence of 0/1 (length should be even).
    samples_per_symbol: samples per symbol.
    pulse_shaping:      RRC-shape the symbols if True.
    gray:               use Gray coding (match the demod).

    Returns complex64 baseband.
    """
    bits = np.asarray(bits).astype(np.int8)
    if len(bits) % 2:
        bits = bits[:-1]                      # consume in pairs
    pairs = bits.reshape(-1, 2)
    # Gray-coded map matching qpsk_demod's quadrant convention
    gray_map = {(0, 0): 1 + 1j, (0, 1): -1 + 1j,
                (1, 1): -1 - 1j, (1, 0): 1 - 1j}
    bin_map = {(0, 0): 1 + 1j, (0, 1): -1 + 1j,
               (1, 0): -1 - 1j, (1, 1): 1 - 1j}
    m = gray_map if gray else bin_map
    pts = np.array([m[(int(a), int(b))] for a, b in pairs],
                   dtype=np.complex64) / np.sqrt(2)
    return _emit(pts, samples_per_symbol, pulse_shaping, beta, span_symbols)


def _emit(symbols, sps, pulse_shaping, beta, span_symbols):
    """Render symbols to baseband: rectangular hold or RRC pulse shaping."""
    sps = int(sps)
    if pulse_shaping:
        return pulse_shape(symbols, sps, span_symbols=span_symbols, beta=beta)
    if sps == 1:
        return np.asarray(symbols, dtype=np.complex64)
    return np.repeat(symbols, sps).astype(np.complex64)
