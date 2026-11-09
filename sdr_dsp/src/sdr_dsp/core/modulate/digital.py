"""Digital modulators: the transmit-side inverses of the digital demods.

Each turns a bit (or symbol) stream into IQ. OOK and FSK are the robust,
constant-or-simple-envelope schemes -- the best first candidates for real
hardware. BPSK/QPSK carry bits in phase and pair with the recovery layer on
receive. All are verified by round-tripping through their demods.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .shaping import pulse_shape
from .._validate import bits as _bits, positive, positive_int


def ook_modulate(
    bits: Any,
    samples_per_symbol: int,
    high: float = 1.0,
    low: float = 0.0,
    pad_symbols: int = 0,
) -> np.ndarray:
    """On-off keying: bit 1 -> carrier on, bit 0 -> off. Inverse of ook_slice.

    The simplest digital modulation. Each bit becomes samples_per_symbol samples
    at amplitude `high` (for 1) or `low` (for 0). ook_envelope + ook_slice
    recover the bits by thresholding the magnitude.

    bits:               sequence of 0/1.
    samples_per_symbol: samples per bit.
    pad_symbols:        symbols of silence appended to EACH end of the burst
                        (default 0: exact legacy output). The first/last
                        symbols of an unpadded burst sit flush against the
                        buffer edges, where instantaneous-frequency and filter
                        edge effects corrupt them -- a loopback at zero delay
                        hides this; any real capture (arbitrary delay, burst
                        embedded in noise) exposes it. Use >= 4 for anything
                        leaving a same-buffer loopback.

    Returns complex64 baseband (real-valued amplitude, zero phase).
    """
    bits = _bits("bits", bits)
    sps = positive_int("samples_per_symbol", samples_per_symbol)
    levels = np.where(bits == 1, high, low).astype(np.float64)
    iq = np.repeat(levels, sps).astype(np.complex64)
    return _pad(iq, pad_symbols, sps)


def fsk_modulate(
    bits: Any,
    samples_per_symbol: int,
    deviation_hz: float,
    sample_rate: float,
    pad_symbols: int = 0,
) -> np.ndarray:
    """Binary FSK: bit selects one of two frequencies. Inverse of fsk_demod.

    Bit 1 -> +deviation_hz, bit 0 -> -deviation_hz, encoded as a continuous-phase
    frequency shift (CPFSK -- the phase is integrated so there are no jumps,
    which keeps the spectrum clean). fsk_demod recovers bits from the
    instantaneous frequency's sign.

    bits:               sequence of 0/1.
    samples_per_symbol: samples per bit.
    deviation_hz:       frequency shift magnitude (match the demod's threshold).
    sample_rate:        sample rate in Hz.
    pad_symbols:        symbols of silence appended to EACH end of the burst
                        (default 0: exact legacy output). The first/last
                        symbols of an unpadded burst sit flush against the
                        buffer edges, where instantaneous-frequency and filter
                        edge effects corrupt them -- a loopback at zero delay
                        hides this; any real capture (arbitrary delay, burst
                        embedded in noise) exposes it. Use >= 4 for anything
                        leaving a same-buffer loopback.

    Returns complex64 IQ, unit magnitude within the burst (FSK is
    constant-envelope); the pad regions are silence.
    """
    bits = _bits("bits", bits)
    sps = positive_int("samples_per_symbol", samples_per_symbol)
    positive("deviation_hz", deviation_hz)
    positive("sample_rate", sample_rate)
    # per-sample frequency: +dev for 1, -dev for 0
    freqs = np.where(bits == 1, deviation_hz, -deviation_hz).astype(np.float64)
    inst_freq = np.repeat(freqs, sps)
    # continuous phase = running integral of frequency
    phase = 2 * np.pi * np.cumsum(inst_freq) / float(sample_rate)
    iq = np.exp(1j * phase).astype(np.complex64)
    return _pad(iq, pad_symbols, sps)


def bpsk_modulate(
    bits: Any,
    samples_per_symbol: int = 1,
    pulse_shaping: bool = False,
    beta: float = 0.35,
    span_symbols: int = 8,
    pad_symbols: int = 0,
) -> np.ndarray:
    """BPSK: bit 1 -> +1, bit 0 -> -1 (phase 0 or pi). Inverse of bpsk_demod.

    Carries one bit per symbol in the carrier phase. With pulse_shaping=False the
    symbols are held rectangular (sps samples each); with pulse_shaping=True they
    are RRC-shaped for a bandlimited spectrum (use a matched RRC filter on
    receive). bpsk_demod recovers bits from the real part's sign: positive is
    a 1, negative a 0 -- the mapping above is that convention, written out.

    bits:               sequence of 0/1.
    samples_per_symbol: samples per symbol (1 = one sample per symbol).
    pulse_shaping:      RRC-shape the symbols if True.

    Returns complex64 baseband.
    """
    bits = _bits("bits", bits)
    positive_int("samples_per_symbol", samples_per_symbol)
    # match bpsk_demod's convention: +1 -> bit 1, -1 -> bit 0
    symbols = np.where(bits == 1, 1.0, -1.0).astype(np.complex64)
    iq = _emit(symbols, samples_per_symbol, pulse_shaping, beta, span_symbols)
    return _pad(iq, pad_symbols, samples_per_symbol)


def qpsk_modulate(
    bits: Any,
    samples_per_symbol: int = 1,
    pulse_shaping: bool = False,
    beta: float = 0.35,
    span_symbols: int = 8,
    gray: bool = True,
    pad_symbols: int = 0,
) -> np.ndarray:
    """QPSK: 2 bits per symbol, Gray-coded quadrants. Inverse of qpsk_demod.

    Pairs of bits map to the four points (1+1j, -1+1j, -1-1j, 1-1j)/sqrt(2) via
    the same Gray convention qpsk_demod uses, so the round-trip is exact. The
    bit count must be even (QPSK consumes bits in pairs); an odd count is a
    ValueError rather than a silently dropped bit.

    bits:               sequence of 0/1 (length must be even).
    samples_per_symbol: samples per symbol.
    pulse_shaping:      RRC-shape the symbols if True.
    gray:               use Gray coding (match the demod).

    Returns complex64 baseband.
    """
    bits = _bits("bits", bits)
    positive_int("samples_per_symbol", samples_per_symbol)
    if len(bits) % 2:
        # QPSK consumes bits in pairs. Dropping the odd one silently (as an
        # earlier version did) loses data the caller thinks was sent.
        raise ValueError(f"qpsk_modulate needs an even number of bits, got {len(bits)}; "
                         "pad with a 0 or drop one yourself")
    pairs = bits.reshape(-1, 2)
    # Gray-coded map matching qpsk_demod's quadrant convention
    gray_map = {(0, 0): 1 + 1j, (0, 1): -1 + 1j,
                (1, 1): -1 - 1j, (1, 0): 1 - 1j}
    bin_map = {(0, 0): 1 + 1j, (0, 1): -1 + 1j,
               (1, 0): -1 - 1j, (1, 1): 1 - 1j}
    m = gray_map if gray else bin_map
    pts = np.array([m[(int(a), int(b))] for a, b in pairs],
                   dtype=np.complex64) / np.sqrt(2)
    iq = _emit(pts, samples_per_symbol, pulse_shaping, beta, span_symbols)
    return _pad(iq, pad_symbols, samples_per_symbol)


def _emit(
    symbols: np.ndarray, sps: int, pulse_shaping: bool, beta: float, span_symbols: int
) -> np.ndarray:
    """Render symbols to baseband: rectangular hold or RRC pulse shaping."""
    sps = int(sps)
    if pulse_shaping:
        return pulse_shape(symbols, sps, span_symbols=span_symbols, beta=beta)
    if sps == 1:
        return np.asarray(symbols, dtype=np.complex64)
    return np.repeat(symbols, sps).astype(np.complex64)


def _pad(iq: np.ndarray, pad_symbols: int, sps: int) -> np.ndarray:
    """Wrap IQ in pad_symbols of silence per side (see pad_symbols docs)."""
    n = int(pad_symbols) * int(sps)
    if n <= 0:
        return iq
    z = np.zeros(n, dtype=np.complex64)
    return np.concatenate([z, iq, z])
