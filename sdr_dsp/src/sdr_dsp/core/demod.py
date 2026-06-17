"""Demodulation: FM, AM, OOK/ASK. Entirely sdr_dsp's own code.

scipy has none of this -- demodulation is the radio logic the library exists to
provide. Each function takes baseband complex64 IQ and returns the recovered
signal (audio for FM/AM, a bit/envelope stream for OOK).
"""

from __future__ import annotations

import numpy as np


def fm_demod(iq, deviation_hz=None, sample_rate=None):
    """Demodulate frequency modulation via the phase discriminator. OUR code.

    The instantaneous frequency is the derivative of phase. The standard,
    efficient discriminator computes the phase difference between consecutive
    samples as angle(x[n] * conj(x[n-1])). Returns a real audio-rate-ish array.

    If deviation_hz and sample_rate are given, the output is scaled to
    approximate normalized audio; otherwise it returns raw radians/sample.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    if len(iq) < 2:
        return np.zeros(0, dtype=np.float64)
    # angle of the product of each sample with the conjugate of the previous
    prod = iq[1:] * np.conj(iq[:-1])
    demod = np.angle(prod).astype(np.float64)  # radians/sample
    if deviation_hz and sample_rate:
        # radians/sample -> Hz -> normalized by peak deviation
        inst_hz = demod * sample_rate / (2.0 * np.pi)
        demod = inst_hz / float(deviation_hz)
    return demod


def am_demod(iq, dc_block=True):
    """Demodulate amplitude modulation: the envelope (magnitude). OUR code.

    Returns the real envelope |iq|. With dc_block, the mean (carrier DC) is
    removed so the output swings around zero like audio.
    """
    iq = np.asarray(iq)
    env = np.abs(iq).astype(np.float64)
    if dc_block:
        env = env - np.mean(env)
    return env


def ook_envelope(iq):
    """On-off-keying / ASK front end: the magnitude envelope. OUR code.

    Returns |iq| (no DC block -- OOK threshold detection wants the absolute
    level). Feed to ``ook_slice`` to recover bits.
    """
    return np.abs(np.asarray(iq)).astype(np.float64)


def ook_slice(envelope, threshold=None):
    """Threshold an OOK envelope into a 0/1 stream. OUR code.

    threshold: level above which a sample is '1'. If None, uses the midpoint
    between the envelope's min and max (a simple, robust default for a clean
    capture). Returns a uint8 array of 0/1.
    """
    env = np.asarray(envelope, dtype=np.float64)
    if threshold is None:
        threshold = (env.min() + env.max()) / 2.0
    return (env > threshold).astype(np.uint8)


def edges(bits):
    """Indices where a 0/1 stream transitions, and the run lengths. OUR code.

    Returns (transition_indices, run_lengths, run_values): the sample index of
    each transition, how many samples each run lasted, and whether that run was
    0 or 1. The building block for recovering symbol timing from a sliced
    on/off stream.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) == 0:
        return np.array([], dtype=int), np.array([], dtype=int), \
            np.array([], dtype=np.uint8)
    change = np.nonzero(np.diff(bits))[0] + 1
    bounds = np.concatenate([[0], change, [len(bits)]])
    run_lengths = np.diff(bounds)
    run_values = bits[bounds[:-1]]
    return change, run_lengths, run_values


def estimate_symbol_rate(bits, sample_rate):
    """Estimate samples-per-symbol from the shortest run in a sliced stream.

    The shortest on/off run is (usually) one symbol period. OUR code -- a
    robust-enough heuristic for clean OOK/ASK bursts. Returns
    (samples_per_symbol, symbol_rate_hz). Use the shortest run as the unit and
    round longer runs to multiples of it.
    """
    _, run_lengths, _ = edges(bits)
    # ignore the leading/trailing idle runs (often huge) by taking the modal
    # short run: the minimum of the interior runs is the symbol period.
    interior = run_lengths[1:-1] if len(run_lengths) > 2 else run_lengths
    if len(interior) == 0:
        return 0.0, 0.0
    spb = float(np.min(interior))
    return spb, (sample_rate / spb if spb else 0.0)


def slice_to_symbols(bits, samples_per_symbol):
    """Collapse an over-sampled 0/1 stream into one bit per symbol. OUR code.

    Given the samples-per-symbol, walk each run and emit its value repeated
    round(run_length / spb) times. Returns a uint8 array of symbol bits.
    """
    _, run_lengths, run_values = edges(bits)
    spb = float(samples_per_symbol)
    if spb <= 0:
        return np.asarray(bits, dtype=np.uint8)
    out = []
    for length, val in zip(run_lengths, run_values):
        count = max(1, int(round(length / spb)))
        out.extend([int(val)] * count)
    return np.array(out, dtype=np.uint8)
