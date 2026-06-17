"""Symbol-timing helpers: edge/run detection and crude symbol-rate recovery.

These are the simple, deterministic timing tools (distinct from the closed-loop
recovery in core.sync). Good for clean bursts where pulse widths are reliable.
"""

from __future__ import annotations

import numpy as np


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


