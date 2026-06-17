"""Demodulation package: analog and digital demods, organized by family.

Split into submodules by modulation family (analog, ask, fsk, psk) plus the
shared phase primitives and the simple timing helpers. Everything is re-exported
here so existing imports (`from sdr_dsp.core import fm_demod`) keep working --
the split is structural, not a behavior or API change.

The closed-loop recovery primitives (carrier_recovery, symbol_sync) live in
core.sync, not here, because they are reusable across demods.
"""

from .phase import instantaneous_phase, instantaneous_frequency
from .analog import fm_demod, am_demod, ssb_demod, deemphasis
from .ask import ook_envelope, ook_slice
from .fsk import fsk_demod
from .psk import bpsk_demod
from .timing import edges, estimate_symbol_rate, slice_to_symbols

__all__ = [
    "instantaneous_phase", "instantaneous_frequency",
    "fm_demod", "am_demod", "ssb_demod", "deemphasis",
    "ook_envelope", "ook_slice",
    "fsk_demod",
    "bpsk_demod",
    "edges", "estimate_symbol_rate", "slice_to_symbols",
]
