"""Demodulation package: analog and digital demods, organized by family.

Split into submodules by modulation family (analog, ask, fsk, psk) plus the
shared phase primitives and the simple timing helpers. Everything is re-exported
here so existing imports (`from sdr_dsp.core import fm_demod`) keep working --
the split is structural, not a behavior or API change.

The closed-loop recovery primitives (carrier_recovery, symbol_sync) live in
core.sync, not here, because they are reusable across demods.
"""

from .phase import instantaneous_phase, instantaneous_frequency
from .analog import (fm_demod, am_demod, ssb_demod, deemphasis,
                     dsb_sc_demod, cw_decode)
from .ask import ook_envelope, ook_slice, nask_slice
from .fsk import fsk_demod, fsk_demod_nlevel
from .psk import (bpsk_demod, dbpsk_demod, dqpsk_demod,
                  qpsk_demod, psk8_demod)
from .timing import edges, estimate_symbol_rate, slice_to_symbols

__all__ = [
    "instantaneous_phase", "instantaneous_frequency",
    "fm_demod", "am_demod", "ssb_demod", "deemphasis",
    "dsb_sc_demod", "cw_decode",
    "ook_envelope", "ook_slice", "nask_slice",
    "fsk_demod", "fsk_demod_nlevel",
    "bpsk_demod", "dbpsk_demod", "dqpsk_demod",
    "qpsk_demod", "psk8_demod",
    "edges", "estimate_symbol_rate", "slice_to_symbols",
]
