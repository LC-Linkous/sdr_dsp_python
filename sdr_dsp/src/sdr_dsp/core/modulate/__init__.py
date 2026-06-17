"""Modulators: the transmit-side inverses of the demods.

Turn messages, bits, and symbols into IQ ready to transmit. Each modulator is
the inverse of a demod in core/demod/, and is tested by round-tripping:
demod(modulate(x)) == x. Mirrors the structure of the demod package.
"""

from .analog import fm_modulate, am_modulate, ssb_modulate
from .digital import (ook_modulate, fsk_modulate, bpsk_modulate,
                      qpsk_modulate)
from .shaping import rrc_taps, upsample, pulse_shape

__all__ = [
    "fm_modulate", "am_modulate", "ssb_modulate",
    "ook_modulate", "fsk_modulate", "bpsk_modulate", "qpsk_modulate",
    "rrc_taps", "upsample", "pulse_shape",
]
