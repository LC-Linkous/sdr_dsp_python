"""The return type of the symbol-domain demodulators.

The library's demodulators come in two families, and the difference is in
what they consume, not a stylistic accident:

* SAMPLE-domain demods -- fm_demod, am_demod, ssb_demod, fsk_demod, ook_slice,
  nask_slice -- take an IQ *stream* and return a per-sample *stream* (audio,
  an envelope, hard decisions at the sample rate). The caller slices or
  resamples that stream; it is a plain ndarray because it is used like one.
* SYMBOL-domain demods -- bpsk_demod, qpsk_demod, psk8_demod, qam16_demod,
  dbpsk_demod, dqpsk_demod -- take recovered *symbols* (one complex value per
  symbol, after carrier and timing recovery) and return a `DemodResult`: the
  hard bits and the soft decision information that produced them.

`DemodResult` is a NamedTuple, so ``bits, soft = qpsk_demod(symbols)`` keeps
working, and so does ``qpsk_demod(symbols).bits``. What it is NOT is a bits
array: ``find_frames(qpsk_demod(symbols))`` is a type error, where it used to
return [] in silence because the 2-tuple happened to be iterable.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class DemodResult(NamedTuple):
    """Hard bits plus the soft information behind them.

    bits: uint8 array of 0/1, k bits per symbol for a 2^k-ary scheme.
    soft: per-symbol decision information; its meaning depends on the demod
          (bpsk/dbpsk: the real-axis statistic, float64; qpsk/qam16: the
          chosen constellation points, complex64; psk8: the sector index,
          uint8; dqpsk: the differential angle, float64). Confidence,
          constellation plots and EVM all start from here.
    """

    bits: np.ndarray
    soft: np.ndarray
