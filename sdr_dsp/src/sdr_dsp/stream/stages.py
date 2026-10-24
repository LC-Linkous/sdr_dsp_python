"""Stateful streaming stages: the pure DSP core, made block-continuous.

A block boundary is not a signal boundary. The functions in ``sdr_dsp.core``
are stateless on purpose (arrays in, arrays out, easy to verify), but calling
them once per block in a live chain restarts every delay line at every block:
a FIR ramps up from empty taps, a mixer's local oscillator jumps back to phase
zero, the FM discriminator differences the first sample against nothing, and a
resampler zero-pads both ends. The result is a click, a phase glitch, or a
spectral splatter at every seam -- at 50 k-sample blocks, several times a
second.

Each class here wraps one core function and carries exactly the state that
function needs between calls, so a stream processed block by block produces
the same samples as the whole record processed at once (the tests pin this to
float precision). They are plain callables, so they drop straight into a
``Pipeline``::

    chain = (Pipeline(source)
             .add(StreamingFIR(taps), "channel")
             .add(StreamingDecimator(8), "decimate")
             .add(StreamingFMDemod(75e3, fs / 8), "fm_demod")
             .add(StreamingResampler(48_000, int(fs / 8)), "audio"))

Every stage has ``reset()`` (forget history, e.g. on retune). The stages that
hold samples back for look-ahead (``StreamingResampler``) also have
``flush()`` to drain what remains when the stream ends.

The whole-record functions remain the reference: a stage must never implement
DSP of its own, only bookkeeping around a call into ``sdr_dsp.core``.
"""

from __future__ import annotations

from math import gcd

import numpy as np

from ..core.filters import fir_apply
from ..core.demod.phase import instantaneous_frequency
from ..core.demod.analog import deemphasis
from ..core.resample import resample_poly, _antialias_taps

__all__ = [
    "StreamingFIR",
    "StreamingDecimator",
    "StreamingMixer",
    "StreamingFMDemod",
    "StreamingResampler",
    "StreamingDeemphasis",
]


def _roll_tail(tail: np.ndarray, new: np.ndarray) -> np.ndarray:
    """The last len(tail) samples of (tail + new)."""
    n = len(tail)
    if n == 0:
        return tail
    if new.size >= n:
        return new[-n:].copy()
    return np.concatenate([tail[new.size:], new])


class StreamingFIR:
    """``fir_apply`` with its delay line carried across blocks (overlap-save).

    Output for a stream of blocks is identical to ``fir_apply`` on the
    concatenated stream: the previous block's last ``len(taps) - 1`` samples
    are prepended, filtered, and discarded, which is exactly the history a
    causal FIR has at that point.
    """

    def __init__(self, taps):
        self.taps = np.asarray(taps, dtype=np.float64)
        if self.taps.ndim != 1 or self.taps.size == 0:
            raise ValueError("taps must be a non-empty 1-D array")
        self._tail: np.ndarray | None = None

    def reset(self) -> None:
        """Forget all carried history, e.g. after a retune or a stream restart."""
        self._tail = None

    def __call__(self, block):
        block = np.asarray(block)
        m = len(self.taps) - 1
        if self._tail is None or self._tail.dtype != block.dtype:
            self._tail = np.zeros(m, dtype=block.dtype)
        if block.size == 0:
            return block
        out = fir_apply(np.concatenate([self._tail, block]), self.taps)[m:]
        self._tail = _roll_tail(self._tail, block)
        return out


class StreamingDecimator:
    """Keep every ``factor``-th sample, with the sampling phase carried.

    Plain ``block[::factor]`` restarts the phase at each block, so unless every
    block length is a multiple of ``factor`` the output has a different
    (and drifting) sample grid from ``whole[::factor]``. This remembers how
    many samples into the stride the last block ended. No filtering -- put a
    ``StreamingFIR`` (or use ``StreamingResampler``) in front of it.
    """

    def __init__(self, factor):
        self.factor = int(factor)
        if self.factor < 1:
            raise ValueError("factor must be >= 1")
        self._skip = 0           # samples to drop at the start of the next block

    def reset(self) -> None:
        """Forget all carried history, e.g. after a retune or a stream restart."""
        self._skip = 0

    def __call__(self, block):
        block = np.asarray(block)
        if block.size <= self._skip:
            self._skip -= block.size
            return block[:0]
        out = block[self._skip::self.factor]
        # index (relative to this block) of the sample after the last kept one
        last_kept = self._skip + (len(out) - 1) * self.factor
        self._skip = last_kept + self.factor - block.size
        return out


class StreamingMixer:
    """``frequency_shift`` with a phase-continuous local oscillator.

    ``frequency_shift`` builds its LO from ``t = arange(n) / fs``, so every
    block starts the oscillator at phase 0 -- a phase step at every seam that
    smears energy across the band. This tracks the absolute sample index and
    continues the LO from where the previous block left it. Output matches
    ``frequency_shift`` on the whole record.
    """

    def __init__(self, shift_hz, sample_rate):
        self.shift_hz = float(shift_hz)
        self.sample_rate = float(sample_rate)
        self._n0 = 0

    def reset(self) -> None:
        """Forget all carried history, e.g. after a retune or a stream restart."""
        self._n0 = 0

    def __call__(self, block):
        block = np.asarray(block, dtype=np.complex64)
        n = block.size
        if n == 0:
            return block
        # keep the phase argument bounded: the LO is periodic in fs/shift
        # samples (when that ratio is rational) and in any case exp() of a
        # huge argument loses precision, so reduce the index modulo the
        # period where one exists.
        idx = np.arange(self._n0, self._n0 + n, dtype=np.float64)
        phase = 2.0 * np.pi * self.shift_hz * idx / self.sample_rate
        lo = np.exp(1j * np.mod(phase, 2.0 * np.pi)).astype(np.complex64)
        self._n0 += n
        return (block * lo).astype(np.complex64)


class StreamingFMDemod:
    """``fm_demod`` with the previous sample carried.

    The discriminator differences consecutive phases, so the first output of a
    block needs the last input of the block before it. The very first block
    has no predecessor and yields ``len(block) - 1`` samples; every later block
    yields ``len(block)``, so the concatenated stream equals ``fm_demod`` on the
    whole record (which is one shorter than its input).

    ``deviation_hz`` and ``sample_rate`` follow ``fm_demod``: both given scales
    to roughly normalized audio, otherwise raw radians/sample.
    """

    def __init__(self, deviation_hz=None, sample_rate=None):
        self.deviation_hz = deviation_hz
        self.sample_rate = sample_rate
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        """Forget all carried history, e.g. after a retune or a stream restart."""
        self._prev = None

    def __call__(self, block):
        block = np.asarray(block, dtype=np.complex64)
        if block.size == 0:
            return np.zeros(0, dtype=np.float64)
        primed = block if self._prev is None else np.concatenate([self._prev, block])
        self._prev = block[-1:].copy()
        if primed.size < 2:
            return np.zeros(0, dtype=np.float64)
        if self.deviation_hz and self.sample_rate:
            return (instantaneous_frequency(primed, sample_rate=self.sample_rate)
                    / float(self.deviation_hz))
        return instantaneous_frequency(primed)


class StreamingResampler:
    """``resample_poly`` made block-continuous.

    ``resample_poly`` is zero-phase: it compensates the anti-alias filter's
    group delay, so an output sample depends on input both behind and AHEAD
    of it. A streaming version therefore needs history on both sides. Past
    context is carried from the previous block; future context is obtained by
    holding the newest samples back until the next block arrives, instead of
    letting the filter see zeros there. The price is ``latency_samples`` of
    input delay (a fraction of a millisecond at audio rates); the payoff is
    output identical to resampling the whole stream at once, apart from the
    held-back tail, which ``flush()`` emits when the stream ends.

    ``up``/``down`` are the output/input rates (any common factor is removed).
    """

    def __init__(self, up, down, half_len=10, window="hamming"):
        up, down = int(up), int(down)
        if up < 1 or down < 1:
            raise ValueError("up and down must be >= 1")
        g = gcd(up, down)
        self.up, self.down = up // g, down // g
        self.half_len, self.window = half_len, window
        # resample_poly's filter spans ntaps samples at the UPSAMPLED rate;
        # that is `need` input samples. Round the context up to a whole number
        # of `down` so the output accounting is exact rather than drifting on
        # a rounding.
        if self.up == 1 and self.down == 1:
            self.hist = self.ahead = 0        # pass-through, no filter
        else:
            ntaps = len(_antialias_taps(self.up, self.down, half_len=half_len,
                                        window=window))
            need = (ntaps + self.up - 1) // self.up
            self.hist = ((need + self.down - 1) // self.down) * self.down
            self.ahead = self.hist
        self._buf: np.ndarray | None = None
        self._started = False

    @property
    def latency_samples(self) -> int:
        """Input samples held back for look-ahead (output lags by this much)."""
        return self.ahead

    def reset(self) -> None:
        """Forget all carried history, e.g. after a retune or a stream restart."""
        self._buf = None
        self._started = False

    def _emit(self, buf, k):
        chunk = buf[:self.hist + k + self.ahead]
        drop = self.hist * self.up // self.down
        keep = (self.hist + k) * self.up // self.down
        return resample_poly(chunk, self.up, self.down,
                             half_len=self.half_len, window=self.window)[drop:keep]

    def __call__(self, block):
        block = np.asarray(block)
        if self.up == 1 and self.down == 1:
            return block
        if self._buf is None:
            # zeros as the history of a stream that has not started: the same
            # thing resample_poly's own startup transient sees
            self._buf = np.zeros(self.hist, dtype=block.dtype)
        if block.size == 0:
            return block[:0]
        buf = np.concatenate([self._buf, block])
        avail = buf.size - self.hist - self.ahead
        if avail < self.down:
            self._buf = buf
            return block[:0]
        k = (avail // self.down) * self.down
        out = self._emit(buf, k)
        self._buf = buf[k:]
        return out

    def flush(self):
        """Emit the output for samples still held back for look-ahead.

        Pads the future with zeros, exactly as ``resample_poly`` does at the
        end of a whole record; call once when the stream ends.
        """
        if self._buf is None or self.up == 1 and self.down == 1:
            return np.zeros(0)
        buf = self._buf
        rem = buf.size - self.hist               # input not yet accounted for
        if rem <= 0:
            return buf[:0]
        pad = np.zeros(self.ahead, dtype=buf.dtype)
        full = np.concatenate([buf, pad])
        drop = self.hist * self.up // self.down
        n_out = int(np.ceil(rem * self.up / self.down))
        out = resample_poly(full, self.up, self.down,
                            half_len=self.half_len, window=self.window)[drop:drop + n_out]
        self._buf = buf[-self.hist:].copy()      # history only; nothing pending
        return out


class StreamingDeemphasis:
    """``deemphasis`` carried across blocks, exactly.

    The de-emphasis filter is one pole: y[n] = a*x[n] + r*y[n-1], r = 1 - a.
    It is linear, so its output on a block equals the zero-state response
    (what ``deemphasis`` computes) plus the decaying trace of the state it
    was left in: y_prev * r**(k+1) for k = 0, 1, ... That trace is added here,
    which makes the streamed output identical to the whole-record one rather
    than close to it, with no warm-up region to discard.
    """

    def __init__(self, sample_rate, tau_us=75.0):
        self.sample_rate = float(sample_rate)
        self.tau_us = float(tau_us)
        dt = 1.0 / self.sample_rate
        self._r = 1.0 - dt / (self.tau_us * 1e-6 + dt)     # same a as deemphasis
        self._y_prev = 0.0

    def reset(self) -> None:
        """Forget all carried history, e.g. after a retune or a stream restart."""
        self._y_prev = 0.0

    def __call__(self, block):
        block = np.asarray(block, dtype=np.float64)
        n = block.size
        if n == 0:
            return block
        out = deemphasis(block, self.sample_rate, tau_us=self.tau_us)
        if self._y_prev:
            out = out + self._y_prev * self._r ** np.arange(1, n + 1)
        self._y_prev = float(out[-1])
        return out
