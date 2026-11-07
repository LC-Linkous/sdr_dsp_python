"""Automatic gain control (AGC) -- explicit and observable.

AGC tracks a signal's level and continuously adjusts gain to hold it near a
target, so a fading or varying signal stays in a usable range. That's genuinely
useful (fading transmitters, satellite passes, demods that assume normalized
input) but it sits in tension with this library's core principle: the library
never silently changes how you'd interpret your data.

The resolution: this AGC is never silent. It returns the full per-sample gain
trace it applied, so nothing is hidden -- you can see exactly what the loop did,
undo it (adjusted / gain_trace == original), or recover absolute level from the
trace. It is opt-in, in its own module, never wired into any default path. This
is the same bargain the recovery loops make: yes it modifies the signal, but it
shows you precisely how.

IMPORTANT, read this: AGC discards absolute level information -- after AGC, the
amplitude reflects the loop, not the signal. Do any dBm / calibration work
BEFORE applying AGC, never after. With max_gain=None (the default) there is no
ceiling, so during silence the loop will happily amplify pure noise up toward
the target ("gain runaway in the gaps"); set max_gain if that matters for your
signal. Exact digital silence (|x| == 0, e.g. a muted or zero-padded gap) is the
one case the loop refuses to chase: there is no level to track, so the gain is
HELD until signal returns. Without that hold the gain would overflow to inf and
poison every later sample with NaN.

Two entry points, ONE algorithm:
  - agc(iq, ...) processes a whole array (owns its loop state for that call).
  - AGC is a tiny stateful wrapper for streaming that remembers the gain across
    blocks (so it doesn't lurch at block boundaries) and calls the same core.
The stage does not reimplement anything; it is the function plus a memory of the
last gain value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def agc(iq, mode="rms", target=1.0, attack=0.01, decay=0.001, max_gain=None,
        _initial_gain=1.0, _initial_level=None):
    """Apply automatic gain control. Returns (adjusted_iq, gain_trace). OUR code.

    Drives the signal level toward `target` with a one-pole tracking loop. The
    gain rises slowly when the signal is weak (decay rate) and clamps down
    quickly when it's strong (attack rate) -- fast attack avoids clipping, slow
    decay avoids pumping on noise. Both rates are in (0, 1]; larger = faster.

    mode:     "rms" tracks average power (smoother; good for analog/voice),
              "peak" tracks the running peak (twitchier; better anti-clipping).
    target:   the level the loop steers the signal toward.
    attack:   tracking rate when the measured level is ABOVE target (gain down).
    decay:    tracking rate when the measured level is BELOW target (gain up).
    max_gain: optional ceiling on the gain (None = no ceiling; see the module
              note about gain runaway during silence).

    Returns:
        adjusted_iq:  iq * gain_trace
        gain_trace:   the per-sample gain that was applied (same length as iq).
                      This is the whole point -- it makes the AGC observable and
                      reversible: iq == adjusted_iq / gain_trace.

    The _initial_* args let the streaming AGC stage continue a loop across blocks
    and aren't normally set by hand.
    """
    adjusted, gain, _level = _agc_core(iq, mode, target, attack, decay, max_gain,
                                       _initial_gain, _initial_level)
    return adjusted, gain


# Exact-zero detection floor. Below this the sample carries no level
# information (digital silence / zero padding), so the loop holds rather than
# chasing an infinite error. complex64 quantization is ~1e-7 of full scale, so
# any real sample, however quiet, sits far above this.
_SILENCE = 1e-30


def _agc_core(iq, mode, target, attack, decay, max_gain, initial_gain,
              initial_level):
    """The AGC loop. Returns (adjusted, gain_trace, final_level).

    `final_level` is the loop's level-tracker state after the last sample --
    what the streaming stage must carry to the next block (as `initial_level`)
    so the loop is continuous. It is the tracked INPUT level (smoothed |x| in
    rms mode, the held peak of |x| in peak mode): the exact variable the loop
    iterates, so a streamed run reproduces a whole-record run bit for bit.
    """
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    if n == 0:
        return iq, np.zeros(0, dtype=np.float64), initial_level
    if mode not in ("rms", "peak"):
        raise ValueError("mode must be 'rms' or 'peak'")
    peak = mode == "peak"

    mag = np.abs(iq).astype(np.float64)
    gain = np.empty(n, dtype=np.float64)
    g = float(initial_gain)
    eps = 1e-12

    # The loop tracks the INPUT level and steers the gain so that
    # (tracked input level) * gain -> target. `lvl_in` is the running estimate
    # of |x|:
    #   rms  : one-pole smoothed |x|, rising at `attack`, falling at `decay`
    #          (a step up is seen quickly, a fade is believed slowly)
    #   peak : classic peak detector -- jumps to |x| instantly when it is above
    #          the held value, releases at `decay`. Twitchier, and what you
    #          want when the goal is "never clip".
    # Tracking the input rather than the output matters. An earlier version
    # smoothed the OUTPUT level, which depends on the very gain it steers:
    # after a +20 dB step the gain cut at `attack`, the output-level estimate
    # could only fall at `decay`, so it stayed above target and the gain kept
    # cutting -- the output collapsed to ~0 and crawled back over thousands of
    # samples. The input level does not move when the gain does, so the gain
    # converges to target/lvl_in without that feedback.
    if initial_level is not None:
        lvl_in = float(initial_level)
    elif mag[0] > 0:
        lvl_in = mag[0]
    else:
        lvl_in = target / g if g > 0 else target

    for i in range(n):
        if mag[i] < _SILENCE:
            # digital silence: no level to track, hold the gain (see module note)
            gain[i] = g
            continue
        if peak:
            if mag[i] > lvl_in:
                lvl_in = mag[i]
            else:
                lvl_in += decay * (mag[i] - lvl_in)
        else:
            a = attack if mag[i] > lvl_in else decay
            lvl_in += a * (mag[i] - lvl_in)
        level = lvl_in * g                        # where the output level sits
        # error between where the output level is and where we want it; adjust
        # gain multiplicatively (AGC works in log/ratio space, not additively)
        err = target / max(level, eps)
        # move gain a fraction of the way toward the corrected value; use attack
        # when we need to REDUCE gain (err < 1), decay when increasing
        rate = attack if err < 1.0 else decay
        g *= (1.0 + rate * (err - 1.0))
        if max_gain is not None:
            g = min(g, float(max_gain))
        g = max(g, 0.0)
        gain[i] = g

    adjusted = (iq * gain.astype(np.float64)).astype(np.complex64)
    return adjusted, gain, lvl_in


@dataclass
class AGC:
    """Streaming AGC: the agc() loop with memory across blocks.

    A per-block AGC that reset each block would lurch at every boundary. This
    holds the gain and level between blocks so the loop is continuous, and calls
    the same agc() core -- it adds memory, not new DSP. Use it as a Pipeline
    stage; the last gain trace is kept on `.last_gain` so a .tap() can watch it.

        stage = AGC(mode="rms", target=1.0, attack=0.01, decay=0.001)
        pipe.add(stage, "agc").tap(lambda b: meter.update(stage.last_gain))

    Like the function, it never hides what it did: every processed block has its
    gain trace available, and the same caveats apply (do calibration before AGC;
    set max_gain to bound silent-gap runaway).
    """
    mode: str = "rms"
    target: float = 1.0
    attack: float = 0.01
    decay: float = 0.001
    max_gain: float | None = None

    def __post_init__(self):
        self._gain = 1.0
        self._level = None
        self.last_gain = np.zeros(0, dtype=np.float64)

    def __call__(self, block):
        """Process one block, continuing the loop from the previous block."""
        adjusted, gain, level = _agc_core(block, self.mode, self.target,
                                          self.attack, self.decay, self.max_gain,
                                          self._gain, self._level)
        if len(gain):
            self._gain = float(gain[-1])
            # carry the loop's SMOOTHED output-level estimate. Carrying
            # |last output sample| instead (an earlier version did) restarts
            # the level tracker from an unsmoothed value at every block, so a
            # streamed run diverged from the whole-record run by several %.
            self._level = float(level)
        self.last_gain = gain
        return adjusted

    def reset(self):
        """Forget the carried state (start the next block fresh)."""
        self._gain = 1.0
        self._level = None
        self.last_gain = np.zeros(0, dtype=np.float64)
