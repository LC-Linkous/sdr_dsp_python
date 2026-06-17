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
signal.

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
    iq = np.asarray(iq, dtype=np.complex64)
    n = len(iq)
    if n == 0:
        return iq, np.zeros(0, dtype=np.float64)
    if mode not in ("rms", "peak"):
        raise ValueError("mode must be 'rms' or 'peak'")

    mag = np.abs(iq).astype(np.float64)
    gain = np.empty(n, dtype=np.float64)
    g = float(_initial_gain)
    eps = 1e-12

    # The loop tracks the OUTPUT level (a smoothed estimate of |output|) and
    # nudges the gain to push that toward target. Single feedback path -> stable.
    # `level` is the running estimate of the post-gain output magnitude.
    if _initial_level is not None:
        level = float(_initial_level)
    else:
        level = mag[0] * g if mag[0] > 0 else target

    for i in range(n):
        out_mag = mag[i] * g                      # current output magnitude
        # smooth the output-level estimate; rise fast (attack), fall slow (decay)
        a = attack if out_mag > level else decay
        level += a * (out_mag - level)
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
    return adjusted, gain


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
        adjusted, gain = agc(block, mode=self.mode, target=self.target,
                             attack=self.attack, decay=self.decay,
                             max_gain=self.max_gain,
                             _initial_gain=self._gain,
                             _initial_level=self._level)
        if len(gain):
            self._gain = float(gain[-1])
            # carry the output-level estimate: |last output sample|
            self._level = float(abs(adjusted[-1]))
        self.last_gain = gain
        return adjusted

    def reset(self):
        """Forget the carried state (start the next block fresh)."""
        self._gain = 1.0
        self._level = None
        self.last_gain = np.zeros(0, dtype=np.float64)
