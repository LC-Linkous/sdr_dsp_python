"""Gain search: find receiver gain settings that capture a usable signal.

OUR code. Device-agnostic in two layers:

* ``search_gain_model`` is the general engine. It drives an arbitrary radio
  described by a :class:`GainModel` -- a set of named gain STAGES, each either
  a "frontend" stage (improves signal-to-noise: RF amp, LNA, a tuner's single
  gain) or a "level" stage (adjusts ADC level after SNR is set: the HackRF's
  VGA). The caller supplies ``probe(settings) -> iq``; the engine decides where
  to point the gain. This is what keeps the search from being HackRF-locked --
  point it at ``rtlsdr_gain_model(...)`` (one knob, no separate level stage) and
  the same algorithm works.
* ``search_gain`` is the HackRF specialization kept for existing callers: the
  familiar ``probe(lna, vga, amp)`` signature and ``{lna, vga, amp, ...}``
  result. It builds the HackRF model and delegates to the engine.

Why level alone is not enough
-----------------------------
The old searches walked the VGA until the peak ADC level landed in a target
window. That converges -- but on the wrong thing. The VGA sits after the point
in the chain where the signal-to-noise ratio is already decided, so raising it
amplifies signal and receiver noise together: on a weak antenna the search
happily lifts the NOISE FLOOR into the target window, reports success, and the
recording is hiss at a healthy-looking level. Only the front end -- the RF amp
and the LNA -- improves what is there to record.

So the search runs in two stages when a quality metric is available:

1. **Front end for SNR.** Probe frontend combinations at the level stage's
   minimum (so nothing is excluded by clipping) and score each with the
   caller's ``quality`` function. Keep the front end that measures best; on
   ties prefer less gain.
2. **Level.** With the front end fixed, walk the level stage until the peak ADC
   level sits in the target window. On a radio with no separate level stage
   (e.g. an RTL-SDR's single gain), this walks the SNR-chosen frontend knob
   itself, trading a little SNR for a sane level rather than pretending a
   second knob exists.

Without a quality metric it falls back to the level-only walk, escalating
through the remaining frontend stages before declaring ``too_weak``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

DEFAULT_LNA_STEPS = tuple(range(0, 41, 8))    # HackRF: 0..40 dB in 8 dB steps
DEFAULT_VGA_STEPS = tuple(range(0, 63, 2))    # HackRF: 0..62 dB in 2 dB steps
# R820T tuner gains reported by librtlsdr, in dB -- the legal values for a
# typical RTL-SDR. A single knob: it is a frontend stage with no separate VGA.
DEFAULT_RTLSDR_GAIN_STEPS = (
    0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7,
    22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4,
    43.9, 44.5, 48.0, 49.6)


def peak_counts(iq, full_scale=128.0):
    """Peak per-component ADC utilization of a normalized capture, in counts.

    Per component -- max(|I|, |Q|) -- because I and Q are separate converters
    and each clips on its own; the complex magnitude can read above full
    scale, which no converter can produce. ``full_scale`` matches the
    loader's normalization (``load_iq`` divides int8 by 128).
    """
    iq = np.asarray(iq)
    if iq.size == 0:
        return 0.0
    return float(max(np.max(np.abs(iq.real)),
                     np.max(np.abs(iq.imag)))) * float(full_scale)


def _snap(value, steps):
    return min(steps, key=lambda s: abs(s - value))


# ==========================================================================
# The device abstraction: a radio's gain as named stages.
# ==========================================================================
@dataclass(frozen=True)
class GainStage:
    """One gain control on a radio.

    name:    the key the probe expects (e.g. "lna", "vga", "amp", "gain").
    steps:   the legal settings, ascending by gain (device values -- ints/floats
             of dB, or booleans for an on/off amp).
    role:    "frontend" (improves SNR) or "level" (sets ADC level after SNR).
    db:      value -> dB weight, used to rank "less gain" on ties and to size
             level moves. Defaults to float(value); an on/off amp passes its
             own (e.g. 14 dB when True).
    default: the value used when this stage is NOT chosen by a quality sweep
             (no quality metric available). None means the lowest step.
    """

    name: str
    steps: tuple
    role: str
    db: Callable[..., float] = float
    default: object = None

    def __post_init__(self):
        if self.role not in ("frontend", "level"):
            raise ValueError(f"role must be 'frontend' or 'level', got {self.role!r}")
        if not self.steps:
            raise ValueError(f"stage {self.name!r} has no steps")

    def default_value(self):
        return self.steps[0] if self.default is None else self.default

    def min_gap(self):
        """Smallest positive spacing between consecutive numeric steps."""
        nums = [s for s in self.steps if isinstance(s, (int, float))
                and not isinstance(s, bool)]
        gaps = [b - a for a, b in zip(nums, nums[1:]) if b > a]
        return min(gaps) if gaps else 1.0


class GainModel:
    """A radio's gain, as an ordered set of :class:`GainStage`.

    Frontend stages are swept for SNR; the last level stage (or, if there is
    none, the last frontend stage) is walked for ADC level. Escalation on
    clip/too-weak steps the remaining frontend stages in declared order, so
    put the cheaper-to-change stage first (LNA before amp).
    """

    def __init__(self, stages: Sequence[GainStage]):
        self.stages = tuple(stages)
        if not self.stages:
            raise ValueError("a GainModel needs at least one stage")
        self.frontend = tuple(s for s in self.stages if s.role == "frontend")
        self.level = tuple(s for s in self.stages if s.role == "level")

    def by_name(self, name):
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)

    def min_settings(self):
        return {s.name: s.steps[0] for s in self.stages}

    def default_frontend(self):
        return {s.name: s.default_value() for s in self.frontend}

    def frontend_grid(self):
        """Every combination of frontend stage settings, as dicts."""
        if not self.frontend:
            yield {}
            return
        names = [s.name for s in self.frontend]
        for combo in itertools.product(*(s.steps for s in self.frontend)):
            yield dict(zip(names, combo))

    def frontend_db(self, combo):
        return sum(s.db(combo[s.name]) for s in self.frontend if s.name in combo)

    def level_stage(self):
        """The stage walked for ADC level: the last level stage, or -- when the
        radio has none -- the last frontend stage (one knob does both)."""
        return self.level[-1] if self.level else self.frontend[-1]


def hackrf_gain_model(lna_steps=DEFAULT_LNA_STEPS, vga_steps=DEFAULT_VGA_STEPS,
                      *, use_amp=True) -> GainModel:
    """The HackRF One: LNA + optional +14 dB amp (frontend) and VGA (level)."""
    lna = tuple(sorted(lna_steps))
    vga = tuple(sorted(vga_steps))
    stages = [GainStage("lna", lna, "frontend",
                        default=lna[min(2, len(lna) - 1)])]   # legacy start
    if use_amp:
        stages.append(GainStage("amp", (False, True), "frontend",
                                db=lambda v: 14.0 if v else 0.0, default=False))
    stages.append(GainStage("vga", vga, "level"))
    return GainModel(stages)


def rtlsdr_gain_model(gain_steps=DEFAULT_RTLSDR_GAIN_STEPS) -> GainModel:
    """An RTL-SDR: a SINGLE tuner-gain knob, no separate level stage.

    One frontend stage means stage 1 picks the SNR-best gain and stage 2 then
    nudges that SAME knob for level -- the honest behavior for a radio that
    cannot set SNR and level independently.
    """
    g = tuple(sorted(gain_steps))
    mid = g[len(g) // 2]
    return GainModel([GainStage("gain", g, "frontend", default=mid)])


# ==========================================================================
# The general engine.
# ==========================================================================
def search_gain_model(probe, model: GainModel, *, quality=None,
                      target=(45.0, 110.0), clip=120.0, full_scale=128.0,
                      max_level_probes=14, min_quality_db=3.0):
    """Find gain settings for a usable capture on any radio. Returns a dict.

    probe:   callable ``probe(settings: dict) -> iq`` returning a short
             normalized complex capture, where ``settings`` maps each stage's
             name to a legal value from its ``steps``.
    model:   the radio's :class:`GainModel`.
    quality: optional ``quality(iq) -> float | None`` scoring how much SIGNAL a
             probe holds, in dB. When given, the front end is chosen by this
             score, not by level.
    target, clip, full_scale, max_level_probes, min_quality_db: as documented
             on :func:`search_gain`.

    Returns ``{"settings", "counts", "quality_db", "status", "probes"}`` where
    settings is the chosen ``{stage: value}``, status is "ok"|"too_weak"|
    "clipping", and probes is every ``(settings, counts)`` taken.
    """
    lo, hi = target
    probes: list = []

    def take(settings):
        iq = probe(settings)
        c = peak_counts(iq, full_scale)
        probes.append((dict(settings), round(c, 1)))
        return iq, c

    settings = model.min_settings()
    lvl = model.level_stage()
    steps = lvl.steps
    min_step = lvl.min_gap()
    # frontend stages we may escalate through (never the level knob itself)
    escalate = [s for s in model.frontend if s.name != lvl.name]

    # ---- stage 1: choose the front end by measured signal quality --------
    best_q = None
    if quality is not None and model.frontend:
        best = None                                  # (q, -db, combo)
        for combo in model.frontend_grid():
            s = dict(settings)
            s.update(combo)
            s[lvl.name] = steps[0]                    # score at min level
            iq, c = take(s)
            if c >= clip:
                continue                             # clipping: can't score SNR
            q = quality(iq)
            if q is None:
                continue
            key = (q, -model.frontend_db(combo))
            if best is None or key > best[:2]:
                best = (q, -model.frontend_db(combo), combo)
        if best is not None:
            best_q = best[0]
            settings.update(best[2])
    else:
        settings.update(model.default_frontend())

    # ---- stage 2: walk the level stage until the level sits in the window -
    # If the level knob is a frontend stage that stage 1 just chose by SNR,
    # start the walk FROM that choice; otherwise from the legacy 1/3 point.
    if lvl.role == "frontend" and quality is not None and model.frontend:
        pass                                         # keep the SNR-chosen value
    else:
        settings[lvl.name] = steps[len(steps) // 3]

    def step_down_frontend():
        for s in escalate:
            cur = settings[s.name]
            idx = s.steps.index(cur)
            if idx > 0:
                settings[s.name] = s.steps[idx - 1]
                return True
        return False

    def step_up_frontend():
        for s in escalate:
            cur = settings[s.name]
            idx = s.steps.index(cur)
            if idx < len(s.steps) - 1:
                settings[s.name] = s.steps[idx + 1]
                return True
        return False

    counts = 0.0
    status = "ok"
    for _ in range(max_level_probes):
        _, counts = take(settings)
        cur = settings[lvl.name]
        if lo <= counts <= hi:
            break
        if counts >= clip:
            if cur > steps[0]:
                settings[lvl.name] = _snap(max(steps[0], cur - 6.0), steps)
            elif step_down_frontend():
                settings[lvl.name] = steps[0]
            else:
                status = "clipping"
                break
        elif counts < lo:
            if cur < steps[-1]:
                deficit_db = 20 * np.log10(max(lo, 1.0) / max(counts, 0.5))
                settings[lvl.name] = _snap(
                    min(steps[-1], cur + max(min_step, deficit_db)), steps)
            elif quality is None and step_up_frontend():
                settings[lvl.name] = steps[len(steps) // 3]
            else:
                status = "too_weak"
                break
        else:                                        # above window, below clip
            if cur > steps[0]:
                settings[lvl.name] = _snap(max(steps[0], cur - min_step), steps)
            elif step_down_frontend():
                settings[lvl.name] = steps[0]
            else:
                status = "clipping"
                break

    if status == "ok" and not (lo <= counts <= hi):
        # probe budget exhausted: report honestly rather than pretending
        status = "too_weak" if counts < lo else "clipping"
    if (status == "ok" and quality is not None and best_q is not None
            and best_q < min_quality_db):
        # level converged, but the best front end never saw a signal: the
        # window is filled with amplified noise, and saying "ok" here is
        # exactly the failure mode this module exists to prevent.
        status = "too_weak"

    return {"settings": settings, "counts": round(counts, 1),
            "quality_db": None if best_q is None else round(best_q, 1),
            "status": status, "probes": probes}


# ==========================================================================
# The HackRF specialization (backward-compatible public API).
# ==========================================================================
def search_gain(probe, *, lna_steps=DEFAULT_LNA_STEPS,
                vga_steps=DEFAULT_VGA_STEPS, use_amp=True, quality=None,
                target=(45.0, 110.0), clip=120.0, full_scale=128.0,
                max_level_probes=14, min_quality_db=3.0):
    """Find (lna, vga, amp) for a usable HackRF capture. Returns a result dict.

    A thin wrapper over :func:`search_gain_model` with the HackRF gain model,
    kept for the ``probe(lna, vga, amp)`` callers. Args and semantics are
    unchanged from before the GainModel refactor:

        probe:     ``probe(lna, vga, amp) -> iq`` (a short normalized capture).
        lna_steps, vga_steps: the hardware's legal gain values, in dB.
        use_amp:   whether the +14 dB front-end RF amp may be enabled.
        quality:   optional ``quality(iq) -> float | None`` (dB); when given,
                   the front end is chosen by this score, not by level.
        target:    (lo, hi) peak-count window the level stage aims for.
        clip:      counts at/above which a probe is treated as clipping.
        full_scale: counts scale of a normalized full-scale sample.
        max_level_probes: probe budget for the level stage.
        min_quality_db: front-end scores below this count as "nothing detected".

    Returns ``{lna, vga, amp, counts, quality_db, status, probes}`` with probes
    a list of ``(lna, vga, amp, counts)`` -- the original shape.
    """
    model = hackrf_gain_model(lna_steps, vga_steps, use_amp=use_amp)

    def probe_settings(s):
        return probe(s["lna"], s["vga"], s.get("amp", False))

    res = search_gain_model(
        probe_settings, model, quality=quality, target=target, clip=clip,
        full_scale=full_scale, max_level_probes=max_level_probes,
        min_quality_db=min_quality_db)

    s = res["settings"]
    probes = [(ps["lna"], ps["vga"], ps.get("amp", False), c)
              for ps, c in res["probes"]]
    return {"lna": s["lna"], "vga": s["vga"], "amp": bool(s.get("amp", False)),
            "counts": res["counts"], "quality_db": res["quality_db"],
            "status": res["status"], "probes": probes}
