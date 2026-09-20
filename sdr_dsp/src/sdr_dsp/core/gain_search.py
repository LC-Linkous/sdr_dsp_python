"""Gain search: find receiver gain settings that capture a usable signal.

OUR code. Device-agnostic: the caller supplies a ``probe(lna, vga, amp)``
callable that returns a short normalized IQ capture, and this module decides
where to point the gain. Previously three near-copies of this logic lived in
``examples/collect_sample_data.py``, ``tools/collect_dev_data.py`` and
``examples/live_fm_listen.py``; they now all call here.

Why level alone is not enough
-----------------------------
The old searches walked the VGA until the peak ADC level landed in a target
window. That converges -- but on the wrong thing. The VGA sits after the
point in the chain where the signal-to-noise ratio is already decided, so
raising it amplifies signal and receiver noise together: on a weak antenna
the search happily lifts the NOISE FLOOR into the target window, reports
success, and the recording is hiss at a healthy-looking level. Only the
front end -- the RF amp and the LNA -- improves what is there to record, and
the old searches never touched the amp at all.

So this search runs in two stages when a quality metric is available:

1. **Front end for SNR.** Probe amp/LNA combinations at VGA=0 (so nothing is
   excluded by clipping) and score each with the caller's ``quality``
   function -- e.g. channel excess from ``capture_health``, or
   ``fm_pilot_excess_db`` for broadcast FM. Keep the front end that measures
   best; on ties prefer less gain.
2. **VGA for level.** With the front end fixed, walk the VGA until the peak
   ADC level sits in the target window. This stage changes how the ADC is
   used, not what can be heard.

Without a quality metric (nothing known to be transmitting, so there is
nothing to score), it falls back to the level-only walk -- but will now
escalate through the LNA and finally the amp before declaring ``too_weak``.
"""

from __future__ import annotations

import numpy as np

DEFAULT_LNA_STEPS = tuple(range(0, 41, 8))    # HackRF: 0..40 dB in 8 dB steps
DEFAULT_VGA_STEPS = tuple(range(0, 63, 2))    # HackRF: 0..62 dB in 2 dB steps


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


def search_gain(probe, *, lna_steps=DEFAULT_LNA_STEPS,
                vga_steps=DEFAULT_VGA_STEPS, use_amp=True, quality=None,
                target=(45.0, 110.0), clip=120.0, full_scale=128.0,
                max_level_probes=14, min_quality_db=3.0):
    """Find (lna, vga, amp) for a usable capture. Returns a result dict.

    Args:
        probe:     callable ``probe(lna, vga, amp) -> iq`` returning a short
                   normalized complex capture at those settings.
        lna_steps, vga_steps: the hardware's legal gain values, in dB.
        use_amp:   whether the front-end RF amp may be enabled at all.
        quality:   optional callable ``quality(iq) -> float | None`` scoring
                   how much *signal* a probe contains, in dB (higher =
                   better). E.g. channel excess from ``capture_health``, or
                   ``fm_pilot_excess_db`` for broadcast FM. When given, the
                   front end is chosen by this score, not by level.
        target:    (lo, hi) peak-count window the VGA stage aims for.
        clip:      counts at/above which a probe is treated as clipping.
        full_scale: counts scale of a normalized full-scale sample (128 for
                   int8 normalized by the loader).
        max_level_probes: probe budget for the VGA/level stage.
        min_quality_db: front-end scores below this count as "nothing
                   detected" for the status decision.

    Returns a dict:
        ``lna``, ``vga``, ``amp``   -- the chosen settings
        ``counts``                  -- peak counts measured at them
        ``quality_db``              -- best front-end score (None w/o quality)
        ``status``                  -- "ok" | "too_weak" | "clipping"
        ``probes``                  -- every probe as (lna, vga, amp, counts)
    """
    lna_steps = sorted(lna_steps)
    vga_steps = sorted(vga_steps)
    lo, hi = target
    probes = []

    def take(lna, vga, amp):
        iq = probe(lna, vga, amp)
        c = peak_counts(iq, full_scale)
        probes.append((lna, vga, amp, round(c, 1)))
        return iq, c

    # ---- stage 1: choose the front end by measured signal quality --------
    amp = False
    lna = lna_steps[min(2, len(lna_steps) - 1)]   # legacy starting point
    best_q = None
    if quality is not None:
        best = None                                # (score, -gain, lna, amp)
        for amp_try in ((False, True) if use_amp else (False,)):
            for lna_try in lna_steps:
                iq, c = take(lna_try, vga_steps[0], amp_try)
                if c >= clip:
                    break                          # more LNA only clips harder
                q = quality(iq)
                if q is None:
                    continue
                key = (q, -(lna_try + (14 if amp_try else 0)))
                if best is None or key > best[0:2]:
                    best = (*key, lna_try, amp_try)
        if best is not None:
            best_q, _, lna, amp = best

    # ---- stage 2: walk the VGA until the level sits in the window --------
    vga = vga_steps[len(vga_steps) // 3]           # legacy starting point
    counts = 0.0
    status = "ok"
    for _ in range(max_level_probes):
        _, counts = take(lna, vga, amp)
        if lo <= counts <= hi:
            break
        if counts >= clip:
            if vga > vga_steps[0]:
                vga = _snap(max(vga_steps[0], vga - 6), vga_steps)
            elif lna > lna_steps[0]:
                lna = _snap(lna - 8, lna_steps)
                vga = vga_steps[0]
            elif amp:
                amp = False
                vga = vga_steps[0]
            else:
                status = "clipping"
                break
        elif counts < lo:
            if vga < vga_steps[-1]:
                deficit_db = 20 * np.log10(max(lo, 1.0) / max(counts, 0.5))
                vga = _snap(min(vga_steps[-1],
                                vga + max(2, int(deficit_db // 2) * 2)),
                            vga_steps)
            elif quality is None and lna < lna_steps[-1]:
                # level-only fallback may still escalate the front end
                lna = _snap(lna + 8, lna_steps)
                vga = vga_steps[len(vga_steps) // 3]
            elif quality is None and use_amp and not amp:
                amp = True
                vga = vga_steps[len(vga_steps) // 3]
            else:
                status = "too_weak"
                break
        else:                                       # above window, below clip
            if vga > vga_steps[0]:
                vga = _snap(max(vga_steps[0], vga - 2), vga_steps)
            elif lna > lna_steps[0]:
                lna = _snap(lna - 8, lna_steps)
                vga = vga_steps[0]
            elif amp:
                amp = False
                vga = vga_steps[0]
            else:
                status = "clipping"
                break

    if (status == "ok" and not (lo <= counts <= hi)):
        # probe budget exhausted: report honestly rather than pretending
        status = "too_weak" if counts < lo else "clipping"
    if (status == "ok" and quality is not None and best_q is not None
            and best_q < min_quality_db):
        # level converged, but the best front end never saw a signal: the
        # window is filled with amplified noise, and saying "ok" here is
        # exactly the failure mode this module exists to prevent.
        status = "too_weak"

    return {"lna": lna, "vga": vga, "amp": bool(amp),
            "counts": round(counts, 1),
            "quality_db": None if best_q is None else round(best_q, 1),
            "status": status, "probes": probes}
