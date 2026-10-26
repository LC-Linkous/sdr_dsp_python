#! /usr/bin/python3
"""gain_search_demo.py -- auto-gain ANY radio by describing its knobs.

`search_gain_model` finds usable gain settings for a radio it has never seen,
given only a `GainModel`: an ordered list of `GainStage`s, each naming one
gain control, its legal steps, and whether it is a *frontend* stage (sets SNR,
before the noise is added) or a *level* stage (sets ADC fill, after). The
engine then does two things in order:

  1. sweep the frontend stages and keep the combination that scores best on a
     measured QUALITY metric -- here the FM stereo pilot, which amplified
     noise cannot fake;
  2. walk the level stage until the ADC sits in its target window, stepping a
     frontend stage only when the level knob runs out of range.

No radio is needed to see this work. The script simulates three radios with
different knob layouts and runs the same engine on each:

  hackrf   LNA + switchable +14 dB amp (frontend), VGA (level) -- the
           HackRF One, via the shipped `hackrf_gain_model()`
  rtlsdr   one tuner gain, no separate level stage -- `rtlsdr_gain_model()`
  custom   a made-up two-knob radio built from GainStage by hand, to show
           that the model is the whole adapter: no change to the library

Each radio reports its chosen settings, final ADC counts, measured quality
and status, and the full probe trace with `--trace`. Try `--antenna 0.03`
(weak signal: watch the amp come on) or `--antenna 300` (overload: watch the
back-off), and `--no-quality` to see what level-only search does instead.

Needs: nothing beyond the library. Pure synthetic, no hardware.

Usage:
    python examples/gain_search_demo.py
    python examples/gain_search_demo.py --radio rtlsdr --antenna 0.03 --trace
    python examples/gain_search_demo.py --radio custom --no-quality
"""
import argparse

import numpy as np

from sdr_dsp.core import fm_pilot_excess_db
from sdr_dsp.core.gain_search import (GainModel, GainStage, hackrf_gain_model,
                                      rtlsdr_gain_model, search_gain_model)

FS = 2_000_000.0


class SimulatedRadio:
    """A radio with one or more pre-ADC gain stages and an 8-bit ADC.

    The physics the search relies on, in two lines: FRONTEND gain amplifies
    the signal before the receiver's own noise is added, so it raises SNR;
    LEVEL gain amplifies signal and noise together, so it only moves the
    ADC fill. Too much of either clips the ADC at +-127 counts.

    `frontend_db(settings)` and `level_db(settings)` map the dict the search
    hands us onto those two numbers, so one class simulates every layout.
    """

    def __init__(self, antenna_counts, frontend_db, level_db, seed=2):
        self.ant = float(antenna_counts)
        self.frontend_db = frontend_db
        self.level_db = level_db
        rng = np.random.default_rng(seed)
        n = int(FS * 0.06)
        t = np.arange(n) / FS
        # a stereo-FM-shaped message: audio plus the 19 kHz pilot
        msg = 0.6 * np.sin(2 * np.pi * 500 * t) + 0.09 * np.sin(
            2 * np.pi * 19_000 * t)
        self._clean = np.exp(2j * np.pi * 75e3 * np.cumsum(msg) / FS)
        self._noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)
                       ) / np.sqrt(2)

    def probe(self, settings):
        g_fe = 10 ** (self.frontend_db(settings) / 20)
        g_lv = 10 ** (self.level_db(settings) / 20)
        sig = self.ant * g_fe * g_lv
        noise = 1.5 * g_lv + 0.7                 # receiver noise, after the frontend
        iq = (sig * self._clean + noise * self._noise) / 128.0
        lim = 127 / 128
        return (np.clip(iq.real, -1.0, lim)
                + 1j * np.clip(iq.imag, -1.0, lim)).astype(np.complex64)


def build(radio_name, antenna):
    """Return (GainModel, probe) for one of the three simulated radios."""
    if radio_name == "hackrf":
        model = hackrf_gain_model()
        radio = SimulatedRadio(
            antenna,
            frontend_db=lambda s: s["lna"] + (14.0 if s.get("amp") else 0.0),
            level_db=lambda s: s["vga"])
    elif radio_name == "rtlsdr":
        model = rtlsdr_gain_model()
        # one knob does both jobs: the model has no level stage, and the
        # engine walks the same frontend knob for level in stage 2
        radio = SimulatedRadio(antenna,
                               frontend_db=lambda s: s["gain"],
                               level_db=lambda s: 0.0)
    else:
        # A radio the library has never heard of: a 0-30 dB RF stage in 6 dB
        # steps and a 0-40 dB IF stage in 4 dB steps. Describing it IS the
        # adapter; nothing in src/ changes.
        model = GainModel([
            GainStage("rf", tuple(range(0, 31, 6)), "frontend"),
            GainStage("if", tuple(range(0, 41, 4)), "level"),
        ])
        radio = SimulatedRadio(antenna,
                               frontend_db=lambda s: s["rf"],
                               level_db=lambda s: s["if"])
    return model, radio.probe


def quality(iq):
    """How much SIGNAL a probe holds: the 19 kHz pilot above its neighbourhood,
    in dB. None when the record cannot be scored."""
    return fm_pilot_excess_db(iq, FS, nfft=8192)


def main():
    p = argparse.ArgumentParser(description="Device-agnostic gain search on "
                                "simulated radios.")
    p.add_argument("--radio", choices=["hackrf", "rtlsdr", "custom", "all"],
                   default="all")
    p.add_argument("--antenna", type=float, default=1.0,
                   help="signal strength at the antenna, in ADC counts at 0 dB "
                        "gain (0.03 weak, 1 normal, 300 overload)")
    p.add_argument("--no-quality", action="store_true",
                   help="search on ADC level only, without the pilot metric")
    p.add_argument("--trace", action="store_true",
                   help="print every probe the search took")
    args = p.parse_args()

    radios = ["hackrf", "rtlsdr", "custom"] if args.radio == "all" else [args.radio]
    for name in radios:
        model, probe = build(name, args.antenna)
        stages = ", ".join(f"{s.name}[{s.role}]" for s in model.stages)
        print(f"\n== {name}: {stages}")
        r = search_gain_model(probe, model,
                              quality=None if args.no_quality else quality)
        q = r["quality_db"]
        print(f"   settings : {r['settings']}")
        print(f"   counts   : {r['counts']:.0f} / 128  (target 45-110)")
        print(f"   quality  : {'n/a' if q is None else f'{q:+.1f} dB pilot'}")
        print(f"   status   : {r['status']}   ({len(r['probes'])} probes)")
        if args.trace:
            for i, (s, c) in enumerate(r["probes"]):
                print(f"     {i:2d}  {s}  -> {c:5.1f} counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
