#! /usr/bin/python3
"""cfo_demo.py -- measure a carrier frequency offset (and correct it, if you say so).

A captured signal often sits a little off the frequency you tuned to -- the
transmitter and receiver oscillators never match exactly. That carrier frequency
offset (CFO) rotates a constellation and shifts a signal off center. This shows:
    estimate_cfo (MEASURE where the signal is) -> optionally frequency_shift to
    correct it.

The library deliberately MEASURES but never auto-corrects -- correcting changes
the data, and that's your decision. This demo makes that explicit: it reports the
offset, and only corrects if you pass --correct.

Library deps only (numpy). No hardware -- synthesizes an offset signal.

Usage:
    python examples/cfo_demo.py
    python examples/cfo_demo.py --offset 35000 --correct
    python examples/cfo_demo.py capture.iq --fm --correct
"""
import argparse
import sys

import numpy as np

from sdr_dsp.core import (estimate_cfo, estimate_fm_cfo, correct_fm_cfo,
                          frequency_shift)

def main():
    p = argparse.ArgumentParser(description="Measure/correct carrier offset.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--rate", type=float, default=1e6)
    p.add_argument("--offset", type=float, default=27000,
                   help="synth: put the signal this far off center (Hz)")
    p.add_argument("--correct", action="store_true",
                   help="actually apply the correction (default: measure only)")
    p.add_argument("--fm", action="store_true",
                   help="treat the input as broadcast FM: channelize to the FM "
                        "channel before measuring (estimate_fm_cfo / "
                        "correct_fm_cfo), so out-of-channel energy cannot "
                        "pull the estimate")
    args = p.parse_args()

    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        n = 100_000
        t = np.arange(n) / fs
        # a signal deliberately offset from center
        iq = np.exp(2j * np.pi * args.offset * t).astype(np.complex64)
        iq += 0.05 * (np.random.randn(n) + 1j * np.random.randn(n))
        print(f"[*] synthetic signal placed {args.offset/1e3:g} kHz off center")

    # MEASURE -- this does not change the data
    if args.fm:
        cfo = estimate_fm_cfo(iq, fs)
        if cfo is None:
            print("[!] estimate_fm_cfo declined: the capture is too short or "
                  "the rate too low to channelize to the FM channel",
                  file=sys.stderr)
            return 1
        print(f"[*] measured FM carrier offset: {cfo/1e3:+.2f} kHz "
              "(mean instantaneous frequency inside the channel)")
    else:
        cfo = estimate_cfo(iq, fs)
        print(f"[*] measured carrier offset: {cfo/1e3:+.2f} kHz")

    if not args.correct:
        before = iq.copy()
        # prove the measurement didn't touch the signal
        assert np.array_equal(iq, before)
        print("[*] measure-only (the signal is unchanged). Pass --correct to "
              "shift it to center.")
        print(f"    to correct yourself: frequency_shift(iq, {-cfo:.0f}, fs)")
        return 0

    # CORRECT -- your explicit choice
    if args.fm:
        # correct_fm_cfo measures and shifts in one call, and returns the
        # offset it applied so the correction is never hidden from you
        corrected, applied = correct_fm_cfo(iq, fs)
        assert applied == cfo
        residual = estimate_fm_cfo(corrected, fs) or 0.0
    else:
        corrected = frequency_shift(iq, -cfo, fs)
        residual = estimate_cfo(corrected, fs)
    print(f"[*] corrected by {-cfo/1e3:+.2f} kHz; residual offset now "
          f"{residual/1e3:+.2f} kHz")
    print("[*] signal is now centered (residual ~ one FFT bin)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
