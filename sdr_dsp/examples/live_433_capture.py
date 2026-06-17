#! /usr/bin/python3
"""live_433_capture.py -- capture a 433 MHz burst (e.g. a key fob) and save it.

Records live IQ at 433.92 MHz via a HackRF and writes a SigMF recording you can
then decode offline with examples/ook_decoder.py. Press your key fob (or trigger
whatever 433 MHz device) while this runs.

Requires hackrfpy + hackrf-tools (examples extra):  pip install hackrfpy

Usage:
    python examples/live_433_capture.py --seconds 3 --out keyfob_433.iq
    python examples/live_433_capture.py --freq 315e6   # some fobs use 315 MHz
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "examples")
from sdr_dsp.io import save_iq


def main():
    p = argparse.ArgumentParser(description="Capture a 433 MHz burst to SigMF.")
    p.add_argument("--freq", type=float, default=433.92e6)
    p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--seconds", type=float, default=3.0)
    p.add_argument("--lna", type=int, default=32)
    p.add_argument("--vga", type=int, default=30)
    p.add_argument("--amp", action="store_true", help="enable front-end amp")
    p.add_argument("--out", default="keyfob_433.iq")
    args = p.parse_args()

    try:
        from hackrf_capture import HackRFCapture
    except ImportError as e:
        print(f"needs hackrfpy: pip install hackrfpy  ({e})", file=sys.stderr)
        return 1

    n = int(args.rate * args.seconds)
    print(f"[*] capturing {args.seconds}s @ {args.freq/1e6:g} MHz, "
          f"{args.rate/1e6:g} Msps  (press your device NOW)")
    with HackRFCapture(args.freq, args.rate, lna=args.lna, vga=args.vga,
                       amp=args.amp) as src:
        iq = src.read(n)

    # quick check: did we catch a burst? look at envelope dynamic range
    env = np.abs(iq)
    hi, lo = np.percentile(env, 99), np.percentile(env, 50)
    print(f"[*] captured {len(iq):,} samples; envelope hi/lo = "
          f"{hi:.3f}/{lo:.3f}")
    if hi < lo * 2:
        print("[!] little amplitude variation -- may not have caught a burst. "
              "Try again, press the device closer, or raise --lna/--vga.")

    save_iq(args.out, iq, args.rate, center_freq=args.freq)
    print(f"[*] saved -> {args.out}")
    print(f"    decode it: python examples/ook_decoder.py {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
