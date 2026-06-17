#! /usr/bin/python3
"""burst_detector.py -- find where the packets are in a capture.

Real ISM-band captures are mostly silence with short bursts (a sensor wakes,
transmits a packet, sleeps). Before decoding anything you need to find those
bursts. This uses the energy detector:
    envelope -> threshold -> start/stop index spans, with gap-merging and
    minimum-length filtering to reject noise blips.

Shows each detected burst's timing and power -- the triage step for packet work.

Library deps only (numpy). No hardware -- synthesizes a capture with bursts.

Usage:
    python examples/burst_detector.py
    python examples/burst_detector.py capture.iq --min-len 500
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import find_bursts, power_dbfs


def make_capture(fs, seed=0):
    """Silence with a few bursts of activity at random spots."""
    rng = np.random.default_rng(seed)
    n = 200_000
    sig = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    # insert 3 bursts of a tone
    for start, length in [(20_000, 8_000), (90_000, 4_000), (150_000, 12_000)]:
        t = np.arange(length) / fs
        sig[start:start + length] += np.exp(2j * np.pi * 50e3 * t)
    return sig.astype(np.complex64)


def main():
    p = argparse.ArgumentParser(description="Detect bursts in a capture.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--threshold", type=float, default=None,
                   help="envelope threshold (default: auto)")
    p.add_argument("--min-gap", type=int, default=500,
                   help="merge bursts closer than this many samples")
    p.add_argument("--min-len", type=int, default=1000,
                   help="discard bursts shorter than this")
    args = p.parse_args()

    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        iq = make_capture(fs)
        print(f"[*] synthetic capture @ {fs/1e6:g} Msps (3 bursts hidden)")

    bursts = find_bursts(iq, threshold=args.threshold, min_gap=args.min_gap,
                         min_len=args.min_len)
    print(f"[*] {len(bursts)} burst(s) found:\n")
    print(f"    {'#':>3} {'start (ms)':>12} {'dur (ms)':>10} {'power':>10}")
    print("    " + "-" * 38)
    for i, (s, e) in enumerate(bursts):
        seg = iq[s:e]
        print(f"    {i:>3} {s/fs*1e3:>12.2f} {(e-s)/fs*1e3:>10.2f} "
              f"{power_dbfs(seg):>7.1f} dB")
    if not bursts:
        print("    (none -- lower --threshold or --min-len, or no signal)")
    else:
        print(f"\n[*] extract burst 0 for decoding: iq[{bursts[0][0]}:"
              f"{bursts[0][1]}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
