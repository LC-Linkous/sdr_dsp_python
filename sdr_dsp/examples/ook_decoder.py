#! /usr/bin/python3
"""ook_decoder.py -- decode an on-off-keyed signal (e.g. a 433 MHz key fob).

The full digital-demod chain on a real captured burst:
    load IQ -> envelope -> threshold (slice) -> recover symbol timing ->
    collapse to bits -> print the bitstream.

Capture the signal first (press your key fob while running a capture at
433.92 MHz), save it, then run this offline. See examples/live_433_capture.py
(or use hackrfpy directly) to make the recording.

Library deps only (numpy). No hardware -- runs on a saved capture.

Usage:
    python examples/ook_decoder.py keyfob_433.iq
    python examples/ook_decoder.py keyfob_433.iq --threshold 0.1
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from src.sdr_dsp.io import load_iq
from src.sdr_dsp.core import (
    ook_envelope, ook_slice, estimate_symbol_rate, slice_to_symbols, edges,
)


def main():
    p = argparse.ArgumentParser(description="Decode an OOK/ASK capture to bits.")
    p.add_argument("iq_file")
    p.add_argument("--threshold", type=float, default=None,
                   help="envelope slice level (default: auto midpoint)")
    p.add_argument("--smooth", type=int, default=0,
                   help="moving-average the envelope over N samples first")
    args = p.parse_args()

    iq, meta = load_iq(args.iq_file)
    fs = float(meta.get("global", {}).get("core:sample_rate", 0.0))
    print(f"[*] loaded {len(iq):,} samples @ {fs/1e6:g} Msps")

    # 1. envelope
    env = ook_envelope(iq)
    if args.smooth > 1:
        k = np.ones(args.smooth) / args.smooth
        env = np.convolve(env, k, mode="same")

    # 2. is there even a signal? compare on vs off levels
    hi, lo = np.percentile(env, 95), np.percentile(env, 5)
    if hi < lo * 1.5:
        print(f"[!] weak/absent modulation (hi {hi:.3f} vs lo {lo:.3f}); "
              f"is the burst in this capture?")
    print(f"[*] envelope: on~{hi:.3f}  off~{lo:.3f}")

    # 3. slice to a 0/1 stream
    bits = ook_slice(env, threshold=args.threshold)
    on_frac = float(np.mean(bits))
    print(f"[*] sliced: {on_frac*100:.1f}% on")

    # 4. recover symbol timing from the shortest pulse
    spb, sym_rate = estimate_symbol_rate(bits, fs)
    if spb <= 0:
        print("[!] no transitions found -- nothing to decode")
        return 1
    print(f"[*] symbol period ~ {spb:.0f} samples "
          f"({sym_rate/1e3:.2f} ksym/s)" if fs else
          f"[*] symbol period ~ {spb:.0f} samples")

    # 5. collapse to symbols/bits
    syms = slice_to_symbols(bits, spb)
    # trim leading/trailing idle (long runs of the idle level)
    _, runs, vals = edges(bits)
    print(f"[*] {len(syms)} symbols recovered\n")

    bitstr = "".join(str(b) for b in syms)
    # print in byte-ish groups for readability
    grouped = " ".join(bitstr[i:i + 8] for i in range(0, len(bitstr), 8))
    print("bitstream:")
    print(" ", grouped)

    # a few interpretations the user might want
    print("\n[*] tips: many fobs repeat the frame several times; look for the "
          "repeating unit. Try --smooth 20-100 if the envelope is noisy, or "
          "--threshold to tune the on/off decision.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
