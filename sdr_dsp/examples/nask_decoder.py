#! /usr/bin/python3
"""nask_decoder.py -- decode multi-level amplitude-shift keying (4-ASK).

OOK is 2-level ASK (on/off). N-ASK uses several amplitude levels to carry more
bits per symbol -- 4-ASK carries 2 bits/symbol. The chain:
    load IQ -> envelope -> slice into N levels -> symbols.

Shows 4-ASK by default; --levels sets the count. Demonstrates why more levels
means more bits but tighter spacing (more noise-sensitive).

Library deps only (numpy). No hardware -- synthesizes a demo signal.

Usage:
    python examples/nask_decoder.py
    python examples/nask_decoder.py --levels 8 --snr-db 25
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import ook_envelope, nask_slice, estimate_symbol_rate


def make_demo(fs, levels, nsym, spb, snr_db, seed=0):
    rng = np.random.default_rng(seed)
    syms = rng.integers(0, levels, nsym)
    # amplitudes evenly spaced from 0.2 to 1.0
    amps = np.linspace(0.2, 1.0, levels)
    sig = np.repeat([amps[s] for s in syms], spb).astype(np.complex64)
    sig_p = np.mean(np.abs(sig) ** 2)
    npow = sig_p / (10 ** (snr_db / 10))
    sig += np.sqrt(npow / 2) * (rng.standard_normal(len(sig))
                                + 1j * rng.standard_normal(len(sig)))
    return sig.astype(np.complex64), syms, amps


def main():
    p = argparse.ArgumentParser(description="Decode N-ASK.")
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--rate", type=float, default=1e6)
    p.add_argument("--nsym", type=int, default=40)
    p.add_argument("--snr-db", type=float, default=30)
    args = p.parse_args()

    spb = 200
    fs = args.rate
    sig, truth, amps = make_demo(fs, args.levels, args.nsym, spb, args.snr_db)
    print(f"[*] synthetic {args.levels}-ASK @ {fs/1e6:g} Msps, "
          f"{args.snr_db} dB SNR")
    print(f"[*] amplitude levels: {[round(a,2) for a in amps]}")

    env = ook_envelope(sig)
    # smooth a touch so per-sample noise doesn't split symbols
    env = np.convolve(env, np.ones(20) / 20, mode="same")

    # slice into N levels (explicit levels = the known amplitudes; here we let
    # it auto-spread for the demo, which is the honest default)
    per_sample = nask_slice(env, n_levels=args.levels)

    # collapse to symbols at centers (we know spb for the demo)
    syms = np.array([int(np.round(np.median(
        per_sample[int((i + 0.2) * spb):int((i + 0.8) * spb)])))
        for i in range(args.nsym)])

    n = min(len(syms), len(truth))
    errs = sum(int(a) != int(b) for a, b in zip(syms[:n], truth[:n]))
    print(f"[*] {n - errs}/{n} symbols correct ({args.levels} levels = "
          f"{int(np.log2(args.levels))} bits/symbol)")
    print(f"[*] recovered: {''.join(str(int(s)) for s in syms[:40])}")
    print(f"[*] sent:      {''.join(str(int(s)) for s in truth[:40])}")
    if errs > 0:
        print("    (more levels = tighter spacing = more noise-sensitive; "
              "raise --snr-db)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
