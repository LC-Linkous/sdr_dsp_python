#! /usr/bin/python3
"""differential_psk_demo.py -- DBPSK / DQPSK: phase decoding without carrier recovery.

Coherent PSK (bpsk/qpsk) needs the carrier recovered first -- a loop that locks
onto the absolute phase. Differential PSK sidesteps that entirely: it encodes
bits in the phase CHANGE between consecutive symbols, so a constant phase offset
(the thing carrier recovery exists to remove) cancels out when you compare
neighbors. That makes it robust and a great fit for block processing.

This demo proves the property: it applies a big arbitrary phase offset to the
signal -- the kind that would scramble coherent BPSK -- and shows differential
decoding recovers the bits anyway, with NO carrier recovery step.

Library deps only (numpy). No hardware -- synthetic demonstration.

Usage:
    python examples/differential_psk_demo.py --mod dbpsk
    python examples/differential_psk_demo.py --mod dqpsk --offset 2.5
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import dbpsk_demod, dqpsk_demod


def make_dbpsk(nbits, offset, noise, seed=0):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, nbits)
    phase = 0.0
    syms = []
    for b in bits:
        if b:
            phase += np.pi              # differential: flip phase on a 1
        syms.append(np.exp(1j * phase))
    syms = np.array(syms, dtype=np.complex64) * np.exp(1j * offset)
    syms += noise * (rng.standard_normal(len(syms))
                     + 1j * rng.standard_normal(len(syms)))
    return syms.astype(np.complex64), bits


def make_dqpsk(nsym, offset, noise, seed=0):
    rng = np.random.default_rng(seed)
    sym_bits = [(int(rng.integers(0, 2)), int(rng.integers(0, 2)))
                for _ in range(nsym)]
    step = {(0, 0): 0, (0, 1): np.pi / 2, (1, 1): np.pi, (1, 0): -np.pi / 2}
    phase = 0.0
    syms = [1 + 0j]
    for b in sym_bits:
        phase += step[b]
        syms.append(np.exp(1j * phase))
    syms = np.array(syms, dtype=np.complex64) * np.exp(1j * offset)
    syms += noise * (rng.standard_normal(len(syms))
                     + 1j * rng.standard_normal(len(syms)))
    tx_bits = [b for pair in sym_bits for b in pair]
    return syms.astype(np.complex64), tx_bits


def main():
    p = argparse.ArgumentParser(description="Differential PSK demo.")
    p.add_argument("--mod", choices=["dbpsk", "dqpsk"], default="dbpsk")
    p.add_argument("--offset", type=float, default=2.0,
                   help="arbitrary phase offset (rad) -- DPSK should ignore it")
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--nsym", type=int, default=200)
    args = p.parse_args()

    print(f"[*] {args.mod.upper()} with a {args.offset:.1f} rad phase offset")
    print("    (this offset would scramble coherent BPSK/QPSK; differential "
          "ignores it)")

    if args.mod == "dbpsk":
        syms, tx = make_dbpsk(args.nsym, args.offset, args.noise)
        bits, soft = dbpsk_demod(syms)
        # differential decoding yields n-1 bits (first symbol is the reference)
        expected = tx[1:]
    else:
        syms, tx = make_dqpsk(args.nsym, args.offset, args.noise)
        bits, ang = dqpsk_demod(syms)
        expected = tx

    n = min(len(bits), len(expected))
    errors = int(np.sum(np.array(bits[:n]) != np.array(expected[:n])))
    print(f"[*] decoded {n} bits with NO carrier recovery")
    print(f"[*] {n - errors}/{n} bits correct "
          f"({'perfect' if errors == 0 else f'{errors} errors'})")
    if errors == 0:
        print("[*] the constant phase offset cancelled -- that's the whole "
              "point of differential PSK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
