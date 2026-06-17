#! /usr/bin/python3
"""dsss_demo.py -- despread a direct-sequence spread-spectrum signal.

DSSS hides data under a fast pseudo-random code, spreading it across a wide
band so it looks like noise -- until you correlate against the known code, which
collapses it back to the data and pushes interference down (processing gain).
This is how GPS and some IoT links work.

Shows: spread a message with a PN code, add noise/interference, then despread
with the known code and recover the bits. Also shows the "before" (looks like
noise) vs "after" (clean) so the processing gain is visible.

Library deps only (numpy). No hardware -- fully synthetic demonstration.

Usage:
    python examples/dsss_demo.py
    python examples/dsss_demo.py --snr-db -10 --code-len 31
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import dsss_despread, to_db


def pn_code(length, seed=1):
    """A simple +/-1 pseudo-noise code (random for the demo; real systems use
    structured sequences like Gold codes)."""
    rng = np.random.default_rng(seed)
    return (2 * rng.integers(0, 2, length) - 1).astype(np.complex64)


def main():
    p = argparse.ArgumentParser(description="DSSS despreading demo.")
    p.add_argument("--code-len", type=int, default=31)
    p.add_argument("--nbits", type=int, default=20)
    p.add_argument("--snr-db", type=float, default=-6,
                   help="SNR of the spread signal (can be negative!)")
    args = p.parse_args()

    code = pn_code(args.code_len)
    rng = np.random.default_rng(0)
    data = (2 * rng.integers(0, 2, args.nbits) - 1).astype(np.complex64)

    # spread: each data bit multiplies the whole code
    spread = np.concatenate([d * code for d in data]).astype(np.complex64)

    # bury it in noise at the requested SNR
    sig_power = np.mean(np.abs(spread) ** 2)
    snr_lin = 10 ** (args.snr_db / 10)
    noise_power = sig_power / snr_lin
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(spread))
                                        + 1j * rng.standard_normal(len(spread)))
    rxed = (spread + noise).astype(np.complex64)

    print(f"[*] {args.nbits} bits spread by a length-{args.code_len} code")
    print(f"[*] channel SNR: {args.snr_db} dB "
          f"(signal {'below' if args.snr_db < 0 else 'above'} noise)")

    # despread with the KNOWN code
    recovered = dsss_despread(rxed, code)
    bits_out = (np.real(recovered) > 0).astype(int)
    bits_in = (np.real(data) > 0).astype(int)
    n = min(len(bits_out), len(bits_in))
    errors = np.sum(bits_out[:n] != bits_in[:n])

    # processing gain: the despread SNR is better by ~10log10(code_len)
    gain_db = to_db(args.code_len)
    print(f"[*] processing gain from the code: +{gain_db:.1f} dB")
    print(f"[*] despread: {n - errors}/{n} bits correct")
    print(f"[*] recovered: {''.join(str(b) for b in bits_out[:n])}")
    print(f"[*] sent:      {''.join(str(b) for b in bits_in[:n])}")
    if errors == 0:
        print("[*] perfect recovery -- the code pulled the signal out of the "
              "noise")
    return 0


if __name__ == "__main__":
    sys.exit(main())
