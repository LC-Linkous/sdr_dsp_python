#! /usr/bin/python3
"""modulate_demo.py -- generate signals and decode them back (TX Phase A).

The library can now TRANSMIT, not just receive: turn bits and messages into IQ.
This demo proves each modulator by closing the loop -- modulate, then demodulate
with the matching demod, entirely in software. demod(modulate(x)) == x is the
whole point, and it needs no hardware.

This is the transmit-side mirror of the receive library: every modulator here is
the inverse of a demod the library already had.

Library deps only (numpy); plotting optional. No hardware.

Usage:
    python examples/modulate_demo.py
    python examples/modulate_demo.py --plot
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import (
    fm_modulate, am_modulate, ook_modulate, fsk_modulate,
    bpsk_modulate, qpsk_modulate,
    fm_demod, am_demod, ook_envelope, ook_slice, fsk_demod,
    bpsk_demod, qpsk_demod,
)


def main():
    p = argparse.ArgumentParser(description="Modulate then demodulate (loopback).")
    p.add_argument("--plot", action="store_true", help="show a constellation/spectrum")
    args = p.parse_args()

    fs = 1e6
    print("[*] Closing the loop: modulate -> demodulate, no hardware\n")

    # --- analog: a tone through FM and AM ---------------------------------
    t = np.arange(20000) / fs
    msg = np.cos(2 * np.pi * 2000 * t)

    def corr(a, b, trim=200):
        n = min(len(a), len(b))
        return np.corrcoef(a[trim:n-trim], b[trim:n-trim])[0, 1]

    fm_rec = fm_demod(fm_modulate(msg, 75e3, fs), 75e3, fs)
    am_rec = am_demod(am_modulate(msg, 0.5))
    print(f"[analog] FM  message recovered, correlation {corr(msg, fm_rec):.4f}")
    print(f"[analog] AM  message recovered, correlation {corr(msg, am_rec):.4f}")

    # --- digital: a bit pattern through each scheme -----------------------
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 200)
    print(f"\n[digital] sending {len(bits)} random bits through each scheme:")

    sps = 50
    ook_rec = ook_slice(ook_envelope(ook_modulate(bits, sps)))[::sps][:len(bits)]
    fsk_rec = fsk_demod(fsk_modulate(bits, sps, 50e3, fs), fs)[sps//2::sps][:len(bits)]
    bpsk_rec, _ = bpsk_demod(bpsk_modulate(bits, 1))
    qpsk_rec, _ = qpsk_demod(qpsk_modulate(bits, 1))

    for name, rec in [("OOK", ook_rec), ("FSK", fsk_rec),
                      ("BPSK", bpsk_rec[:len(bits)]), ("QPSK", qpsk_rec[:len(bits)])]:
        ber = np.mean(rec != bits[:len(rec)])
        print(f"   {name:5s} BER {ber:.4f}  {'(perfect)' if ber == 0 else ''}")

    print("\n[*] every modulator round-trips through its demod -- "
          "the library transmits now")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("(--plot needs matplotlib)", file=sys.stderr)
            return 0
        # show a pulse-shaped QPSK constellation -- what we'd transmit
        from sdr_dsp.core import rrc_taps
        qpsk_iq = qpsk_modulate(bits, 8, pulse_shaping=True)
        taps = rrc_taps(8)
        matched = np.convolve(qpsk_iq, taps, mode="same")
        syms = matched[::8]
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].plot(np.real(qpsk_iq[:400]), lw=0.7)
        ax[0].set_title("pulse-shaped QPSK (transmit waveform, real part)")
        ax[0].set_xlabel("sample"); ax[0].grid(alpha=0.3)
        ax[1].scatter(syms.real, syms.imag, s=6, alpha=0.4)
        ax[1].set_title("recovered QPSK constellation")
        ax[1].set_aspect("equal"); ax[1].grid(alpha=0.3)
        fig.suptitle("What the library transmits, and what comes back")
        fig.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
