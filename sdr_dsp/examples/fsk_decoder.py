#! /usr/bin/python3
"""fsk_decoder.py -- decode FSK (the ISM-band workhorse).

FSK encodes bits as two (or more) frequencies. It's everywhere on 433/915 MHz:
weather stations, TPMS, IoT sensors, pagers. The chain:
    load IQ -> instantaneous frequency -> slice to levels -> timing -> bits.

Handles 2-FSK (default) and N-level FSK (4-FSK for DMR/P25) via --levels.

Library deps only (numpy). No hardware -- runs on a saved capture, or
synthesizes a demo signal if no file is given.

Usage:
    python examples/fsk_decoder.py capture.iq
    python examples/fsk_decoder.py capture.iq --levels 4
    python examples/fsk_decoder.py                    # synthetic demo
"""
import argparse
import sys

import numpy as np

from sdr_dsp.core import (fsk_demod, fsk_demod_nlevel, estimate_symbol_rate,
                          slice_to_symbols, instantaneous_frequency)

def make_demo(fs, levels):
    rng = np.random.default_rng(0)
    nsym = 40
    spb = 200
    if levels == 2:
        syms = rng.integers(0, 2, nsym)
        freqs = [-50e3, 50e3]
    else:
        syms = rng.integers(0, levels, nsym)
        freqs = list(np.linspace(-75e3, 75e3, levels))
    parts = [np.exp(2j * np.pi * freqs[s] * np.arange(spb) / fs) for s in syms]
    iq = np.concatenate(parts).astype(np.complex64)
    iq += 0.05 * (rng.standard_normal(len(iq))
                  + 1j * rng.standard_normal(len(iq)))
    return iq, syms

def main():
    p = argparse.ArgumentParser(description="Decode an FSK capture.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--levels", type=int, default=2, help="2 for 2-FSK, 4, ...")
    p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--spb", type=float, default=None, help="samples per symbol (for N-FSK)")
    args = p.parse_args()

    truth = None
    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        iq, truth = make_demo(fs, args.levels)
        print(f"[*] synthetic {args.levels}-FSK demo @ {fs/1e6:g} Msps")

    print(f"[*] {len(iq):,} samples; demodulating {args.levels}-FSK")

    # show the frequency excursion (this is what FSK looks like)
    inst = instantaneous_frequency(iq, sample_rate=fs)
    print(f"[*] frequency swing: {np.percentile(inst,2)/1e3:.0f} .. "
          f"{np.percentile(inst,98)/1e3:.0f} kHz")

    # Demodulate with the LIBRARY, not a hand-rolled discriminator: fsk_demod
    # smooths the instantaneous frequency over ~half a symbol (a cheap
    # matched-filter stand-in) before slicing, which is what makes decoding
    # work at moderate SNR. smooth_win is a modest fixed window -- we don't
    # know the symbol rate yet, enough to kill per-sample transients without
    # blurring real transitions.
    smooth_win = 20

    if args.levels == 2:
        # 'auto' threshold self-centers on a carrier offset (see fsk_demod);
        # harmless here where the tones straddle 0 and correct for real radios
        bits = fsk_demod(iq, fs, threshold_hz="auto", smooth_samples=smooth_win)
        spb, rate = estimate_symbol_rate(bits, fs, min_run=3)
        syms = slice_to_symbols(bits, spb) if spb > 0 else bits
    else:
        # N-level timing is harder: a 2-level slice of a multi-level signal
        # doesn't align transitions with symbols, so blind rate estimation is
        # unreliable. Honest approach: use a provided --spb (you usually know
        # the symbol rate of a protocol you're decoding). Default assumes the
        # demo's rate.
        spb = args.spb if args.spb else 200.0
        rate = fs / spb
        # Demodulate N-level with the LIBRARY: smooth_samples pre-smooths the
        # instantaneous frequency (percentile band centers are then computed
        # from the smoothed signal too), exactly what the 2-level path gets
        # from fsk_demod. Read each symbol's level at its CENTER; boundary
        # samples straddle bands and misclassify.
        persample = fsk_demod_nlevel(iq, fs, n_levels=args.levels,
                                     smooth_samples=smooth_win)
        nsym = int(len(persample) / spb)
        syms = np.array([int(persample[int((i + 0.5) * spb)])
                         for i in range(nsym)
                         if int((i + 0.5) * spb) < len(persample)])

    if spb > 0:
        print(f"[*] symbol rate ~ {rate/1e3:.2f} ksym/s ({spb:.0f} samples/sym)")
    print(f"[*] {len(syms)} symbols recovered:")
    print("   ", "".join(str(int(s)) for s in syms[:80]))

    if truth is not None:
        n = min(len(syms), len(truth))
        # align (recovery may offset by a symbol) and report agreement
        errs = sum(int(a) != int(b) for a, b in zip(syms[:n], truth[:n]))
        print(f"[*] vs known truth: {n-errs}/{n} symbols match")
    return 0

if __name__ == "__main__":
    sys.exit(main())
