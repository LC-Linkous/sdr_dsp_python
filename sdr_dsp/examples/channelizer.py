#! /usr/bin/python3
"""channelizer.py -- pull channels out of a wide capture, one or many.

Two jobs, two functions (both now live in the library core; this example shows
them):

  SINGLE -- channelize(): extract one channel at an arbitrary offset and
    bandwidth. tune -> filter -> decimate. For when you want one specific signal.

  BANK -- channelize_bank(): split the whole band into N equal channels at once
    with a polyphase filterbank, far cheaper than running the single extractor N
    times (one filter + an FFT instead of N mix-filter-decimate chains). For
    monitoring a whole slice -- every FM station, every ISM channel -- at once.

Library deps only (numpy); plotting is optional. Runs on a saved capture or a
synthesized multi-signal band.

Usage:
    python examples/channelizer.py wideband.iq --offset 250e3 --bw 100e3
    python examples/channelizer.py wideband.iq --bank 16
    python examples/channelizer.py                       # synthetic demo
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import channelize, channelize_bank, power_dbfs


def synth_band(fs, n=200000):
    """A wide band with several signals at different offsets."""
    t = np.arange(n) / fs
    sig = (np.exp(2j * np.pi * (0.30 * fs / 2) * t)            # high channel
           + 0.7 * np.exp(2j * np.pi * (-0.15 * fs / 2) * t)  # lower channel
           + 0.5 * np.exp(2j * np.pi * (0.02 * fs / 2) * t))  # near center
    sig += 0.02 * (np.random.randn(n) + 1j * np.random.randn(n))
    return sig.astype(np.complex64)


def main():
    p = argparse.ArgumentParser(description="Single or multi-channel extraction.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--offset", type=float, default=300e3, help="single: offset Hz")
    p.add_argument("--bw", type=float, default=100e3, help="single: bandwidth Hz")
    p.add_argument("--bank", type=int, default=None,
                   help="bank: split into this many channels instead")
    p.add_argument("--oversample", action="store_true",
                   help="bank: oversample by 2 (decim = N/2) for cleaner edges")
    args = p.parse_args()

    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        iq = synth_band(fs)
        print(f"[*] synthetic wide band @ {fs/1e6:g} Msps")

    if args.bank:
        # MULTI-CHANNEL: split the whole band at once
        N = args.bank
        decim = (N // 2) if args.oversample else N
        chans, rate, freqs = channelize_bank(iq, fs, N, decim=decim)
        print(f"[*] polyphase bank: {N} channels, each {rate/1e3:g} kHz wide "
              f"({'oversampled' if args.oversample else 'critically sampled'})")
        print(f"\n    {'ch':>3} {'center (kHz)':>14} {'power':>10}")
        print("    " + "-" * 32)
        for i, (f, c) in enumerate(zip(freqs, chans)):
            pw = power_dbfs(c)
            mark = "  <-- signal" if pw > -20 else ""
            print(f"    {i:>3} {f/1e3:>14.1f} {pw:>7.1f} dB{mark}")
        print(f"\n[*] each row is an IQ stream ready to demod or save")
    else:
        # SINGLE CHANNEL: one specific channel
        ch, rate = channelize(iq, fs, args.offset, args.bw)
        print(f"[*] extracted channel @ {args.offset/1e3:g} kHz, "
              f"{args.bw/1e3:g} kHz wide -> {rate/1e3:g} kHz rate, "
              f"{len(ch)} samples")
        print(f"[*] channel power: {power_dbfs(ch):.1f} dBFS")
        print(f"[*] ready to demod or save (use --bank N to split the whole band)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
