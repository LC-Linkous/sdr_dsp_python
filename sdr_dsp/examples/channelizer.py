#! /usr/bin/python3
"""channelizer.py -- pull one narrow channel out of a wide capture.

The capstone that ties three primitives together:
    tune (mixing) -> lowpass (filter) -> decimate (resample).
Given a wide capture and a target offset, it shifts that channel to baseband,
filters to the channel bandwidth, and decimates to a lower rate -- giving you
just that one channel at a manageable sample rate, ready to demod or save.

Library deps only (numpy). No hardware -- runs on a saved capture. Can save the
extracted channel back out as its own SigMF recording.

Usage:
    python examples/channelizer.py wideband.iq --offset 250e3 --bw 100e3
    python examples/channelizer.py wideband.iq --offset 250e3 --bw 100e3 \
        --save channel.sigmf-data
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.sources import FileSource
from sdr_dsp.core import (
    tune_to_baseband, design_lowpass, fir_apply, decimate, psd,
)


def channelize(iq, sample_rate, offset_hz, channel_bw, decim=None):
    """Extract the channel at offset_hz with bandwidth channel_bw.

    Returns (channel_iq, new_sample_rate). decim defaults to the largest
    integer that keeps the channel comfortably inside the new Nyquist.
    """
    # 1. tune the channel to baseband
    base = tune_to_baseband(iq, offset_hz, sample_rate)
    # 2. lowpass to the channel half-bandwidth
    taps = design_lowpass(channel_bw / 2, sample_rate, num_taps=201)
    filt = fir_apply(base, taps)
    # 3. decimate -- keep the new rate >= ~2.5x the channel bandwidth
    if decim is None:
        decim = max(1, int(sample_rate / (channel_bw * 2.5)))
    out = decimate(filt, decim)
    return out, sample_rate / decim


def main():
    p = argparse.ArgumentParser(description="Channelizer: extract one channel.")
    p.add_argument("iq_file")
    p.add_argument("--offset", type=float, required=True,
                   help="channel offset from capture center (Hz)")
    p.add_argument("--bw", type=float, default=100e3,
                   help="channel bandwidth to keep (Hz)")
    p.add_argument("--decim", type=int, default=None,
                   help="decimation factor (default: auto)")
    p.add_argument("--save", default=None, help="save channel as SigMF")
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    src = FileSource(args.iq_file)
    fs = src.sample_rate
    print(f"[*] {src}")
    print(f"[*] extracting channel at {args.offset/1e3:+g} kHz, "
          f"{args.bw/1e3:g} kHz wide")

    chan, new_fs = channelize(src.iq, fs, args.offset, args.bw, args.decim)
    print(f"[*] channel: {len(chan):,} samples @ {new_fs/1e3:g} ksps "
          f"(decimated {int(fs/new_fs)}x)")

    if args.save:
        from sdr_dsp.io import save_iq
        new_center = src.center_freq + args.offset
        dp, mp = save_iq(args.save, chan, new_fs, center_freq=new_center)
        print(f"[*] saved -> {dp}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("(install matplotlib to plot)")
            return 0
        f_wide, p_wide = psd(src.iq, fs, nfft=2048, center_freq=0)
        f_chan, p_chan = psd(chan, new_fs, nfft=1024, center_freq=0)
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8))
        a1.plot(f_wide / 1e3, p_wide, lw=0.7)
        a1.axvline(args.offset / 1e3, color="r", ls="--", lw=0.8,
                   label="target channel")
        a1.set_title("wide capture (red = extracted channel)")
        a1.set_xlabel("offset (kHz)")
        a1.set_ylabel("PSD (dB)")
        a1.legend()
        a1.grid(alpha=0.3)
        a2.plot(f_chan / 1e3, p_chan, lw=0.8, color="#2ca02c")
        a2.set_title(f"extracted channel @ {new_fs/1e3:g} ksps")
        a2.set_xlabel("baseband frequency (kHz)")
        a2.set_ylabel("PSD (dB)")
        a2.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
