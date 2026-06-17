#! /usr/bin/python3
"""inspect_capture.py -- "what did I just capture?"

Point this at any SigMF recording and it prints everything worth knowing before
you process it: format, rate, frequency, duration, power, DC offset, and whether
the ADC was clipping. The first tool to reach for when an example misbehaves --
nine times out of ten the answer is in the capture, not the DSP.

Library deps only (numpy). No hardware.

Usage:
    python examples/inspect_capture.py sample_data/fm_2Msps.iq
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.io import load_iq, read_meta
from sdr_dsp.core import power_dbfs


def main():
    p = argparse.ArgumentParser(description="Inspect a SigMF IQ capture.")
    p.add_argument("iq_file")
    p.add_argument("--max-samples", type=int, default=2_000_000,
                   help="cap how many samples to load for the stats")
    args = p.parse_args()

    meta = read_meta(args.iq_file)
    g = meta.get("global", {})
    caps = meta.get("captures", [{}])

    iq, _ = load_iq(args.iq_file, count=args.max_samples)
    n = len(iq)
    fs = float(g.get("core:sample_rate", 0.0))
    fc = float(caps[0].get("core:frequency", 0.0)) if caps else 0.0

    # core stats
    mag = np.abs(iq)
    dc_i = float(np.mean(iq.real))
    dc_q = float(np.mean(iq.imag))
    peak = float(mag.max()) if n else 0.0
    # "clipping": magnitude pressed against full scale (1.0 after normalize)
    clip_frac = float(np.mean(mag > 0.98)) if n else 0.0

    print(f"file              : {args.iq_file}")
    print(f"datatype          : {g.get('core:datatype', '?')}")
    print(f"samples loaded     : {n:,}" + ("" if n < args.max_samples
                                            else "  (capped)"))
    if fs > 0:
        print(f"sample rate       : {fs/1e6:g} Msps")
        print(f"duration          : {n/fs*1e3:.1f} ms")
    print(f"center frequency  : {fc/1e6:g} MHz" if fc else "center frequency  : (unset)")
    print(f"mean power        : {power_dbfs(iq):.1f} dBFS")
    print(f"peak |amplitude|  : {peak:.3f}  (1.0 = ADC full scale)")
    print(f"DC offset (I, Q)  : ({dc_i:+.4f}, {dc_q:+.4f})"
          + ("   <- significant; expect a center spike" if abs(dc_i) > 0.02
             or abs(dc_q) > 0.02 else ""))
    if clip_frac > 0.001:
        print(f"CLIPPING          : {clip_frac*100:.1f}% of samples near full "
              f"scale -- reduce gain")
    else:
        print(f"clipping          : none detected")

    # a quick spectral peek: where's the energy?
    if n >= 1024:
        from sdr_dsp.core import psd
        freqs, p_db = psd(iq, fs or 1.0, nfft=1024, center_freq=fc)
        k = int(np.argmax(p_db))
        units = "MHz" if fs else "norm"
        fpk = freqs[k] / 1e6 if fs else freqs[k]
        print(f"spectral peak     : {p_db[k]:.1f} dB @ {fpk:.3f} {units}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
