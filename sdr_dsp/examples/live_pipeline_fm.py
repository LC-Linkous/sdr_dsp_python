#! /usr/bin/python3
"""live_pipeline_fm.py -- streaming FM demod with a live display tap.

Shows the recommended streaming pattern: build a Pipeline from the STATEFUL
streaming stages (filter -> FM demod -> resample, each carrying its delay line
across blocks so the seams are silent), and attach a TAP that updates a live
display each block -- here a scrolling text power/level meter, but it could
just as easily drive a matplotlib constellation or a decoded-message readout.

Why the Streaming* stages and not `lambda b: fir_apply(b, taps)`: the core
functions are stateless, so per-block calls restart every filter from empty
taps and difference the first FM sample against nothing -- a click at every
block boundary. The stages wrap the same core functions and only add the
bookkeeping; the output equals the whole record processed at once.

The whole point: "show it live" is a tap on a normal pipeline, not a separate
engine. The same Pipeline runs on a file (this example) or a HackRF (swap the
source for HackRFCapture from hackrf_capture.py).

Library deps only for the file version (numpy). No hardware.

Usage:
    python examples/live_pipeline_fm.py                 # synthetic FM
    python examples/live_pipeline_fm.py capture.iq
"""
import argparse
import sys
import time

import numpy as np

from sdr_dsp import Pipeline
from sdr_dsp.sources import ArraySource, FileSource
from sdr_dsp.core import design_lowpass, to_db
from sdr_dsp.stream import (StreamingFIR, StreamingDecimator, StreamingFMDemod,
                            StreamingResampler)

AUDIO_RATE = 48_000

def make_synth_fm(fs, seconds=2.0):
    n = int(fs * seconds)
    t = np.arange(n) / fs
    # a 2 kHz tone, warbling so the level meter visibly moves
    msg = np.cos(2 * np.pi * 2000 * t) * (0.5 + 0.5 * np.cos(2 * np.pi * 3 * t))
    return np.exp(1j * 2 * np.pi * 75000 * np.cumsum(msg) / fs).astype(
        np.complex64)

def meter_bar(level_db, lo=-60, hi=0, width=40):
    frac = max(0.0, min(1.0, (level_db - lo) / (hi - lo)))
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {level_db:6.1f} dB"

def main():
    p = argparse.ArgumentParser(description="Streaming FM with a live tap.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--rate", type=float, default=1_000_000)
    p.add_argument("--block", type=int, default=50_000)
    args = p.parse_args()

    if args.iq_file:
        source = FileSource(args.iq_file, block_size=args.block)
        fs = source.sample_rate
    else:
        fs = args.rate
        source = ArraySource(make_synth_fm(fs), fs, block_size=args.block)
        print(f"[*] synthetic FM @ {fs/1e6:g} Msps, block {args.block:,}")

    # Channel filter at the capture rate, then decimate to an intermediate
    # rate (~250 kHz, still well above the 200 kHz FM channel) BEFORE the
    # discriminator and the audio resampler. Without this the resampler runs
    # at the full capture rate and is the slowest stage by far.
    taps = design_lowpass(100e3, fs, num_taps=101)
    dec = max(1, int(fs) // 250_000)
    inter = fs / dec
    resampler = StreamingResampler(AUDIO_RATE, int(inter))  # reduces the ratio itself

    # the live display tap: print a level meter for each processed block
    def display(audio_block):
        rms = float(np.std(audio_block)) or 1e-12
        sys.stdout.write("\r  live audio " + meter_bar(to_db(rms, power=False)))
        sys.stdout.flush()
        time.sleep(0.02)   # just to make the live update visible in a demo

    pipe = (Pipeline(source)
            .add(StreamingFIR(taps), "filter")
            .add(StreamingDecimator(dec), "decimate")
            .add(StreamingFMDemod(75000, inter), "fm_demod")
            .add(resampler, "resample")
            .tap(display, "level_meter"))

    print("[*] pipeline:")
    print(pipe.describe())
    print("[*] streaming (live meter updates per block):")
    results, stats = pipe.run(profile=True)
    results.append(resampler.flush())   # the look-ahead tail, once the stream ends
    print()   # newline after the live meter
    print(f"[*] done: {stats.blocks} blocks, "
          f"{sum(len(r) for r in results):,} audio samples")
    print("[*] per-stage timing:")
    print(stats)
    return 0

if __name__ == "__main__":
    sys.exit(main())
