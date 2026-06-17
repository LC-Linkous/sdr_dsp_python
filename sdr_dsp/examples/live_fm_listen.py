#! /usr/bin/python3
"""live_fm_listen.py -- tune an FM station and play it through your speakers.

The live counterpart to fm_receiver.py: instead of writing a WAV, it streams
from the HackRF, demodulates block by block, and plays the audio in real time.
Same DSP chain, a live source and an audio sink.

Requires hackrfpy + hackrf-tools AND sounddevice (examples extras):
    pip install hackrfpy sounddevice

Usage:
    python examples/live_fm_listen.py --freq 96.5e6
    python examples/live_fm_listen.py --freq 101.1e6 --rate 2e6
"""
import argparse
import sys
from math import gcd

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "examples")
from sdr_dsp.core import design_lowpass, fir_apply, fm_demod, resample_poly

FM_DEVIATION = 75_000
AUDIO_RATE = 48_000


def main():
    p = argparse.ArgumentParser(description="Live FM receiver -> speakers.")
    p.add_argument("--freq", type=float, required=True, help="station Hz")
    p.add_argument("--rate", type=float, default=2e6)
    p.add_argument("--audio-bw", type=float, default=100e3)
    p.add_argument("--lna", type=int, default=16)
    p.add_argument("--vga", type=int, default=20)
    args = p.parse_args()

    try:
        from hackrf_capture import HackRFCapture
    except ImportError as e:
        print(f"needs hackrfpy: pip install hackrfpy  ({e})", file=sys.stderr)
        return 1
    try:
        import sounddevice as sd
    except ImportError:
        print("needs sounddevice: pip install sounddevice", file=sys.stderr)
        return 1

    fs = args.rate
    taps = design_lowpass(args.audio_bw, fs, num_taps=101)
    g = gcd(int(AUDIO_RATE), int(fs))
    up, down = int(AUDIO_RATE) // g, int(fs) // g

    print(f"[*] tuning {args.freq/1e6:g} MHz; Ctrl-C to stop")
    stream = sd.OutputStream(samplerate=AUDIO_RATE, channels=1, dtype="float32")
    stream.start()

    try:
        with HackRFCapture(args.freq, fs, lna=args.lna, vga=args.vga,
                           block_size=int(fs // 10)) as src:
            for iq in src.blocks():
                if len(iq) < len(taps) * 2:
                    continue
                filt = fir_apply(iq, taps)
                audio = fm_demod(filt, deviation_hz=FM_DEVIATION,
                                 sample_rate=fs)
                audio = resample_poly(audio, up, down)
                # normalize softly and play
                peak = np.max(np.abs(audio)) or 1.0
                stream.write((audio / peak * 0.7).astype(np.float32))
    except KeyboardInterrupt:
        print("\n[*] stopped")
    finally:
        stream.stop()
        stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
