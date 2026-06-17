#! /usr/bin/python3
"""ssb_receiver.py -- demodulate single-sideband (SSB) voice to audio.

SSB is how ham, marine, and aviation HF voice is transmitted: one sideband of an
AM signal, carrier suppressed -- efficient, but it sounds like a duck until you
demodulate the right sideband. The chain:
    load IQ -> (tune) -> SSB demod (USB or LSB) -> resample -> WAV.

Selecting the wrong sideband gives garbled audio, which is itself a good
demonstration of why sideband choice matters.

Library deps only (numpy). No hardware -- runs on a capture, or synthesizes a
USB test signal.

Usage:
    python examples/ssb_receiver.py capture.iq --sideband usb
    python examples/ssb_receiver.py capture.iq --sideband lsb --tune -1500
"""
import argparse
import sys
import wave
from math import gcd

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import ssb_demod, resample_poly, frequency_shift, normalize

AUDIO_RATE = 48000


def synth_ssb(fs, sideband="usb"):
    """A simple USB/LSB test signal: a couple of audio tones on one sideband.

    A proper SSB signal is the analytic signal of the audio (USB = the audio's
    positive-frequency content; LSB = its conjugate). This is exactly what an
    SSB transmitter produces after the carrier and one sideband are removed.
    """
    n = 200_000
    t = np.arange(n) / fs
    audio = (np.cos(2 * np.pi * 800 * t) + 0.6 * np.cos(2 * np.pi * 1900 * t))
    # analytic signal: real audio -> complex with only positive frequencies
    spec = np.fft.fft(audio)
    spec[n // 2:] = 0           # zero negative frequencies -> analytic (USB)
    analytic = 2 * np.fft.ifft(spec)
    iq = analytic.astype(np.complex64)
    if sideband == "lsb":
        iq = np.conj(iq)        # flip to the lower sideband
    iq += 0.02 * (np.random.randn(n) + 1j * np.random.randn(n))
    return iq.astype(np.complex64)


def main():
    p = argparse.ArgumentParser(description="SSB receiver -> WAV.")
    p.add_argument("iq_file", nargs="?", default=None)
    p.add_argument("--sideband", choices=["usb", "lsb"], default="usb")
    p.add_argument("--tune", type=float, default=0.0, help="tune offset Hz")
    p.add_argument("--bfo", type=float, default=0.0, help="BFO fine-tune Hz")
    p.add_argument("--rate", type=float, default=192000)
    p.add_argument("--out", default="ssb_audio.wav")
    args = p.parse_args()

    if args.iq_file:
        from sdr_dsp.io import load_iq
        iq, meta = load_iq(args.iq_file)
        fs = float(meta.get("global", {}).get("core:sample_rate", args.rate))
    else:
        fs = args.rate
        iq = synth_ssb(fs, args.sideband)
        print(f"[*] synthetic {args.sideband.upper()} test signal @ "
              f"{fs/1e3:g} kHz")

    if args.tune:
        iq = frequency_shift(iq, -args.tune, fs)
        print(f"[*] tuned {args.tune/1e3:g} kHz")

    # demod the chosen sideband
    audio = ssb_demod(iq, fs, sideband=args.sideband, bfo_hz=args.bfo)
    print(f"[*] {args.sideband.upper()} demodulated: {len(audio):,} samples")

    # resample to audio rate
    g = gcd(AUDIO_RATE, int(fs))
    audio = resample_poly(audio, AUDIO_RATE // g, int(fs) // g)
    audio = normalize(audio, mode="peak", target=0.9)

    pcm = np.int16(np.clip(audio, -1, 1) * 32767)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.tobytes())
    print(f"[*] wrote {args.out}: {len(pcm)/AUDIO_RATE:.1f}s")
    print("    (try the wrong --sideband to hear why it matters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
