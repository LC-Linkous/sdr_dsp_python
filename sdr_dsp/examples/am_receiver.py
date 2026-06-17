#! /usr/bin/python3
"""am_receiver.py -- AM demodulation: capture file -> audio WAV.

The AM counterpart to fm_receiver.py. AM carries the message in the signal's
amplitude, so demodulation is just the envelope (magnitude) with the DC carrier
removed. The chain:
    load IQ -> (tune) -> lowpass to the channel -> AM envelope demod ->
    resample to audio -> write WAV.

Good targets: AM broadcast (530-1700 kHz, needs an upconverter or a capture),
aircraft band (~118-137 MHz AM), or any amplitude-modulated capture.

Library deps only (numpy). No hardware -- runs on a saved capture.

Usage:
    python examples/am_receiver.py capture.iq --out am_audio.wav
    python examples/am_receiver.py capture.iq --tune -50e3 --audio-bw 8e3
"""
import argparse
import sys
import wave

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.sources import FileSource
from sdr_dsp.core import (
    design_lowpass, fir_apply, am_demod, resample_poly, frequency_shift,
)

AUDIO_RATE = 48_000


def main():
    p = argparse.ArgumentParser(description="AM receiver: IQ file -> WAV.")
    p.add_argument("iq_file")
    p.add_argument("--out", default="am_audio.wav")
    p.add_argument("--tune", type=float, default=0.0,
                   help="tune offset Hz if station isn't at capture center")
    p.add_argument("--audio-bw", type=float, default=8_000,
                   help="AM channel half-bandwidth (Hz); AM voice is narrow")
    args = p.parse_args()

    src = FileSource(args.iq_file)
    print(f"[*] {src}")
    fs = src.sample_rate
    iq = src.iq
    if fs <= 0:
        print("error: sample rate missing from metadata", file=sys.stderr)
        return 1

    # tune the station to baseband if offset in the capture
    if args.tune != 0.0:
        print(f"[*] tuning {args.tune/1e3:g} kHz to baseband")
        iq = frequency_shift(iq, -args.tune, fs)

    # lowpass to the (narrow) AM channel
    taps = design_lowpass(args.audio_bw, fs, num_taps=201)
    iq = fir_apply(iq, taps)
    print(f"[*] filtered to +/-{args.audio_bw/1e3:g} kHz channel")

    # AM demod: envelope with DC (carrier) removed
    audio = am_demod(iq, dc_block=True)
    print(f"[*] demodulated: {len(audio):,} samples")

    # resample to audio rate
    from math import gcd
    g = gcd(int(AUDIO_RATE), int(fs))
    up, down = int(AUDIO_RATE) // g, int(fs) // g
    print(f"[*] resampling {fs/1e6:g} Msps -> {AUDIO_RATE/1e3:g} kHz "
          f"(up={up}, down={down})")
    audio = resample_poly(audio, up, down)

    # normalize to int16 WAV
    peak = np.max(np.abs(audio)) or 1.0
    pcm = np.int16(np.clip(audio / peak * 0.9, -1, 1) * 32767)
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.tobytes())
    print(f"[*] wrote {args.out}: {len(pcm)/AUDIO_RATE:.1f}s @ "
          f"{AUDIO_RATE/1e3:g} kHz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
