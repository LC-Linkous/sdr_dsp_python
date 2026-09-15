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
from math import gcd


from sdr_dsp.sources import FileSource
from sdr_dsp.sinks import write_wav
from sdr_dsp.core import (
    design_lowpass, fir_apply, am_demod, resample_poly, frequency_shift,
    capture_health,
)

AUDIO_RATE = 48_000
CHANNEL_TAPS = 201
AUDIO_SETTLE_S = 0.005

def main():
    p = argparse.ArgumentParser(description="AM receiver: IQ file -> WAV.")
    p.add_argument("iq_file")
    p.add_argument("--out", default="am_audio.wav")
    p.add_argument("--tune", type=float, default=0.0,
                   help="tune offset Hz if station isn't at capture center")
    p.add_argument("--audio-bw", type=float, default=8_000,
                   help="AM channel half-bandwidth (Hz); AM voice is narrow")
    p.add_argument("--no-check", action="store_true",
                   help="skip the capture health check")
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

    # is there anything here to demodulate?
    if not args.no_check:
        health = capture_health(iq, fs, channel_bw=args.audio_bw)
        if health["ok"]:
            print(f"[*] capture peaks at ~{health['adc_counts']:.0f} of 127 "
                  f"ADC counts, channel "
                  f"{health['channel_excess_db']:+.1f} dB above the band edges")
        else:
            for reason in health["reasons"]:
                print(f"[!] {reason}")
            print("[!] continuing anyway -- expect noise in the output WAV.")

    # lowpass to the (narrow) AM channel
    taps = design_lowpass(args.audio_bw, fs, num_taps=CHANNEL_TAPS)
    iq = fir_apply(iq, taps)
    print(f"[*] filtered to +/-{args.audio_bw/1e3:g} kHz channel")

    # Drop the filter's startup transient. fir_apply is causal, so the first
    # (len(taps) - 1) samples are convolved against an empty delay line. The
    # envelope detector turns that ramp into a burst well above the program
    # material, and normalizing to it would bury the audio near zero.
    iq = iq[CHANNEL_TAPS:]

    # AM demod: envelope with DC (carrier) removed
    audio = am_demod(iq, dc_block=True)
    print(f"[*] demodulated: {len(audio):,} samples")

    # resample to audio rate
    g = gcd(int(AUDIO_RATE), int(fs))
    up, down = int(AUDIO_RATE) // g, int(fs) // g
    print(f"[*] resampling {fs/1e6:g} Msps -> {AUDIO_RATE/1e3:g} kHz "
          f"(up={up}, down={down})")
    audio = resample_poly(audio, up, down)

    # drop the resampler's settling region before scaling
    settle = int(AUDIO_SETTLE_S * AUDIO_RATE)
    if len(audio) > 4 * settle:
        audio = audio[settle:]
    if audio.size == 0:
        print("error: no audio left after filtering; capture too short",
              file=sys.stderr)
        return 1

    # write_wav normalizes to a high percentile rather than the raw peak, so
    # any residual impulse cannot crush the program material
    write_wav(args.out, audio, AUDIO_RATE)
    print(f"[*] wrote {args.out}: {len(audio)/AUDIO_RATE:.1f}s @ "
          f"{AUDIO_RATE/1e3:g} kHz")
    return 0

if __name__ == "__main__":
    sys.exit(main())
