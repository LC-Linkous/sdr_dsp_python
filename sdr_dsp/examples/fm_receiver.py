#! /usr/bin/python3
"""FM broadcast receiver: capture file -> audio WAV. The end-to-end demo.

Exercises the whole sdr_dsp core on a real wideband-FM recording:
    load IQ  ->  (tune)  ->  lowpass to the station  ->  FM demod  ->
    de-emphasis  ->  resample to 48 kHz  ->  write WAV you can play.

File-based: no hardware needed. Point it at a hackrfpy FM capture (ci8 SigMF).

Usage:
    python examples/fm_receiver.py sample_data/fm_2Msps.iq --out station.wav
    python examples/fm_receiver.py capture.iq --tune -250e3   # tune off-center
"""
import argparse
import sys
import wave

import numpy as np

# allow running from the repo without install
sys.path.insert(0, "src")

from sdr_dsp.sources import FileSource
from sdr_dsp.core import (
    design_lowpass, fir_apply, fm_demod, resample_poly, frequency_shift,
    deemphasis,
)


# FM broadcast parameters
FM_DEVIATION = 75_000          # max deviation of broadcast FM (Hz)
AUDIO_RATE = 48_000            # output WAV rate
DEEMPHASIS_US = 75             # de-emphasis time constant (US: 75 us)




def main():
    p = argparse.ArgumentParser(description="FM receiver: IQ file -> WAV.")
    p.add_argument("iq_file", help="SigMF .iq/.sigmf-data capture")
    p.add_argument("--out", default="fm_audio.wav")
    p.add_argument("--tune", type=float, default=0.0,
                   help="tune offset Hz if station isn't at capture center")
    p.add_argument("--audio-bw", type=float, default=100_000,
                   help="post-demod channel bandwidth (Hz)")
    args = p.parse_args()

    # 1. load the recording
    src = FileSource(args.iq_file)
    print(f"[*] {src}")
    fs = src.sample_rate
    iq = src.iq
    if fs <= 0:
        print("error: sample rate missing from metadata", file=sys.stderr)
        return 1

    # 2. tune the station to baseband if it's offset in the capture
    if args.tune != 0.0:
        print(f"[*] tuning {args.tune/1e3:g} kHz to baseband")
        iq = frequency_shift(iq, -args.tune, fs)

    # 3. lowpass to the FM channel (~200 kHz wide; we keep audio-bw each side)
    taps = design_lowpass(args.audio_bw, fs, num_taps=201)
    iq = fir_apply(iq, taps)
    print(f"[*] filtered to +/-{args.audio_bw/1e3:g} kHz channel")

    # 4. FM demodulate (phase discriminator)
    audio = fm_demod(iq, deviation_hz=FM_DEVIATION, sample_rate=fs)
    print(f"[*] demodulated: {len(audio):,} samples")

    # 5. resample from capture rate down to audio rate.
    #    fs is e.g. 2_000_000; reduce to 48_000. Use an integer-ish ratio.
    #    Decimate in stages: first to ~AUDIO_RATE*k, then to AUDIO_RATE.
    #    Simple path: rational resample AUDIO_RATE/fs reduced.
    from math import gcd
    g = gcd(int(AUDIO_RATE), int(fs))
    up, down = int(AUDIO_RATE) // g, int(fs) // g
    print(f"[*] resampling {fs/1e6:g} Msps -> {AUDIO_RATE/1e3:g} kHz "
          f"(up={up}, down={down})")
    audio = resample_poly(audio, up, down)

    # 6. de-emphasis + normalize to int16 WAV range
    audio = deemphasis(audio, AUDIO_RATE, tau_us=DEEMPHASIS_US)
    peak = np.max(np.abs(audio)) or 1.0
    pcm = np.int16(np.clip(audio / peak * 0.9, -1, 1) * 32767)

    # 7. write WAV
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.tobytes())
    dur = len(pcm) / AUDIO_RATE
    print(f"[*] wrote {args.out}: {dur:.1f}s of audio at {AUDIO_RATE/1e3:g} kHz")
    print("    play it to hear the station.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
