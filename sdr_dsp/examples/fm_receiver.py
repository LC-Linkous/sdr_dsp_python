#! /usr/bin/python3
"""FM broadcast receiver: capture file -> audio WAV. The end-to-end demo.

Exercises the whole sdr_dsp core on a real wideband-FM recording:
    load IQ  ->  (tune)  ->  lowpass to the station  ->  FM demod  ->
    de-emphasis  ->  resample to 48 kHz  ->  write WAV you can play.

File-based: no hardware needed. Point it at a hackrfpy FM capture (ci8 SigMF).

Usage:
    python examples/fm_receiver.py sample_data/fm_2Msps.iq --out station.wav
    python examples/fm_receiver.py capture.iq --tune -250e3   # tune off-center
    python examples/fm_receiver.py capture.iq --no-check      # skip health check

A note on levels: everything before the WAV write is unitless, so the output
has to be scaled into int16. We deliberately do NOT scale by the raw peak.
A causal FIR starts with an empty delay line, so the first `num_taps` output
samples are a ramp-up transient, and a phase discriminator fed that ramp
produces a burst far larger than any real audio. Normalizing to that burst
buries the actual station near zero -- an audible click, then silence. So we
trim the settling regions and normalize to a high percentile instead.
"""
import argparse
import sys
from math import gcd
from pathlib import Path


from sdr_dsp.sources import FileSource
from sdr_dsp.core import (
    design_lowpass, fir_apply, fm_demod, resample_poly, frequency_shift,
    deemphasis, capture_health, fm_pilot_excess_db,
)
from sdr_dsp.sinks import write_wav

# FM broadcast parameters
FM_DEVIATION = 75_000          # max deviation of broadcast FM (Hz)
AUDIO_RATE = 48_000            # output WAV rate
DEEMPHASIS_US = 75             # de-emphasis time constant (US: 75 us)
CHANNEL_TAPS = 201             # length of the channel-select lowpass
AUDIO_SETTLE_S = 0.005         # audio discarded while resampler/IIR settle

def main():
    p = argparse.ArgumentParser(description="FM receiver: IQ file -> WAV.")
    p.add_argument("iq_file", help="SigMF .iq/.sigmf-data capture")
    p.add_argument("--out", default="fm_audio.wav")
    p.add_argument("--tune", type=float, default=0.0,
                   help="tune offset Hz if station isn't at capture center")
    p.add_argument("--audio-bw", type=float, default=100_000,
                   help="post-demod channel bandwidth (Hz)")
    p.add_argument("--no-check", action="store_true",
                   help="skip the capture health check")
    args = p.parse_args()

    # 1. load the recording
    path = Path(args.iq_file)
    if not path.exists():
        print(f"error: no such capture: {path}", file=sys.stderr)
        return 2
    src = FileSource(str(path))
    print(f"[*] {src}")
    fs = src.sample_rate
    iq = src.iq
    if fs <= 0:
        print("error: sample rate missing from metadata", file=sys.stderr)
        return 1
    if not 0 < args.audio_bw < fs / 2:
        print(f"error: --audio-bw must be in (0, {fs/2:g}) Hz", file=sys.stderr)
        return 2
    if len(iq) < 10 * CHANNEL_TAPS:
        print(f"error: capture too short ({len(iq)} samples) to filter and "
              "demodulate", file=sys.stderr)
        return 1

    # 2. tune the station to baseband if it's offset in the capture
    if args.tune != 0.0:
        print(f"[*] tuning {args.tune/1e3:g} kHz to baseband")
        iq = frequency_shift(iq, -args.tune, fs)

    # health check runs after tuning, so it inspects the channel we'll demod
    if not args.no_check:
        health = capture_health(iq, fs, channel_bw=args.audio_bw)
        pilot_db = fm_pilot_excess_db(iq, fs)
        if health["ok"]:
            print(f"[*] capture peaks at ~{health['adc_counts']:.0f} of 128 "
                  f"ADC counts, channel "
                  f"{health['channel_excess_db']:+.1f} dB above the "
                  f"surrounding noise floor"
                  + (f", 19 kHz pilot {pilot_db:+.1f} dB"
                     if pilot_db is not None else ""))
            if pilot_db is not None and pilot_db < 6.0:
                print("[!] but no stereo pilot was found -- if this is meant "
                      "to be broadcast FM, expect noise")
        else:
            for reason in health["reasons"]:
                print(f"[!] {reason}")
            print("[!] continuing anyway -- expect noise in the output WAV.")
            print("    If the station is offset in the capture, tune to it")
            print("    with --tune <offset_hz>.")

    # 3. lowpass to the FM channel (~200 kHz wide; we keep audio-bw each side)
    taps = design_lowpass(args.audio_bw, fs, num_taps=CHANNEL_TAPS)
    iq = fir_apply(iq, taps)
    print(f"[*] filtered to +/-{args.audio_bw/1e3:g} kHz channel")

    # 3b. Drop the filter's startup transient. fir_apply is causal, so the
    #     first (len(taps) - 1) samples are convolved against an empty delay
    #     line: near-zero magnitude with wildly swinging phase. Feeding that
    #     to the discriminator in step 4 yields a huge spurious spike -- the
    #     click that used to dominate the WAV's peak normalization.
    iq = iq[CHANNEL_TAPS:]

    # 4. FM demodulate (phase discriminator)
    audio = fm_demod(iq, deviation_hz=FM_DEVIATION, sample_rate=fs)
    print(f"[*] demodulated: {len(audio):,} samples")

    # 5. resample from capture rate down to audio rate.
    #    fs is e.g. 2_000_000; reduce to 48_000. Use an integer-ish ratio.
    g = gcd(int(AUDIO_RATE), int(fs))
    up, down = int(AUDIO_RATE) // g, int(fs) // g
    print(f"[*] resampling {fs/1e6:g} Msps -> {AUDIO_RATE/1e3:g} kHz "
          f"(up={up}, down={down})")
    audio = resample_poly(audio, up, down)

    # 6. de-emphasis, then bandlimit to mono audio (0..15 kHz). The
    #    demodulated composite still carries the 19 kHz stereo pilot, the
    #    38 kHz L-R remnant, and RDS at 57 kHz; below 24 kHz Nyquist the
    #    pilot would otherwise reach the WAV attenuated only by de-emphasis.
    #    Then drop the settling region of the resampler, the one-pole IIR
    #    (which starts from a zero accumulator), and this filter.
    audio = deemphasis(audio, AUDIO_RATE, tau_us=DEEMPHASIS_US)
    audio = fir_apply(audio, design_lowpass(15_000, AUDIO_RATE,
                                            num_taps=101))
    settle = int(AUDIO_SETTLE_S * AUDIO_RATE)
    if len(audio) > 4 * settle:
        audio = audio[settle:]
    if audio.size == 0:
        print("error: no audio left after filtering; capture too short",
              file=sys.stderr)
        return 1

    # 7. write WAV. write_wav normalizes to a high percentile rather than the
    #    raw peak, so any residual impulse can't crush the program material.
    write_wav(args.out, audio, AUDIO_RATE)
    dur = len(audio) / AUDIO_RATE
    print(f"[*] wrote {args.out}: {dur:.1f}s of audio at {AUDIO_RATE/1e3:g} kHz")
    print("    play it to hear the station.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
