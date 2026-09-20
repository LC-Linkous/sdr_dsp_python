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
import numpy as np

from sdr_dsp.core import (
    design_lowpass, fir_apply, fm_demod, resample_poly, frequency_shift,
    deemphasis, capture_health, fm_pilot_excess_db, fm_stereo_decode,
)
from sdr_dsp.sinks import write_wav

# FM broadcast parameters
FM_DEVIATION = 75_000          # max deviation of broadcast FM (Hz)
AUDIO_RATE = 48_000            # output WAV rate
DEEMPHASIS_US = 75             # de-emphasis time constant (US: 75 us)
CHANNEL_TAPS = 201             # length of the channel-select lowpass
FM_RATE = 250_000              # intermediate demod rate (divides 2/4/8/10/20 Msps)
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
    p.add_argument("--stereo", action="store_true",
                   help="decode L/R stereo (pilot-locked) and write a stereo "
                        "WAV; default is mono (L+R)")
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

    # 3c. Decimate the channel to an intermediate rate BEFORE demodulating.
    #     The signal is already bandlimited to the channel by step 3, so the
    #     capture's full bandwidth is wasted work in the discriminator and the
    #     downstream resampler (at 2 Msps the old path demodulated 2 M
    #     samples then resampled 3/125). Bringing it to FM_RATE = 240 kHz
    #     first means the discriminator and de-emphasis run on ~8x fewer
    #     samples. 250 kHz divides every common HackRF capture rate (2/4/8/
    #     10/20 Msps), so capture -> intermediate is pure decimation -- the
    #     stage with the most samples pays the least. The final 250k -> 48k
    #     is a small resample (24/125). This mirrors the live path
    #     (live_fm_listen decimates before demod for the same reason).
    #     250 kHz comfortably passes the ~200 kHz FM channel.
    if fs > FM_RATE and int(fs) % int(FM_RATE) == 0:
        decim = int(fs) // int(FM_RATE)
        # channel filter above already removed everything past the channel,
        # so plain decimation here does not alias the audio-band content
        iq = iq[::decim]
        demod_rate = FM_RATE
        print(f"[*] decimated {fs/1e6:g} Msps -> {FM_RATE/1e3:g} kHz "
              f"for demod (x{decim})")
    else:
        # non-integer ratio (e.g. odd capture rates): resample the channel
        g0 = gcd(int(FM_RATE), int(fs))
        iq = resample_poly(iq, int(FM_RATE) // g0, int(fs) // g0)
        demod_rate = FM_RATE
        print(f"[*] resampled {fs/1e6:g} Msps -> {FM_RATE/1e3:g} kHz for demod")

    # 4. FM demodulate (phase discriminator) at the intermediate rate.
    #    The output is the COMPOSITE multiplex (L+R, pilot, L-R at 38 kHz).
    composite = fm_demod(iq, deviation_hz=FM_DEVIATION, sample_rate=demod_rate)
    print(f"[*] demodulated: {len(composite):,} samples")

    # 4b. stereo: split the composite into L and R before resampling, using
    #     the 19 kHz pilot to coherently detect the 38 kHz L-R subcarrier.
    #     Mono just takes the composite as-is (its 0-15 kHz part is L+R).
    if args.stereo:
        pilot_db = fm_pilot_excess_db(iq, demod_rate)
        if pilot_db is not None and pilot_db < 6.0:
            print(f"[!] --stereo requested but no pilot ({pilot_db:+.1f} dB); "
                  f"the station is likely mono. Falling back to mono.")
            channels = [composite]
        else:
            left, right = fm_stereo_decode(composite, demod_rate)
            print(f"[*] stereo decoded (pilot "
                  f"{pilot_db:+.1f} dB)" if pilot_db is not None
                  else "[*] stereo decoded")
            channels = [left, right]
    else:
        channels = [composite]

    # 5-6. per channel: resample to audio rate, de-emphasize, bandlimit to
    #      15 kHz (drops the pilot/subcarrier remnants), trim settling.
    g = gcd(int(AUDIO_RATE), int(demod_rate))
    up, down = int(AUDIO_RATE) // g, int(demod_rate) // g
    print(f"[*] resampling {demod_rate/1e3:g} kHz -> {AUDIO_RATE/1e3:g} kHz "
          f"(up={up}, down={down})")
    audio_lp = design_lowpass(15_000, AUDIO_RATE, num_taps=101)
    settle = int(AUDIO_SETTLE_S * AUDIO_RATE)
    out_channels = []
    for chan in channels:
        a = resample_poly(chan, up, down)
        a = deemphasis(a, AUDIO_RATE, tau_us=DEEMPHASIS_US)
        a = fir_apply(a, audio_lp)
        if len(a) > 4 * settle:
            a = a[settle:]
        out_channels.append(a)
    if out_channels[0].size == 0:
        print("error: no audio left after filtering; capture too short",
              file=sys.stderr)
        return 1

    # 7. write WAV (mono = 1 channel, stereo = 2 columns with ONE shared
    #    normalization so the L/R balance is preserved).
    if len(out_channels) == 2:
        n = min(len(out_channels[0]), len(out_channels[1]))
        audio = np.column_stack([out_channels[0][:n], out_channels[1][:n]])
    else:
        audio = out_channels[0]
    write_wav(args.out, audio, AUDIO_RATE)
    dur = len(audio) / AUDIO_RATE
    print(f"[*] wrote {args.out}: {dur:.1f}s of audio at {AUDIO_RATE/1e3:g} kHz")
    print("    play it to hear the station.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
