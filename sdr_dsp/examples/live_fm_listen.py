#! /usr/bin/python3
"""live_fm_listen.py -- tune an FM station and play it through your speakers.

The live counterpart to fm_receiver.py: instead of writing a WAV, it streams
from the HackRF, demodulates block by block, and plays the audio in real time.

Set STATION_HZ below and run it. Everything else has a working default.

    python examples/live_fm_listen.py
    python examples/live_fm_listen.py --freq 101.1e6    # override for one run

Requires hackrfpy + the hackrf-tools binaries, AND sounddevice:
    uv sync --extra examples-hackrf --extra audio

Two things make a streaming receiver different from the file-based one, and
both are handled below rather than papered over.

REAL TIME. Resampling 2 Msps straight down to 48 kHz means up=3, down=125:
resample_poly upsamples to 6 Msps and runs a 2501-tap filter there, which
measures about 2.4x real time on a normal laptop -- it can never keep up, and
you would hear stuttering no matter how fast the machine is. Decimating the
IQ by 5 first (free, since the channel filter has to run anyway) drops the
ratio to up=3, down=25 and the filter to 501 taps at 1.2 Msps. Same audio,
about 0.36x real time.

CONTINUITY. Every stage here has memory, and a block boundary is not a signal
boundary. A filter restarted each block ramps up from an empty delay line; a
phase discriminator restarted each block loses its reference sample; a
one-pole IIR restarted each block jumps from zero. At eight blocks a second
those produce a steady buzz. Each stage below carries its state across
blocks, so the output matches what you would get by processing the whole
stream at once.
"""

# ===========================================================================
#  CONFIG -- EDIT THIS
# ===========================================================================

# The station to listen to, in Hz. 98.5e6 is 98.5 MHz.
STATION_HZ = 103.5e6 #98.5e6

VOLUME = 0.5              # 0..1 output level
DEEMPHASIS_US = 75        # 75 in the Americas/South Korea, 50 most elsewhere
AUTO_GAIN = True          # probe the band and pick LNA/VGA before listening
LNA_DB, VGA_DB = 16, 20   # used when AUTO_GAIN is False
TOOLS_DIR = None          # path to hackrf-tools if not on PATH

# ===========================================================================

import argparse
import sys
from math import gcd

import numpy as np

from sdr_dsp.core import (design_lowpass, fir_apply, fm_demod, resample_poly,
                          deemphasis, capture_health, fm_pilot_excess_db,
                          search_gain)
from sdr_dsp.sources.probe import probe_capture

SAMPLE_RATE = 2_000_000   # HackRF minimum, and plenty for one FM channel
DECIMATION = 5            # 2 Msps -> 400 kHz before demodulating
CHANNEL_BW = 100_000      # half-width of the FM channel we keep
CHANNEL_TAPS = 201
AUDIO_RATE = 48_000
FM_DEVIATION = 75_000
BLOCK_SAMPLES = 250_000   # 0.125 s per block; must divide by DECIMATION

# Warm-up for the de-emphasis IIR, which has no state argument. It forgets
# exponentially, so priming it with the tail of the previous block and
# discarding that region is exact far below the noise floor: tau is 3.6
# samples at 48 kHz, so 64 samples leaves an error around e^-18.
DEEMPH_WARMUP = 64


class FMStream:
    """Stateful FM receive chain: complex64 blocks in, audio blocks out.

    Holds the delay line of every stage between calls, so consecutive blocks
    join seamlessly instead of clicking at each boundary.
    """

    def __init__(self, sample_rate=SAMPLE_RATE, decimation=DECIMATION,
                 channel_bw=CHANNEL_BW, audio_rate=AUDIO_RATE,
                 deemph_us=DEEMPHASIS_US):
        self.fs = float(sample_rate)
        self.dec = int(decimation)
        self.inter_fs = self.fs / self.dec
        self.audio_rate = int(audio_rate)
        self.deemph_us = deemph_us

        self.taps = design_lowpass(channel_bw, self.fs, num_taps=CHANNEL_TAPS)
        g = gcd(self.audio_rate, int(self.inter_fs))
        self.up = self.audio_rate // g
        self.down = int(self.inter_fs) // g

        # resample_poly builds 2*10*max(up,down)+1 taps at the upsampled rate
        # and compensates their group delay, which makes it non-causal: an
        # output sample depends on input both behind and AHEAD of it. So the
        # resampler needs history on both sides. Past context (rs_hist) is
        # carried from the previous block; future context (rs_ahead) is
        # obtained by holding the newest samples back until the block after
        # next, rather than letting resample_poly pad them with zeros. Both
        # are whole multiples of `down` so the output accounting stays exact
        # instead of drifting on a rounding.
        ntaps = 20 * max(self.up, self.down) + 1
        need = (ntaps + self.up - 1) // self.up
        self.rs_hist = ((need + self.down - 1) // self.down) * self.down
        self.rs_ahead = self.rs_hist

        # mono audio is 30 Hz..15 kHz of the composite; everything above
        # (19 kHz pilot, 38 kHz L-R, 57 kHz RDS) is machinery, not program.
        # Without this the pilot reaches the output attenuated only by
        # de-emphasis.
        self.audio_taps = design_lowpass(15_000, self.audio_rate,
                                         num_taps=101)

        # carried state
        self._iq_tail = np.zeros(len(self.taps) - 1, dtype=np.complex64)
        self._demod_tail = np.zeros(1, dtype=np.complex64)
        self._rs_buf = np.zeros(self.rs_hist, dtype=np.float64)
        self._deemph_tail = np.zeros(DEEMPH_WARMUP, dtype=np.float64)
        self._audio_tail = np.zeros(len(self.audio_taps) - 1,
                                    dtype=np.float64)

    @staticmethod
    def _roll_tail(tail, new):
        """Keep the last len(tail) samples of (tail + new)."""
        n = len(tail)
        if new.size >= n:
            return new[-n:].copy()
        return np.concatenate([tail[new.size:], new])

    def process(self, block):
        """One block of complex64 IQ -> one block of real audio."""
        block = np.asarray(block, dtype=np.complex64)
        if block.size % self.dec:
            block = block[:block.size - (block.size % self.dec)]
        if block.size == 0:
            return np.zeros(0)

        # 1. channel filter, overlap-save. Feeding the previous block's tail
        #    and keeping only the new region gives exactly the output of a
        #    filter that never stopped running.
        m = len(self._iq_tail)
        filtered = fir_apply(np.concatenate([self._iq_tail, block]),
                             self.taps)[m:]
        self._iq_tail = self._roll_tail(self._iq_tail, block)

        # 2. decimate. Block length is a multiple of `dec`, so the sampling
        #    phase stays aligned from one block to the next.
        narrow = filtered[::self.dec]
        if narrow.size == 0:
            return np.zeros(0)

        # 3. FM demodulate. The discriminator differences consecutive phases,
        #    so it needs the last sample of the previous block to produce a
        #    correct first output sample instead of a spurious jump.
        primed = np.concatenate([self._demod_tail, narrow])
        audio = fm_demod(primed, deviation_hz=FM_DEVIATION,
                         sample_rate=self.inter_fs)[-narrow.size:]
        self._demod_tail = narrow[-1:].copy()

        # 4. resample to audio rate. Emit only samples that have both past
        #    and future context available; hold the rest until more arrives.
        #    The cost is rs_ahead samples of latency -- here about 0.4 ms,
        #    inaudible -- in exchange for output identical to processing the
        #    whole stream at once.
        buf = np.concatenate([self._rs_buf, audio])
        avail = buf.size - self.rs_hist - self.rs_ahead
        if avail < self.down:
            self._rs_buf = buf
            return np.zeros(0)
        k = (avail // self.down) * self.down
        chunk = buf[:self.rs_hist + k + self.rs_ahead]
        drop = self.rs_hist * self.up // self.down
        keep = (self.rs_hist + k) * self.up // self.down
        out = resample_poly(chunk, self.up, self.down)[drop:keep]
        self._rs_buf = buf[k:]

        # 5. de-emphasis. Broadcast FM pre-emphasizes treble before
        #    transmission; without undoing it the audio is harsh and hissy.
        primed = np.concatenate([self._deemph_tail, out])
        shaped = deemphasis(primed, self.audio_rate,
                            tau_us=self.deemph_us)[len(self._deemph_tail):]
        self._deemph_tail = self._roll_tail(self._deemph_tail, out)

        # 6. audio lowpass (overlap-save, same pattern as the channel
        #    filter): keep 0..15 kHz, drop the pilot and everything above.
        m = len(self._audio_tail)
        final = fir_apply(np.concatenate([self._audio_tail, shaped]),
                          self.audio_taps)[m:]
        self._audio_tail = self._roll_tail(self._audio_tail, shaped)
        return final


def pick_gain(h, freq):
    """Find (lna, vga, amp) that captures the STATION, not just a level.

    Uses sdr_dsp.core.search_gain with the 19 kHz stereo pilot as the
    quality metric: the front end (RF amp + LNA) is chosen by how clearly
    the pilot is measured -- the part of the chain that decides what you can
    hear -- and only then does the VGA set the ADC level. The old level-only
    walk would happily amplify the noise floor into the target window and
    call it done.
    """
    def probe(lna, vga, amp):
        # File-path capture, NOT capture_array: the stdout-pipe path drops
        # samples on Windows and buries the pilot (sdr_dsp.sources.probe).
        return probe_capture(h, freq, SAMPLE_RATE, int(SAMPLE_RATE * 0.05),
                             lna=lna, vga=vga, amp=amp)

    r = search_gain(probe,
                    quality=lambda iq: fm_pilot_excess_db(iq, SAMPLE_RATE))
    return r["lna"], r["vga"], r["amp"], r["counts"], r["quality_db"]


def main():
    p = argparse.ArgumentParser(description="Live FM receiver -> speakers.")
    p.add_argument("--freq", type=float, default=STATION_HZ,
                   help=f"station Hz (default: STATION_HZ = {STATION_HZ:g})")
    p.add_argument("--volume", type=float, default=VOLUME)
    p.add_argument("--lna", type=int, default=LNA_DB)
    p.add_argument("--vga", type=int, default=VGA_DB)
    p.add_argument("--no-auto-gain", action="store_true")
    args = p.parse_args()

    try:
        import sounddevice as sd
    except ModuleNotFoundError:
        print("needs sounddevice:  uv sync --extra audio", file=sys.stderr)
        return 1
    try:
        from hackrfpy import HackRF
    except ModuleNotFoundError:
        print("needs hackrfpy:  uv sync --extra examples-hackrf",
              file=sys.stderr)
        return 1
    from hackrf_capture import HackRFCapture

    h = HackRF(tools_dir=TOOLS_DIR, verbose=False)
    det = h.detect()
    if not det["ready"]:
        print(f"no usable HackRF: {det['problem']}", file=sys.stderr)
        return 1

    lna, vga, amp = args.lna, args.vga, False
    if AUTO_GAIN and not args.no_auto_gain:
        print("[*] probing for a working gain ...")
        lna, vga, amp, counts, pilot_db = pick_gain(h, args.freq)
        print(f"    lna={lna} vga={vga} amp={'on' if amp else 'off'}  "
              f"({counts:.0f}/128 ADC counts"
              + (f", pilot {pilot_db:+.1f} dB)" if pilot_db is not None
                 else ")"))

    # Is the station actually there? Cheaper to say so now than to let
    # someone listen to hiss and wonder whether the DSP is broken.
    probe = probe_capture(h, args.freq, SAMPLE_RATE, int(SAMPLE_RATE * 0.1),
                          lna=lna, vga=vga, amp=amp)
    health = capture_health(probe, SAMPLE_RATE, channel_bw=CHANNEL_BW)
    pilot_db = fm_pilot_excess_db(probe, SAMPLE_RATE)
    if pilot_db is not None and pilot_db < 6.0:
        health["ok"] = False
        health["reasons"].append(
            f"no 19 kHz stereo pilot in the demodulated signal "
            f"({pilot_db:+.1f} dB): whatever the level is, it is not a "
            f"broadcast FM station")
    if health["ok"]:
        print(f"[*] station present: channel "
              f"{health['channel_excess_db']:+.1f} dB above the noise floor"
              + (f", pilot {pilot_db:+.1f} dB" if pilot_db is not None
                 else ""))
    else:
        for r in health["reasons"]:
            print(f"[!] {r}")
        print("[!] listening anyway -- expect hiss. Check the frequency, the")
        print("    antenna, and that this station is on the air locally.")

    chain = FMStream()
    print(f"[*] {args.freq/1e6:g} MHz -> {SAMPLE_RATE/1e6:g} Msps, decimate "
          f"{DECIMATION}x to {chain.inter_fs/1e3:g} kHz, resample "
          f"up={chain.up} down={chain.down} -> {AUDIO_RATE/1e3:g} kHz audio")
    print("[*] Ctrl-C to stop")

    stream = sd.OutputStream(samplerate=AUDIO_RATE, channels=1,
                             dtype="float32")
    stream.start()
    blocks = 0
    try:
        with HackRFCapture(args.freq, SAMPLE_RATE, lna=lna, vga=vga,
                           amp=amp, block_size=BLOCK_SAMPLES,
                           tools_dir=TOOLS_DIR) as src:
            for iq in src.blocks():
                audio = chain.process(iq)
                if audio.size == 0:
                    continue
                stream.write(
                    np.clip(audio * args.volume, -1.0, 1.0).astype(np.float32))
                blocks += 1
                if blocks % 40 == 0:
                    print(f"    {blocks * BLOCK_SAMPLES / SAMPLE_RATE:.0f}s")
    except KeyboardInterrupt:
        print("\n[*] stopped")
    finally:
        stream.stop()
        stream.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
