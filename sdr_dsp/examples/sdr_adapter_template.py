#! /usr/bin/python3
"""sdr_adapter_template.py -- a fill-in-the-blanks adapter for a NEW SDR.

Copy this file to `examples/<yourdevice>_capture.py` (and/or `_sink.py`) and
fill in the four marked spots. When you are done you will have an IQSource
(receive) and optionally a TXSink (transmit) that the device-agnostic core
drives WITHOUT ANY CHANGE to `src/sdr_dsp/`. That is the whole modularity
claim, made concrete: the library is not locked to the HackRF One.

Read docs/ADDING_AN_SDR.md for the full process and the porting-record table,
and docs/MODULARITY.md for the layer model this fits into.

The FOUR things a real adapter must get right (all marked `>>> FILL IN`):
  1. open/tune/start the device (vendor library call)
  2. read native samples in blocks
  3. NORMALIZE the device's native format to complex64  <-- the usual gotcha
  4. (transmit only) hand complex64 back to the device

This file is runnable as-is: a built-in `_FakeSDR` stands in for the vendor
device so `demo()` proves the seam end to end with no hardware. In a real port
you DELETE `_FakeSDR` and wire the four spots to your vendor library.

Library deps only (numpy). No hardware.

Usage:
    python examples/sdr_adapter_template.py          # self-demo over the fake device
"""
import sys

import numpy as np

from sdr_dsp.core import fm_pilot_excess_db, psd
from sdr_dsp.sinks import TXSink
from sdr_dsp.sources import IQSource


# ==========================================================================
# The device stand-in. In a REAL port, delete this class entirely and import
# your vendor library instead (pyrtlsdr, SoapySDR, uhd, ...). It exists only so
# this template runs and is testable with no hardware. It emits cu8 (unsigned
# 8-bit interleaved I/Q) -- the RTL-SDR wire format -- to make the
# normalization step (spot 3) realistic rather than a no-op.
# ==========================================================================
class _FakeSDR:
    """Pretends to be a vendor device handle. Yields cu8 bytes of FM-ish IQ."""

    def __init__(self, sample_rate, center_freq, seed=0):
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self._rng = np.random.default_rng(seed)

    def read_native(self, n_samples):
        # A wideband-FM-ish complex signal (a pilot-bearing tone stack), scaled
        # and offset into the cu8 range [0, 255] the way an RTL-SDR delivers it.
        t = np.arange(n_samples) / self.sample_rate
        msg = (0.6 * np.sin(2 * np.pi * 19_000 * t)
               + 0.4 * np.sin(2 * np.pi * 800 * t))
        z = np.exp(1j * 2 * np.pi * 60_000 * np.cumsum(msg) / self.sample_rate)
        z = z + 0.02 * (self._rng.standard_normal(n_samples)
                        + 1j * self._rng.standard_normal(n_samples))
        i = np.clip(np.round(z.real * 110 + 127.5), 0, 255).astype(np.uint8)
        q = np.clip(np.round(z.imag * 110 + 127.5), 0, 255).astype(np.uint8)
        return np.stack([i, q], axis=1).reshape(-1).tobytes()


# ==========================================================================
# RECEIVE adapter -- satisfies IQSource (sample_rate, center_freq, blocks()).
# ==========================================================================
class TemplateSDRSource:
    """An IQSource for a new SDR. Copy, rename, and fill in the marked spots.

    The core never imports this; it just calls blocks() and works on the
    complex64 arrays it yields. sample_rate / center_freq must reflect what the
    device actually tuned, so downstream measurements (estimate_fm_cfo, the
    pilot check, channelizers) are correct.
    """

    def __init__(self, sample_rate, center_freq, *, block_size=65536,
                 device=None):
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.block_size = int(block_size)
        # >>> FILL IN (1): open/tune/start your device here. Replace _FakeSDR
        #     with your vendor handle, e.g.
        #         from rtlsdr import RtlSdr
        #         self._dev = RtlSdr(); self._dev.sample_rate = sample_rate
        #         self._dev.center_freq = center_freq; self._dev.gain = "auto"
        self._dev = device or _FakeSDR(sample_rate, center_freq)

    @staticmethod
    def _to_complex64(raw):
        """>>> FILL IN (3): normalize the device's NATIVE format to complex64.

        This is the step most often gotten wrong. The core only ever sees
        complex64 in roughly [-1, 1]; do the scaling ONCE, here. Worked example
        below is cu8 (RTL-SDR: unsigned 8-bit, midpoint 127.5). See the table
        in docs/ADDING_AN_SDR.md for other formats (HackRF ci8 -> /128, USRP
        cf32 -> just cast, int16 -> /32768).
        """
        buf = np.frombuffer(raw, dtype=np.uint8)
        n = buf.size - (buf.size % 2)
        i = buf[0:n:2].astype(np.float32)
        q = buf[1:n:2].astype(np.float32)
        return np.ascontiguousarray(((i - 127.5) + 1j * (q - 127.5)) / 127.5,
                                    dtype=np.complex64)

    def blocks(self):
        """Yield complex64 blocks until the source is exhausted or stopped."""
        # >>> FILL IN (2): pull native samples from the device in a loop. Here
        #     the fake device is one-shot; a real stream loops until stopped.
        raw = self._dev.read_native(self.block_size)
        yield self._to_complex64(raw)

    def read(self, n_samples):
        """Optional convenience for bounded use: return one array of n samples."""
        raw = self._dev.read_native(int(n_samples))
        return self._to_complex64(raw)


# ==========================================================================
# TRANSMIT adapter -- satisfies TXSink (sample_rate, center_freq, transmit()).
# Optional: many SDRs are RX-only. Guarded like examples/hackrf_sink.py so it
# can never key a radio by accident.
# ==========================================================================
class TemplateSDRSink:
    """A TXSink for a new SDR. Guarded: transmit() refuses unless armed."""

    def __init__(self, center_freq, sample_rate, *, armed=False, device=None):
        self.center_freq = float(center_freq)
        self.sample_rate = float(sample_rate)
        self.armed = bool(armed)
        self._dev = device

    def transmit(self, iq):
        iq = np.asarray(iq, dtype=np.complex64)
        if not self.armed:
            raise RuntimeError(
                "TemplateSDRSink is not armed: refusing to transmit. Arm only "
                "with a legal, ideally wired (TX -> attenuator -> RX) setup.")
        # >>> FILL IN (4): convert complex64 to the device TX format and send.
        #     e.g. cu8: bytes = ((iq.real*127.5+127.5), (iq.imag*127.5+127.5))
        raise NotImplementedError(
            "Wire the vendor TX call here on the bench. Left unimplemented so "
            "software can't pretend to transmit.")


def demo():
    """Prove the seam: a NON-HackRF source drives the real library unchanged."""
    fs, fc = 2_000_000.0, 103_700_000.0
    src = TemplateSDRSource(fs, fc, block_size=200_000)

    print(f"[*] {src.__class__.__name__} satisfies IQSource: "
          f"{isinstance(src, IQSource)}")
    print(f"[*] TemplateSDRSink satisfies TXSink: "
          f"{isinstance(TemplateSDRSink(fc, fs), TXSink)}")

    # Run the ACTUAL library DSP over the adapter's output -- no core changes.
    iq = src.read(200_000)
    freqs, power = psd(iq, fs, nfft=4096)
    pilot = fm_pilot_excess_db(iq, fs)
    print(f"[*] pulled {len(iq)} complex64 samples via the adapter")
    print(f"[*] core DSP ran on them: PSD {power.shape[0]} bins, "
          f"pilot excess {pilot:+.1f} dB" if pilot is not None
          else "[*] core DSP ran on them (pilot n/a for this synthetic)")
    print("[*] the point: the core imported no device, yet processed a "
          "non-HackRF source. Fill in the 4 spots for YOUR radio and the same "
          "holds. See docs/ADDING_AN_SDR.md.")
    return 0


if __name__ == "__main__":
    sys.exit(demo())
