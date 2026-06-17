#! /usr/bin/python3
"""hackrf_sink.py -- a HackRF transmit adapter (TXSink), the RX-side mirror.

This is the transmit counterpart to hackrf_capture.py: it implements the TXSink
protocol from OUTSIDE the library, so the device-agnostic core can transmit
through a HackRF without knowing anything about it. Pair it with the LiveLink
driver to run the ARQ protocol over real radio.

  *** READ THIS BEFORE TRANSMITTING ***
  Transmitting over the air is legally regulated. You are responsible for
  operating within the law: frequency, power, and licensing rules apply, and
  many bands are off-limits. The safest bench setup is WIRED -- TX -> coax ->
  attenuator -> RX -- which exercises the full system without radiating. Do not
  connect an antenna and transmit unless you know the band is permitted and you
  are licensed where required.

  hackrfpy gates transmit deliberately. This adapter shows the SHAPE of a TXSink;
  enabling actual transmit requires the gated hackrfpy TX path, which you opt
  into knowingly.

This file is a template/seam: the transmit() body is the one piece that calls
the device, and it is left guarded so it cannot transmit by accident.
"""
import sys

import numpy as np

sys.path.insert(0, "src")


class HackRFSink:
    """A TXSink that transmits complex64 IQ through a HackRF.

    Implements the TXSink protocol (sample_rate, center_freq, transmit). The
    constructor is safe; transmit() is guarded by `armed` so the device is never
    keyed unless you explicitly arm it -- a deliberate safety gate against
    accidental transmission.

    Args:
        center_freq: TX center frequency (Hz). YOUR responsibility to ensure
                     this is legal for your location and license.
        sample_rate: sample rate (Hz).
        tx_gain:     IF/TX gain (device units).
        armed:       must be True to actually transmit. Default False.
    """

    def __init__(self, center_freq, sample_rate, *, tx_gain=20, armed=False):
        self.center_freq = float(center_freq)
        self.sample_rate = float(sample_rate)
        self.tx_gain = tx_gain
        self.armed = bool(armed)
        self._dev = None

    def _open(self):
        try:
            from hackrfpy import HackRF
        except ImportError as e:
            raise ImportError(
                "HackRFSink needs hackrfpy (and its gated TX path). Install with "
                "'uv sync --extra examples-hackrf'. Note hackrfpy gates transmit "
                "deliberately; enabling it is a knowing opt-in."
            ) from e
        self._dev = HackRF()
        # device-specific TX config would go here (freq, rate, gain, TX mode)

    def transmit(self, iq):
        """Transmit one buffer of complex64 IQ -- GUARDED.

        Raises unless `armed` is True, so this never keys the radio by accident.
        The actual device call is intentionally left for you to enable on the
        bench, once you've confirmed your setup is legal and (ideally) wired.
        """
        iq = np.asarray(iq, dtype=np.complex64)
        if not self.armed:
            raise RuntimeError(
                "HackRFSink is not armed: refusing to transmit. Set armed=True "
                "only when you have confirmed a legal/wired setup. (This guard "
                "exists so the protocol stack can be wired and tested without "
                "any risk of accidental transmission.)")
        if self._dev is None:
            self._open()
        # --- the one device-specific line, left for bench enablement: ---
        # convert complex64 -> the device's TX sample format and send:
        #   self._dev.transmit(iq, center_freq=self.center_freq, ...)
        raise NotImplementedError(
            "Enable the hackrfpy TX call here on the bench, with a wired/"
            "attenuated setup. Left unimplemented so software can't pretend to "
            "transmit.")


def demo():
    """Show that the sink wires into the protocol (without transmitting)."""
    from sdr_dsp.sinks import TXSink
    sink = HackRFSink(433.92e6, 2e6)            # safe: not armed
    print(f"[*] {sink.__class__.__name__} satisfies TXSink protocol: "
          f"{isinstance(sink, TXSink)}")
    print("[*] constructing a sink is safe; transmit() is guarded by `armed` "
          "and the device call is left for bench enablement.")
    print("[*] for the full software-provable exchange, use LoopbackSink "
          "(see examples/two_station_link.py).")


if __name__ == "__main__":
    demo()
