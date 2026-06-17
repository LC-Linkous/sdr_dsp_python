"""TXSink: the transmit-side device boundary (the mirror of IQSource).

Just as IQSource is the seam that lets the library receive from any device
without knowing about it, TXSink is the seam for transmitting TO any device. The
core never imports a sink implementation; it hands complex64 IQ to something
satisfying this protocol, and a device adapter (HackRF, USRP, ...) -- living
OUTSIDE the library, in example/user code -- turns that into real RF.

This keeps the library device-agnostic on transmit exactly as on receive. The
library provides:
  - the TXSink protocol (this file),
  - a LoopbackSink that "transmits" into an in-memory buffer (for testing the
    wiring without a radio),
and the real device adapters are examples (see examples/hackrf_sink.py,
examples/usrp_sink.py) that the user supplies, like hackrf_capture.py on RX.

HONESTY: everything downstream of a real TXSink -- actual radiation, timing,
regulatory compliance -- cannot be verified in software. The protocol and the
LoopbackSink let the FULL protocol stack be exercised (ARQ -> modulate -> sink),
but only a real device adapter on a bench proves over-the-air transmission.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TXSink(Protocol):
    """Anything that can transmit complex64 IQ at a center frequency.

    Attributes:
        sample_rate: samples per second (Hz) the IQ is sampled at.
        center_freq: RF center frequency to transmit at (Hz).

    Implementations provide ``transmit(iq)`` to send one buffer of complex64
    samples. A device adapter handles gain, timing, and the half-duplex
    TX/RX turnaround; the library only hands it baseband IQ.
    """

    sample_rate: float
    center_freq: float

    def transmit(self, iq: np.ndarray) -> None:
        """Transmit one buffer of complex64 IQ samples."""
        ...


class LoopbackSink:
    """A TXSink that 'transmits' into an in-memory buffer instead of a radio.

    The transmit-side counterpart of ArraySource: it satisfies the TXSink
    protocol but keeps every transmitted sample in `.buffer`, so the full stack
    (ARQ -> modulate -> sink) can be wired and tested without hardware. Pair it
    with an ArraySource on the receive side to close a software loop, optionally
    through the simulated channel.

    This is how the live-driver wiring is verified in software: the protocol
    really drives a sink, the sink really receives the IQ -- only the antenna is
    missing.
    """

    def __init__(self, sample_rate, center_freq=0.0):
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.buffer = np.zeros(0, dtype=np.complex64)
        self.transmit_count = 0

    def transmit(self, iq):
        """Append the IQ to the in-memory buffer (no radio)."""
        iq = np.asarray(iq, dtype=np.complex64)
        self.buffer = np.concatenate([self.buffer, iq])
        self.transmit_count += 1

    def clear(self):
        """Reset the buffer and counter."""
        self.buffer = np.zeros(0, dtype=np.complex64)
        self.transmit_count = 0

    def __repr__(self):
        return (f"LoopbackSink({len(self.buffer)} samples buffered, "
                f"{self.transmit_count} transmits, "
                f"{self.sample_rate/1e6:g} Msps @ {self.center_freq/1e6:g} MHz)")
