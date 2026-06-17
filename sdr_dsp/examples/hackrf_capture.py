#! /usr/bin/python3
"""HackRF capture helper for the examples -- NOT part of the sdr_dsp library.

This lives in examples/ on purpose. The sdr_dsp library knows nothing about any
device: its job starts when it receives complex64 + a sample rate. Acquiring
those samples from a HackRF is an APPLICATION concern, so it lives here, in the
example code, and uses the separately-installed `hackrfpy` package.

It doubles as the reference for the library's modularity: this class implements
sdr_dsp's IQSource protocol from OUTSIDE the library. Anyone wiring in a
different SDR writes a class shaped like this one -- the DSP core consumes it
unchanged. "We provide the hooks; you provide the hardware."

Requires (examples only, not the library):
    pip install hackrfpy            # plus the hackrf-tools binaries (OS level)
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class HackRFCapture:
    """A live HackRF source shaped to sdr_dsp's IQSource protocol.

    sample_rate / center_freq attributes + a blocks() generator is the entire
    contract sdr_dsp cares about. Use as a context manager so the device is
    released:

        from hackrf_capture import HackRFCapture
        from sdr_dsp.core import psd

        with HackRFCapture(100e6, 2e6) as src:
            for block in src.blocks():
                freqs, p = psd(block, src.sample_rate, center_freq=src.center_freq)
                ...
    """

    def __init__(self, center_freq, sample_rate, *, lna=16, vga=20, amp=False,
                 block_size=262144, tools_dir=None):
        from hackrfpy import HackRF        # example dependency, imported here
        self.center_freq = float(center_freq)
        self.sample_rate = float(sample_rate)
        self.block_size = int(block_size)
        self._h = HackRF(tools_dir=tools_dir)
        self._lna, self._vga, self._amp = lna, vga, amp
        self._rx = None

    def __enter__(self):
        self._rx = self._h.open_receiver(
            self.center_freq, self.sample_rate,
            lna=self._lna, vga=self._vga, amp=self._amp,
            read_samples=self.block_size,
        ).__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._rx is not None:
            self._rx.__exit__(exc_type, exc, tb)
            self._rx = None
        return False

    def blocks(self) -> Iterator[np.ndarray]:
        if self._rx is None:
            raise RuntimeError("use `with HackRFCapture(...) as src:`")
        yield from self._rx.blocks()

    def read(self, n_samples: int) -> np.ndarray:
        if self._rx is None:
            raise RuntimeError("use `with HackRFCapture(...) as src:`")
        return self._rx.read(int(n_samples))

    def capture_array(self, n_samples):
        """Convenience: grab a fixed number of samples without a context block.

        Opens the receiver, reads n_samples, closes. Returns complex64.
        """
        with self:
            return self._rx.read(int(n_samples))
