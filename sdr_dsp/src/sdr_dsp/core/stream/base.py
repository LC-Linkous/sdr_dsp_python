"""The source seam: anything that yields IQ blocks with metadata.

This Protocol is the boundary that keeps sdr_dsp device-agnostic. The core DSP
never imports a source; it operates on the complex64 arrays a source hands it.
A HackRF, an RTL-SDR, or a file are all just IQSources -- write one adapter and
the whole library works against it.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class IQSource(Protocol):
    """Anything that can provide IQ samples plus the metadata to interpret them.

    Attributes:
        sample_rate: samples per second (Hz).
        center_freq: RF center frequency the samples were captured at (Hz).

    Implementations provide ``blocks()`` to stream decoded complex64 arrays.
    A bounded source may also support ``read(n)``; unbounded/live sources need
    only ``blocks()``.
    """

    sample_rate: float
    center_freq: float

    def blocks(self) -> Iterator[np.ndarray]:
        """Yield complex64 blocks until the source is exhausted or stopped."""
        ...


class ArraySource:
    """The simplest source: wrap an in-memory complex64 array.

    Useful for tests, synthetic signals, and feeding already-loaded data into
    the same pipeline code a file or device would drive.
    """

    def __init__(self, iq: np.ndarray, sample_rate: float,
                 center_freq: float = 0.0, block_size: int = 65536):
        self.iq = np.asarray(iq, dtype=np.complex64)
        self.sample_rate = float(sample_rate)
        self.center_freq = float(center_freq)
        self.block_size = int(block_size)

    def blocks(self) -> Iterator[np.ndarray]:
        n = len(self.iq)
        for start in range(0, n, self.block_size):
            yield self.iq[start:start + self.block_size]

    def read(self, n_samples: int) -> np.ndarray:
        """Return the whole array (or the first n_samples)."""
        return self.iq[:int(n_samples)]

    def __repr__(self):
        return (f"ArraySource({len(self.iq):,} samples @ "
                f"{self.sample_rate/1e6:g} Msps)")
