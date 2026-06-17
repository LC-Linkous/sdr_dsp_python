"""FileSource: read a SigMF recording (e.g. a hackrfpy capture) as an IQSource.

This is the development workhorse -- it needs no hardware, so the whole DSP
pipeline can be built and tested against saved captures. It satisfies the same
IQSource protocol a live device adapter would, so example code written against
a file runs unchanged against hardware later.

Streaming: blocks() reads the file incrementally from disk, one block at a time,
so a recording larger than RAM streams fine. The full array is available via the
.iq property, which loads lazily on first access -- so small-file code that does
`src.iq` still works, but you never pay that cost unless you ask for it.
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from ..io.sigmf import load_iq, iq_info


class FileSource:
    """An IQSource backed by a SigMF recording on disk.

    sample_rate and center_freq come from the sidecar (read cheaply, without
    loading the samples). blocks() streams the file in block_size chunks read
    incrementally from disk; .iq loads the whole recording lazily if you ask.

    Args:
        path:           the .iq / .sigmf-data / .sigmf-meta path.
        block_size:     samples per block from blocks().
        count:          limit reading to this many samples (None = whole file).
        offset_samples: skip this many samples from the start.
    """

    def __init__(self, path, block_size=65536, count=None, offset_samples=0):
        self.path = path
        self.block_size = int(block_size)
        self._count = count
        self._offset = int(offset_samples)
        # cheap: stat + sidecar only, no sample data read
        info = iq_info(path)
        self.sample_rate = info["sample_rate"]
        self.center_freq = info["center_freq"]
        self.datatype = info["datatype"]
        self._total = info["total_samples"]
        self._iq = None        # lazy full-array cache

    @property
    def n_samples(self) -> int:
        """Total samples available to this source (respecting count/offset)."""
        avail = max(0, self._total - self._offset)
        return avail if self._count is None else min(avail, int(self._count))

    @property
    def iq(self) -> np.ndarray:
        """The whole recording as complex64, loaded lazily on first access.

        Convenient for small files and tests. For large recordings, prefer
        blocks() -- touching .iq loads everything into RAM.
        """
        if self._iq is None:
            self._iq, _ = load_iq(self.path, count=self._count,
                                  offset_samples=self._offset)
        return self._iq

    def blocks(self) -> Iterator[np.ndarray]:
        """Yield the recording in block_size chunks, read from disk on demand.

        Each block is read with its own seek+read, so memory use stays at one
        block regardless of file size. If .iq has already been loaded (small-file
        path), slice that instead of re-reading.
        """
        total = self.n_samples
        if self._iq is not None:
            # already in memory; just slice it
            for start in range(0, total, self.block_size):
                yield self._iq[start:start + self.block_size]
            return
        # true streaming: read each block from disk
        read = 0
        while read < total:
            this = min(self.block_size, total - read)
            block, _ = load_iq(self.path, count=this,
                               offset_samples=self._offset + read)
            if len(block) == 0:
                break
            yield block
            read += len(block)

    def read(self, n_samples: int) -> np.ndarray:
        """Read up to n_samples from the start (respecting offset). Streams from
        disk without loading the whole file."""
        block, _ = load_iq(self.path, count=int(n_samples),
                           offset_samples=self._offset)
        return block

    def __len__(self):
        return self.n_samples

    def __repr__(self):
        return (f"FileSource({self.path!r}, {self.n_samples} samples, "
                f"{self.sample_rate/1e6:g} Msps @ {self.center_freq/1e6:g} MHz, "
                f"{self.datatype})")
