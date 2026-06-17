"""FileSource: read a SigMF recording (e.g. a hackrfpy capture) as an IQSource.

This is the development workhorse -- it needs no hardware, so the whole DSP
pipeline can be built and tested against saved captures. It satisfies the same
IQSource protocol a live device adapter would, so example code written against
a file runs unchanged against hardware later.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ..io.sigmf import load_iq, read_meta


class FileSource:
    """An IQSource backed by a SigMF recording on disk.

    sample_rate and center_freq are read from the sidecar. blocks() streams the
    file in block_size chunks; the full array is also available via .iq.
    """

    def __init__(self, path, block_size=65536, count=None, offset_samples=0):
        self.path = path
        self.block_size = int(block_size)
        iq, meta = load_iq(path, count=count, offset_samples=offset_samples)
        self.iq = iq
        self.meta = meta
        g = meta.get("global", {})
        self.sample_rate = float(g.get("core:sample_rate", 0.0))
        caps = meta.get("captures", [{}])
        self.center_freq = float(caps[0].get("core:frequency", 0.0)) if caps \
            else 0.0
        self.datatype = g.get("core:datatype", "ci8")

    def blocks(self) -> Iterator[np.ndarray]:
        n = len(self.iq)
        for start in range(0, n, self.block_size):
            yield self.iq[start:start + self.block_size]

    def read(self, n_samples: int) -> np.ndarray:
        return self.iq[: int(n_samples)]

    def __len__(self):
        return len(self.iq)

    def __repr__(self):
        return (f"FileSource({self.path!r}, {len(self.iq)} samples, "
                f"{self.sample_rate/1e6:g} Msps @ {self.center_freq/1e6:g} MHz, "
                f"{self.datatype})")
