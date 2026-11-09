"""IQ sink: save processed complex64 IQ back out as a SigMF recording.

A thin convenience over io.sigmf.save_iq, so example pipelines can write their
processed output (a filtered channel, a tuned signal) as a new cf32_le
recording with metadata. The library produces analysis; this is how processed
IQ leaves.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..io.sigmf import save_iq


def write_iq(
    path: str | os.PathLike[str],
    iq: np.ndarray,
    sample_rate: float,
    center_freq: float = 0.0,
    **extra_global: Any,
) -> tuple[str, str]:
    """Save complex64 IQ + a SigMF sidecar (cf32_le). Returns (data, meta) paths.

    Pass-through to io.sigmf.save_iq; kept here so sinks are a uniform place to
    look for "where results go". extra_global lands in the SigMF global object.
    """
    return save_iq(path, iq, sample_rate, center_freq=center_freq,
                   extra_global=extra_global or None)
