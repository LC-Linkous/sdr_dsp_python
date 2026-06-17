"""SigMF metadata + data I/O.

Reads the recordings hackrfpy produces (ci8 = complex int8) and writes
processed output as cf32_le (complex float32, SigMF's canonical RF type), since
sdr_dsp works internally in complex64. This module is the file <-> array bridge;
it has no device knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# SigMF datatype string -> (numpy dtype, is_complex, bytes/component)
_DTYPES = {
    "ci8": (np.int8, True),
    "ci16_le": (np.int16, True),
    "cf32_le": (np.float32, True),
    "cf64_le": (np.float64, True),
    "ri8": (np.int8, False),
    "rf32_le": (np.float32, False),
}


def _meta_path(path):
    p = Path(path)
    if p.suffix == ".sigmf-meta":
        return p
    if p.suffix in (".sigmf-data", ".iq"):
        # try .sigmf-meta sibling, else same-stem .sigmf-meta
        cand = p.with_suffix(".sigmf-meta")
        return cand
    return p.with_suffix(".sigmf-meta")


def _data_path(path):
    p = Path(path)
    if p.suffix == ".sigmf-meta":
        for ext in (".sigmf-data", ".iq"):
            cand = p.with_suffix(ext)
            if cand.exists():
                return cand
        return p.with_suffix(".sigmf-data")
    return p


def read_meta(path):
    """Read a .sigmf-meta sidecar into a dict. Accepts the meta or data path."""
    mp = _meta_path(path)
    with open(mp, "r") as f:
        return json.load(f)


def load_iq(path, count=None, offset_samples=0):
    """Load a SigMF recording into complex64, using its sidecar to interpret.

    path:           the .iq/.sigmf-data or .sigmf-meta path.
    count:          max samples to read (None = all).
    offset_samples: skip this many complex samples from the start.

    Returns (iq_complex64, meta_dict). The datatype is read from the sidecar so
    hackrfpy's ci8 captures decode correctly; everything is normalized to
    complex64 in roughly [-1, 1).
    """
    meta = read_meta(path)
    dp = _data_path(path)
    dtype_str = meta.get("global", {}).get("core:datatype", "ci8")
    if dtype_str not in _DTYPES:
        raise ValueError(f"unsupported SigMF datatype: {dtype_str}")
    np_dtype, is_complex = _DTYPES[dtype_str]

    comps = 2 if is_complex else 1
    itemsize = np.dtype(np_dtype).itemsize
    byte_offset = offset_samples * comps * itemsize
    n_read = -1 if count is None else count * comps

    raw = np.fromfile(dp, dtype=np_dtype, count=n_read, offset=byte_offset)
    if is_complex:
        raw = raw[: (len(raw) // 2) * 2]
        iq = np.empty(len(raw) // 2, dtype=np.complex64)
        iq.real = raw[0::2]
        iq.imag = raw[1::2]
    else:
        iq = raw.astype(np.complex64)

    # normalize integer formats to ~[-1, 1)
    if np.issubdtype(np_dtype, np.integer):
        iq /= float(2 ** (8 * itemsize - 1))
    return iq.astype(np.complex64), meta


def save_iq(path, iq, sample_rate, center_freq=0.0, extra_global=None):
    """Write complex64 IQ + a SigMF sidecar as cf32_le (processed-output type).

    Writes <stem>.sigmf-data and <stem>.sigmf-meta. cf32_le is SigMF's canonical
    complex-float type; we use it for processed output (vs hackrfpy's raw ci8).
    """
    p = Path(path)
    stem = p.with_suffix("")
    data_p = stem.with_suffix(".sigmf-data")
    meta_p = stem.with_suffix(".sigmf-meta")

    iq = np.asarray(iq, dtype=np.complex64)
    # interleave I,Q as float32 little-endian
    inter = np.empty(len(iq) * 2, dtype="<f4")
    inter[0::2] = iq.real
    inter[1::2] = iq.imag
    inter.tofile(data_p)

    g = {
        "core:datatype": "cf32_le",
        "core:sample_rate": float(sample_rate),
        "core:version": "1.0.0",
    }
    if extra_global:
        g.update(extra_global)
    meta = {
        "global": g,
        "captures": [{"core:sample_start": 0, "core:frequency": float(center_freq)}],
        "annotations": [],
    }
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=2)
    return str(data_p), str(meta_p)
