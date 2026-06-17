"""SigMF metadata + data I/O.

Reads the recordings hackrfpy produces (ci8 = complex int8) and writes
processed output as cf32_le (complex float32, SigMF's canonical RF type), since
sdr_dsp works internally in complex64. This module is the file <-> array bridge;
it has no device knowledge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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


@dataclass
class Annotation:
    """A labeled region of a recording, round-tripped to/from SigMF.

    Maps to a SigMF annotation object. The two required fields locate the region
    in time (sample_start + sample_count); the optional frequency edges locate it
    in frequency, and label/comment/extra carry the human meaning.

    Fields:
        sample_start:     first sample of the region.
        sample_count:     length of the region in samples.
        freq_lower_edge:  lower frequency bound (Hz), or None.
        freq_upper_edge:  upper frequency bound (Hz), or None.
        label:            short label (SigMF core:label), e.g. "key fob burst".
        comment:          longer note (SigMF core:comment).
        extra:            any additional namespaced keys to round-trip verbatim.
    """
    sample_start: int
    sample_count: int
    freq_lower_edge: float | None = None
    freq_upper_edge: float | None = None
    label: str | None = None
    comment: str | None = None
    extra: dict = field(default_factory=dict)

    def to_sigmf(self) -> dict:
        """Serialize to a SigMF annotation dict (core:-namespaced keys)."""
        d = {
            "core:sample_start": int(self.sample_start),
            "core:sample_count": int(self.sample_count),
        }
        if self.freq_lower_edge is not None:
            d["core:freq_lower_edge"] = float(self.freq_lower_edge)
        if self.freq_upper_edge is not None:
            d["core:freq_upper_edge"] = float(self.freq_upper_edge)
        if self.label is not None:
            d["core:label"] = str(self.label)
        if self.comment is not None:
            d["core:comment"] = str(self.comment)
        # preserve any extra namespaced keys verbatim
        d.update(self.extra)
        return d

    @classmethod
    def from_sigmf(cls, d: dict) -> "Annotation":
        """Parse a SigMF annotation dict back into an Annotation.

        Recognized core: keys map to fields; anything else is preserved in
        `extra` so a save -> load round-trip is lossless.
        """
        known = {
            "core:sample_start", "core:sample_count", "core:freq_lower_edge",
            "core:freq_upper_edge", "core:label", "core:comment",
        }
        extra = {k: v for k, v in d.items() if k not in known}
        return cls(
            sample_start=int(d.get("core:sample_start", 0)),
            sample_count=int(d.get("core:sample_count", 0)),
            freq_lower_edge=d.get("core:freq_lower_edge"),
            freq_upper_edge=d.get("core:freq_upper_edge"),
            label=d.get("core:label"),
            comment=d.get("core:comment"),
            extra=extra,
        )

    def time_span(self, sample_rate: float) -> tuple[float, float]:
        """Convenience: (start_seconds, end_seconds) for this region."""
        start = self.sample_start / sample_rate
        return start, start + self.sample_count / sample_rate


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
    # a data path was given (.iq / .sigmf-data / other). If that exact file
    # exists, use it; otherwise try the sibling data extensions, since save_iq
    # always writes .sigmf-data even when handed an .iq name.
    if p.exists():
        return p
    for ext in (".sigmf-data", ".iq"):
        cand = p.with_suffix(ext)
        if cand.exists():
            return cand
    return p


def read_meta(path):
    """Read a .sigmf-meta sidecar into a dict. Accepts the meta or data path."""
    mp = _meta_path(path)
    with open(mp, "r") as f:
        return json.load(f)


def iq_info(path):
    """Inspect a recording WITHOUT loading the IQ data into memory.

    Returns a dict with: meta, datatype, np_dtype, is_complex, itemsize,
    bytes_per_sample, total_samples, sample_rate, center_freq. This is what lets
    a streaming reader know how big the file is and how to seek into it without
    reading the samples. Cheap: it stats the data file and parses the sidecar.
    """
    meta = read_meta(path)
    dp = _data_path(path)
    dtype_str = meta.get("global", {}).get("core:datatype", "ci8")
    if dtype_str not in _DTYPES:
        raise ValueError(f"unsupported SigMF datatype: {dtype_str}")
    np_dtype, is_complex = _DTYPES[dtype_str]
    comps = 2 if is_complex else 1
    itemsize = np.dtype(np_dtype).itemsize
    bytes_per_sample = comps * itemsize
    file_bytes = Path(dp).stat().st_size
    total_samples = file_bytes // bytes_per_sample
    g = meta.get("global", {})
    caps = meta.get("captures", [{}])
    return {
        "meta": meta,
        "data_path": dp,
        "datatype": dtype_str,
        "np_dtype": np_dtype,
        "is_complex": is_complex,
        "itemsize": itemsize,
        "bytes_per_sample": bytes_per_sample,
        "total_samples": int(total_samples),
        "sample_rate": float(g.get("core:sample_rate", 0.0)),
        "center_freq": float(caps[0].get("core:frequency", 0.0)) if caps
        else 0.0,
    }


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


def save_iq(path, iq, sample_rate, center_freq=0.0, extra_global=None,
            annotations=None):
    """Write complex64 IQ + a SigMF sidecar as cf32_le (processed-output type).

    Writes <stem>.sigmf-data and <stem>.sigmf-meta. cf32_le is SigMF's canonical
    complex-float type; we use it for processed output (vs hackrfpy's raw ci8).

    annotations: an optional list of Annotation objects (or raw SigMF annotation
    dicts) to record labeled regions of the capture. They round-trip: load_iq /
    read_annotations read them back as Annotation objects.
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
    # serialize annotations: accept Annotation objects or already-dict form
    ann_list = []
    for a in (annotations or []):
        if isinstance(a, Annotation):
            ann_list.append(a.to_sigmf())
        elif isinstance(a, dict):
            ann_list.append(a)
        else:
            raise TypeError(
                "annotations must be Annotation objects or SigMF dicts, "
                f"got {type(a).__name__}")
    # SigMF wants annotations sorted by sample_start
    ann_list.sort(key=lambda d: d.get("core:sample_start", 0))
    meta = {
        "global": g,
        "captures": [{"core:sample_start": 0,
                      "core:frequency": float(center_freq)}],
        "annotations": ann_list,
    }
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=2)
    return str(data_p), str(meta_p)


def read_annotations(path) -> list:
    """Read the annotations from a recording's sidecar as Annotation objects.

    Accepts any path form (.iq / .sigmf-data / .sigmf-meta). Returns a list of
    Annotation, sorted by sample_start. Empty list if there are none.
    """
    meta = read_meta(path)
    anns = [Annotation.from_sigmf(d) for d in meta.get("annotations", [])]
    anns.sort(key=lambda a: a.sample_start)
    return anns


def bursts_to_annotations(spans, label=None, freq_lower_edge=None,
                          freq_upper_edge=None):
    """Convert find_bursts() output into Annotation objects. The detect->label
    step in one call.

    spans: a list of (start, stop) sample-index pairs (what find_bursts returns).
    label: applied to every burst, optionally with an index suffix if it
           contains "{i}" (e.g. "burst {i}" -> "burst 0", "burst 1", ...).
    freq_lower_edge / freq_upper_edge: optional frequency bounds applied to all.

    Returns a list of Annotation ready to pass to save_iq(annotations=...).
    """
    out = []
    for i, (start, stop) in enumerate(spans):
        lab = None
        if label is not None:
            lab = label.format(i=i) if "{i}" in label else label
        out.append(Annotation(
            sample_start=int(start),
            sample_count=int(stop) - int(start),
            freq_lower_edge=freq_lower_edge,
            freq_upper_edge=freq_upper_edge,
            label=lab,
        ))
    return out
