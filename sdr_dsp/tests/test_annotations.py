"""Tests for SigMF annotation round-trip: the detect -> label -> save -> reload
loop, the Annotation dataclass, and the find_bursts bridge.
"""

import os
import tempfile

import numpy as np
import pytest

from sdr_dsp.io import (save_iq, read_annotations, Annotation,
                        bursts_to_annotations, read_meta)
from sdr_dsp.core import find_bursts


def test_annotation_to_from_sigmf_lossless():
    a = Annotation(sample_start=1000, sample_count=500,
                   freq_lower_edge=1e6, freq_upper_edge=1.1e6,
                   label="test", comment="a note")
    d = a.to_sigmf()
    assert d["core:sample_start"] == 1000
    assert d["core:sample_count"] == 500
    assert d["core:label"] == "test"
    back = Annotation.from_sigmf(d)
    assert back == a


def test_annotation_minimal_fields():
    # only the required fields; optional ones stay None and aren't serialized
    a = Annotation(sample_start=0, sample_count=100)
    d = a.to_sigmf()
    assert "core:freq_lower_edge" not in d
    assert "core:label" not in d
    assert Annotation.from_sigmf(d) == a


def test_annotation_preserves_extra_keys():
    # unknown namespaced keys round-trip via `extra`
    d = {"core:sample_start": 5, "core:sample_count": 10,
         "custom:snr_db": 12.5, "custom:decoded": "ABC"}
    a = Annotation.from_sigmf(d)
    assert a.extra["custom:snr_db"] == 12.5
    assert a.to_sigmf()["custom:decoded"] == "ABC"


def test_annotation_time_span():
    a = Annotation(sample_start=48_000, sample_count=48_000)
    start, end = a.time_span(48_000)
    assert start == pytest.approx(1.0)
    assert end == pytest.approx(2.0)


def test_save_and_read_annotations_roundtrip():
    d = tempfile.mkdtemp()
    iq = np.ones(1000, dtype=np.complex64)
    anns = [Annotation(100, 50, label="a"),
            Annotation(500, 80, label="b", freq_lower_edge=1e3)]
    save_iq(os.path.join(d, "x.iq"), iq, 1e6, annotations=anns)
    loaded = read_annotations(os.path.join(d, "x.iq"))
    assert len(loaded) == 2
    assert loaded[0].label == "a"
    assert loaded[1].freq_lower_edge == 1e3


def test_annotations_sorted_by_sample_start():
    d = tempfile.mkdtemp()
    iq = np.ones(1000, dtype=np.complex64)
    # pass out of order; should come back sorted
    anns = [Annotation(800, 10, label="late"),
            Annotation(100, 10, label="early")]
    save_iq(os.path.join(d, "x.iq"), iq, 1e6, annotations=anns)
    loaded = read_annotations(os.path.join(d, "x.iq"))
    assert [a.label for a in loaded] == ["early", "late"]


def test_save_accepts_raw_dicts():
    d = tempfile.mkdtemp()
    iq = np.ones(100, dtype=np.complex64)
    raw = [{"core:sample_start": 0, "core:sample_count": 10, "core:label": "raw"}]
    save_iq(os.path.join(d, "x.iq"), iq, 1e6, annotations=raw)
    loaded = read_annotations(os.path.join(d, "x.iq"))
    assert loaded[0].label == "raw"


def test_save_rejects_bad_annotation_type():
    d = tempfile.mkdtemp()
    iq = np.ones(100, dtype=np.complex64)
    with pytest.raises(TypeError):
        save_iq(os.path.join(d, "x.iq"), iq, 1e6, annotations=["not valid"])


def test_bursts_to_annotations_bridge():
    spans = [(100, 200), (500, 650)]
    anns = bursts_to_annotations(spans, label="burst {i}")
    assert len(anns) == 2
    assert anns[0].label == "burst 0"
    assert anns[0].sample_start == 100
    assert anns[0].sample_count == 100
    assert anns[1].label == "burst 1"
    assert anns[1].sample_count == 150


def test_bursts_to_annotations_static_label():
    # label without {i} is applied verbatim to all
    anns = bursts_to_annotations([(0, 10), (20, 30)], label="signal")
    assert all(a.label == "signal" for a in anns)


def test_full_detect_label_save_reload():
    # the headline workflow, end to end
    fs = 2e6
    n = 200_000
    rng = np.random.default_rng(0)
    sig = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    for start, length in [(20_000, 8_000), (90_000, 4_000)]:
        t = np.arange(length) / fs
        sig[start:start + length] += np.exp(2j * np.pi * 50e3 * t)
    sig = sig.astype(np.complex64)

    spans = find_bursts(sig, min_gap=500, min_len=1000)
    anns = bursts_to_annotations(spans, label="burst {i}")
    d = tempfile.mkdtemp()
    save_iq(os.path.join(d, "cap.iq"), sig, fs, annotations=anns)
    loaded = read_annotations(os.path.join(d, "cap.iq"))
    assert len(loaded) == len(spans) >= 2
    # the reloaded spans match what was detected
    for orig, got in zip(spans, loaded):
        assert got.sample_start == orig[0]
        assert got.sample_count == orig[1] - orig[0]


def test_empty_annotations_still_valid():
    d = tempfile.mkdtemp()
    iq = np.ones(100, dtype=np.complex64)
    save_iq(os.path.join(d, "x.iq"), iq, 1e6)            # no annotations
    assert read_annotations(os.path.join(d, "x.iq")) == []
    meta = read_meta(os.path.join(d, "x.iq"))
    assert meta["annotations"] == []
