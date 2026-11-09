"""Tests for SigMF I/O and FileSource: round-trip fidelity, ci8 decode, and
the FileSource adapter exposing rate/freq from the sidecar.
"""

import json
import numpy as np

from sdr_dsp.io.sigmf import save_iq, load_iq
from sdr_dsp.sources import FileSource, ArraySource


def test_cf32_roundtrip_lossless(tmp_path):
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(2000)
         + 1j * rng.standard_normal(2000)).astype(np.complex64)
    save_iq(tmp_path / "p.sigmf-data", x, sample_rate=2e6, center_freq=100e6)
    back, meta = load_iq(tmp_path / "p.sigmf-meta")
    assert np.max(np.abs(x - back)) == 0.0
    assert meta["global"]["core:sample_rate"] == 2e6
    assert meta["captures"][0]["core:frequency"] == 100e6
    assert meta["global"]["core:datatype"] == "cf32_le"


def test_load_ci8_capture(tmp_path):
    # simulate a hackrfpy ci8 recording
    rng = np.random.default_rng(0)
    ci8 = rng.integers(-128, 128, 4000, dtype=np.int8)
    ci8.tofile(tmp_path / "cap.iq")
    json.dump(
        {"global": {"core:datatype": "ci8", "core:sample_rate": 2_000_000.0},
         "captures": [{"core:frequency": 96_900_000.0}]},
        open(tmp_path / "cap.sigmf-meta", "w"))
    iq, meta = load_iq(tmp_path / "cap.iq")
    assert iq.dtype == np.complex64
    assert len(iq) == 2000
    assert iq.real.min() >= -1.0 and iq.real.max() < 1.0   # normalized


def test_filesource_exposes_metadata(tmp_path):
    x = np.ones(1000, dtype=np.complex64)
    save_iq(tmp_path / "s.sigmf-data", x, sample_rate=8e6, center_freq=433.9e6)
    fs = FileSource(tmp_path / "s.sigmf-meta")
    assert fs.sample_rate == 8e6
    assert fs.center_freq == 433.9e6
    assert len(fs) == 1000


def test_filesource_blocks(tmp_path):
    x = np.ones(10000, dtype=np.complex64)
    save_iq(tmp_path / "b.sigmf-data", x, sample_rate=2e6)
    fs = FileSource(tmp_path / "b.sigmf-meta", block_size=4096)
    total = sum(len(b) for b in fs.blocks())
    assert total == 10000


def test_arraysource_protocol():
    x = np.ones(5000, dtype=np.complex64)
    s = ArraySource(x, sample_rate=1e6, center_freq=100e6, block_size=1000)
    assert s.sample_rate == 1e6
    assert sum(len(b) for b in s.blocks()) == 5000


def test_sources_satisfy_protocol():
    # FileSource and ArraySource must actually satisfy the runtime-checkable
    # IQSource protocol -- this is the contract the DSP core relies on.
    from sdr_dsp.sources import IQSource
    a = ArraySource(np.ones(100, dtype=np.complex64), sample_rate=1e6)
    assert isinstance(a, IQSource)


def test_filesource_blocks_reassemble_exactly(tmp_path):
    # blocks() must lose no samples at boundaries, even when block_size doesn't
    # divide the length evenly.
    x = (np.arange(10_007) + 1j * np.arange(10_007)).astype(np.complex64)
    save_iq(tmp_path / "r.sigmf-data", x, sample_rate=2e6)
    fs = FileSource(tmp_path / "r.sigmf-meta", block_size=1000)
    rebuilt = np.concatenate(list(fs.blocks()))
    assert len(rebuilt) == len(x)
    assert np.array_equal(rebuilt, x)


def test_load_unknown_datatype_errors(tmp_path):
    import json
    import pytest
    np.zeros(100, dtype=np.int8).tofile(tmp_path / "bad.iq")
    json.dump({"global": {"core:datatype": "not_a_real_type",
                          "core:sample_rate": 1e6}, "captures": [{}]},
              open(tmp_path / "bad.sigmf-meta", "w"))
    with pytest.raises(ValueError):
        load_iq(tmp_path / "bad.iq")


# -- path handling and range edges (second audit) ----------------------------

def test_save_iq_keeps_dotted_stems_intact(tmp_path):
    # pathlib treats everything after the LAST dot as the suffix, so two chained
    # with_suffix calls turned "fm_98.1MHz_8Msps.iq" into "fm_98.sigmf-data" --
    # the project's own file names have dots, and a neighbouring recording
    # could share the shortened name. Every form must round-trip.
    iq = np.arange(8, dtype=np.complex64)
    for given, stem in (("fm_98.1MHz_8Msps.iq", "fm_98.1MHz_8Msps"),
                        ("fm_98.1MHz_8Msps", "fm_98.1MHz_8Msps"),
                        ("plain.iq", "plain"), ("a.b.c.sigmf-data", "a.b.c")):
        save_iq(tmp_path / given, iq, 2e6)
        assert (tmp_path / f"{stem}.sigmf-data").exists(), given
        assert (tmp_path / f"{stem}.sigmf-meta").exists(), given
        back, _ = load_iq(tmp_path / given)
        np.testing.assert_array_equal(back, iq)


def test_load_iq_offset_past_end_is_empty_not_an_error(tmp_path):
    save_iq(tmp_path / "x.iq", np.ones(100, dtype=np.complex64), 2e6)
    assert load_iq(tmp_path / "x.iq", offset_samples=1000)[0].size == 0
    assert load_iq(tmp_path / "x.iq", offset_samples=1000, count=10)[0].size == 0
    assert load_iq(tmp_path / "x.iq", offset_samples=100)[0].size == 0     # exactly at the end
    assert load_iq(tmp_path / "x.iq", offset_samples=99)[0].size == 1
    import pytest
    with pytest.raises(ValueError, match="offset_samples"):
        load_iq(tmp_path / "x.iq", offset_samples=-1)
    src = FileSource(tmp_path / "x.iq", offset_samples=1000)
    assert list(src.blocks()) == [] and len(src) == 0


def test_file_source_read_respects_count(tmp_path):
    save_iq(tmp_path / "x.iq", np.arange(300, dtype=np.complex64), 2e6)
    src = FileSource(tmp_path / "x.iq", count=50, offset_samples=200)
    assert len(src) == 50
    got = src.read(1000)
    assert len(got) == 50 and got[0] == 200 and got[-1] == 249
    assert len(src.read(10)) == 10
    assert src.read(0).size == 0
