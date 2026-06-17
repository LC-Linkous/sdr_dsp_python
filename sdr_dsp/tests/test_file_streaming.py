"""Tests for FileSource true streaming: blocks() reads incrementally from disk
without loading the whole file, while staying byte-identical to a full load and
honoring offset/count. Also covers the iq_info cheap-inspection helper.
"""

import os
import tempfile

import numpy as np
import pytest

from sdr_dsp.io import save_iq, load_iq, iq_info
from sdr_dsp.sources import FileSource


@pytest.fixture
def ramp_file():
    d = tempfile.mkdtemp()
    n = 200_000
    iq = (np.arange(n) + 1j * np.arange(n)).astype(np.complex64)
    path = os.path.join(d, "ramp.iq")
    save_iq(path, iq, 2e6, center_freq=100e6)
    return path, n


def test_iq_info_reads_without_loading(ramp_file):
    path, n = ramp_file
    info = iq_info(path)
    assert info["total_samples"] == n
    assert info["sample_rate"] == 2e6
    assert info["center_freq"] == 100e6
    assert info["is_complex"] is True


def test_metadata_available_before_data_load(ramp_file):
    path, n = ramp_file
    src = FileSource(path, block_size=50_000)
    # constructing the source must not load the samples
    assert src._iq is None
    assert src.n_samples == n
    assert src.sample_rate == 2e6


def test_blocks_stream_without_loading_all(ramp_file):
    path, n = ramp_file
    src = FileSource(path, block_size=50_000)
    blocks = list(src.blocks())
    # streamed in 50k chunks, and the full array was never cached
    assert [len(b) for b in blocks] == [50_000, 50_000, 50_000, 50_000]
    assert src._iq is None


def test_streamed_equals_full_load(ramp_file):
    path, n = ramp_file
    src = FileSource(path, block_size=33_333)
    streamed = np.concatenate(list(src.blocks()))
    full, _ = load_iq(path)
    assert np.array_equal(streamed, full)


def test_iq_property_loads_lazily(ramp_file):
    path, n = ramp_file
    src = FileSource(path)
    assert src._iq is None          # not loaded yet
    arr = src.iq                    # triggers load
    assert src._iq is not None      # now cached
    full, _ = load_iq(path)
    assert np.array_equal(arr, full)


def test_offset_and_count_under_streaming(ramp_file):
    path, n = ramp_file
    src = FileSource(path, block_size=30_000, offset_samples=50_000,
                     count=80_000)
    assert src.n_samples == 80_000
    streamed = np.concatenate(list(src.blocks()))
    assert len(streamed) == 80_000
    full, _ = load_iq(path)
    assert np.array_equal(streamed, full[50_000:130_000])


def test_blocks_after_iq_loaded_uses_memory(ramp_file):
    path, n = ramp_file
    src = FileSource(path, block_size=50_000)
    _ = src.iq                      # force the in-memory path
    blocks = list(src.blocks())     # should slice memory, not re-read
    assert np.concatenate(blocks).shape[0] == n


def test_read_streams_from_offset(ramp_file):
    path, n = ramp_file
    src = FileSource(path, offset_samples=10_000)
    r = src.read(1_000)
    full, _ = load_iq(path)
    assert np.array_equal(r, full[10_000:11_000])


def test_len_reflects_available_samples(ramp_file):
    path, n = ramp_file
    assert len(FileSource(path)) == n
    assert len(FileSource(path, count=1234)) == 1234
