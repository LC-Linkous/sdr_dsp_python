"""Tests for the streaming Pipeline: stage chaining, taps, profiling, the lazy
stream() bridge, and that it orchestrates the pure core without altering results.
"""

import numpy as np

from sdr_dsp import Pipeline
from sdr_dsp.sources import ArraySource
from sdr_dsp.core import design_lowpass, fir_apply, fm_demod


def _fm_signal(fs=1_000_000, n=80_000):
    t = np.arange(n) / fs
    msg = np.cos(2 * np.pi * 2000 * t)
    return np.exp(1j * 2 * np.pi * 75000 * np.cumsum(msg) / fs).astype(
        np.complex64)


def test_pipeline_matches_manual():
    # a pipeline must produce exactly what calling the functions by hand does
    fs = 1_000_000
    iq = _fm_signal(fs)
    taps = design_lowpass(100e3, fs, num_taps=101)
    src = ArraySource(iq, fs, block_size=len(iq))   # one block
    pipe = (Pipeline(src)
            .add(lambda b: fir_apply(b, taps), "filter")
            .add(lambda b: fm_demod(b, 75000, fs), "demod"))
    out = np.concatenate(pipe.run())
    manual = fm_demod(fir_apply(iq, taps), 75000, fs)
    assert np.allclose(out, manual, atol=1e-6)


def test_tap_does_not_alter_flow():
    fs = 1_000_000
    iq = _fm_signal(fs)
    seen = []
    pipe = (Pipeline(ArraySource(iq, fs, block_size=len(iq)))
            .add(lambda b: b * 2, "double")
            .tap(lambda b: seen.append(b.copy()))
            .add(lambda b: b + 1, "offset"))
    out = np.concatenate(pipe.run())
    # the tap saw the post-double, pre-offset data; flow continued unchanged
    assert len(seen) == 1
    assert np.allclose(seen[0], iq * 2, atol=1e-6)
    assert np.allclose(out, iq * 2 + 1, atol=1e-6)


def test_tap_fires_once_per_block():
    fs = 1_000_000
    iq = _fm_signal(fs, n=80_000)
    count = []
    pipe = (Pipeline(ArraySource(iq, fs, block_size=20_000))
            .tap(lambda b: count.append(1)))
    pipe.run()
    assert sum(count) == 4   # 80000 / 20000


def test_sink_receives_results():
    fs = 1_000_000
    iq = _fm_signal(fs)
    collected = []
    pipe = (Pipeline(ArraySource(iq, fs, block_size=20_000))
            .add(lambda b: np.abs(b), "mag"))
    result = pipe.run(sink=collected.append)
    assert result is None          # sink mode returns None
    assert len(collected) == 4


def test_profile_reports_stages():
    fs = 1_000_000
    iq = _fm_signal(fs)
    taps = design_lowpass(100e3, fs, num_taps=51)
    pipe = (Pipeline(ArraySource(iq, fs, block_size=20_000))
            .add(lambda b: fir_apply(b, taps), "filter"))
    _, stats = pipe.run(profile=True)
    assert stats.blocks == 4
    assert "filter" in stats.per_stage_seconds


def test_max_blocks_limits():
    fs = 1_000_000
    iq = _fm_signal(fs)
    pipe = Pipeline(ArraySource(iq, fs, block_size=20_000))
    out = pipe.run(max_blocks=2)
    assert len(out) == 2


def test_stream_is_lazy():
    fs = 1_000_000
    iq = _fm_signal(fs)
    pipe = (Pipeline(ArraySource(iq, fs, block_size=20_000))
            .add(lambda b: np.abs(b), "mag"))
    gen = pipe.stream(max_blocks=2)
    blocks = list(gen)
    assert len(blocks) == 2


def test_describe_is_inspectable():
    fs = 1_000_000
    iq = _fm_signal(fs)
    pipe = (Pipeline(ArraySource(iq, fs))
            .add(lambda b: b, "a")
            .tap(lambda b: None, "watch"))
    text = pipe.describe()
    assert "step a" in text and "tap  watch" in text


def test_process_block_directly():
    # process_block lets an external loop (e.g. a GUI) drive one block at a time
    fs = 1_000_000
    pipe = Pipeline(ArraySource(np.ones(10, dtype=np.complex64), fs))
    pipe.add(lambda b: b * 3, "triple")
    out = pipe.process_block(np.ones(5, dtype=np.complex64))
    assert np.allclose(out, 3.0)
