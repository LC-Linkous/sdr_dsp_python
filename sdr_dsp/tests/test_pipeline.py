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
            .add(lambda b: fm_demod(b, fs, 75000), "demod"))
    out = np.concatenate(pipe.run())
    manual = fm_demod(fir_apply(iq, taps), fs, 75000)
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


def _noise(n=100_000, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)


def test_run_flushes_lookahead_stages():
    # regression: StreamingResampler holds back a look-ahead tail until the
    # stream ends, and Pipeline.run() never asked for it -- a 1/8 resample of
    # 100k samples yielded 604 blocks-worth of samples where resample_poly
    # gives 625. The pipeline must produce the whole-record result exactly.
    from sdr_dsp.stream import StreamingResampler
    from sdr_dsp.core import resample_poly
    iq = _noise()
    ref = resample_poly(iq, 1, 8)
    for bs in (4096, 1000, 7777):
        pipe = Pipeline(ArraySource(iq, 1e6, block_size=bs)).add(
            StreamingResampler(1, 8), "resample")
        out = np.concatenate(pipe.run())
        assert len(out) == len(ref), f"block_size {bs}: {len(out)} vs {len(ref)}"
        np.testing.assert_array_equal(out, ref)


def test_flush_threads_tail_through_later_stages_and_taps():
    # the upstream tail must pass through every later stage (so a second
    # look-ahead stage sees it before being flushed itself) and every tap
    from sdr_dsp.stream import StreamingResampler, StreamingFIR
    from sdr_dsp.core import resample_poly
    iq = _noise()
    taps = design_lowpass(50e3, 1e6, num_taps=31)
    seen = []
    pipe = (Pipeline(ArraySource(iq, 1e6, block_size=4096))
            .add(StreamingResampler(1, 2), "r2")
            .add(StreamingFIR(taps), "lpf")
            .tap(lambda b: seen.append(len(b)), "watch")
            .add(StreamingResampler(1, 4), "r4"))
    out = np.concatenate(pipe.run())
    ref = resample_poly(fir_apply(resample_poly(iq, 1, 2), taps), 1, 4)
    assert len(out) == len(ref)
    assert sum(seen) == len(iq) // 2          # the tap saw the r2 tail too
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_flush_delivers_to_sink_and_stream():
    from sdr_dsp.stream import StreamingResampler
    from sdr_dsp.core import resample_poly
    iq = _noise()
    n_ref = len(resample_poly(iq, 1, 8))
    got = []
    Pipeline(ArraySource(iq, 1e6, block_size=4096)).add(
        StreamingResampler(1, 8)).run(sink=got.append)
    assert sum(len(b) for b in got) == n_ref
    streamed = list(Pipeline(ArraySource(iq, 1e6, block_size=4096)).add(
        StreamingResampler(1, 8)).stream())
    assert sum(len(b) for b in streamed) == n_ref


def test_max_blocks_does_not_flush():
    # a stream cut short by max_blocks has not ENDED; the look-ahead stays put
    from sdr_dsp.stream import StreamingResampler
    iq = _noise()
    pipe = Pipeline(ArraySource(iq, 1e6, block_size=4096)).add(
        StreamingResampler(1, 8))
    assert len(pipe.run(max_blocks=3)) == 3
    assert len(list(pipe.stream(max_blocks=3))) == 3


def test_flush_is_idempotent_and_safe_without_flushable_stages():
    from sdr_dsp.stream import StreamingResampler
    iq = _noise(20_000)
    pipe = Pipeline(ArraySource(iq, 1e6, block_size=4096)).add(
        StreamingResampler(1, 8))
    pipe.run()
    assert pipe.flush() == []                 # already drained
    plain = Pipeline(ArraySource(iq, 1e6)).add(lambda b: np.abs(b), "mag")
    plain.run()
    assert plain.flush() == []


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
