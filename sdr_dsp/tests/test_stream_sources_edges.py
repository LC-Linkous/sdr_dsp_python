"""Edge-case coverage for sdr_dsp.stream, sdr_dsp.sources and sdr_dsp.sinks.

The streaming stages promise two things the main suite only checks in the
happy path: every stage validates its construction arguments, and every stage
has a ``reset()`` that forgets carried state so a retune or stream restart
does not bleed the previous signal's history into the new one. The tests here
pin both, plus the empty-block pass-through, the resampler's pass-through
flush, the Pipeline/PipelineStats text forms, and the lazy-loaded top-level
subpackages.
"""

import numpy as np
import pytest

import sdr_dsp
from sdr_dsp.core import design_lowpass
from sdr_dsp.stream import (Pipeline, PipelineStats, StreamingFIR,
                            StreamingDecimator, StreamingMixer,
                            StreamingFMDemod, StreamingResampler,
                            StreamingDeemphasis)
from sdr_dsp.stream.stages import _roll_tail
from sdr_dsp.sources import ArraySource, FileSource
from sdr_dsp.sinks import LoopbackSink
from sdr_dsp.io.sigmf import save_iq

FS = 1_000_000.0
EMPTY = np.zeros(0, dtype=np.complex64)


def _tone(f_hz, n, fs=FS):
    t = np.arange(n) / fs
    return np.exp(2j * np.pi * f_hz * t).astype(np.complex64)


# -- package: PEP 562 lazy subpackages --------------------------------------

def test_top_level_lazy_subpackages_resolve_once():
    import importlib
    import sys
    # Force the __getattr__ path even if an earlier import already cached it.
    for name in ("io", "sinks", "stream", "link"):
        sdr_dsp.__dict__.pop(name, None)
        mod = getattr(sdr_dsp, name)
        assert mod is sys.modules[f"sdr_dsp.{name}"]
        assert mod is importlib.import_module(f"sdr_dsp.{name}")
        assert name in sdr_dsp.__dict__                 # cached after first access
    with pytest.raises(AttributeError):
        sdr_dsp.no_such_thing


# -- construction validation -------------------------------------------------

def test_streaming_fir_rejects_bad_taps():
    with pytest.raises(ValueError, match="non-empty 1-D"):
        StreamingFIR([])
    with pytest.raises(ValueError, match="non-empty 1-D"):
        StreamingFIR(np.ones((2, 2)))


def test_streaming_decimator_rejects_factor_below_one():
    with pytest.raises(ValueError, match="factor"):
        StreamingDecimator(0)


def test_streaming_resampler_rejects_nonpositive_ratio():
    with pytest.raises(ValueError, match="up and down"):
        StreamingResampler(0, 2)
    with pytest.raises(ValueError, match="up and down"):
        StreamingResampler(2, 0)


# -- reset() forgets state ---------------------------------------------------

def test_decimator_reset_restarts_phase():
    dec = StreamingDecimator(4)
    x = np.arange(6, dtype=np.complex64)
    dec(x)                                           # consumes 6 -> skip 2 next
    dec.reset()
    out = dec(np.arange(8, dtype=np.complex64))
    assert list(out.real) == [0, 4]                  # starts at 0 again


def test_mixer_reset_restarts_lo_phase():
    mix = StreamingMixer(100e3, FS)
    block = np.ones(256, dtype=np.complex64)
    first = mix(block)
    mix(block)                                       # advances the LO
    mix.reset()
    again = mix(block)
    assert np.allclose(first, again)
    assert mix(EMPTY).size == 0                      # empty block passes through


def test_fm_demod_reset_drops_carried_sample():
    dem = StreamingFMDemod(deviation_hz=75e3, sample_rate=FS)
    dem(_tone(10e3, 64))
    dem.reset()
    # after reset the first block has nothing to difference against: a
    # one-sample block yields no output rather than a bogus phase step
    assert dem(_tone(10e3, 1)).size == 0
    assert dem(_tone(10e3, 1)).size == 1             # now primed


def test_deemphasis_reset_and_empty_block():
    de = StreamingDeemphasis(FS)
    step = np.ones(32)
    a = de(step)
    de(step)
    de.reset()
    b = de(step)
    assert np.allclose(a, b)
    assert de(np.zeros(0)).size == 0


def test_resampler_reset_matches_fresh_instance():
    r1 = StreamingResampler(3, 2)
    r2 = StreamingResampler(3, 2)
    x = _tone(20e3, 600)
    r1(x)
    r1.reset()
    assert np.allclose(r1(x), r2(x))


def test_resampler_flush_pass_through_and_untouched():
    # 1:1 is a pass-through with no filter -- flush has nothing to add
    r = StreamingResampler(2, 2)
    assert r.flush().size == 0
    # never fed: nothing buffered either
    assert StreamingResampler(3, 2).flush().size == 0


def test_roll_tail_zero_length_history():
    tail = np.zeros(0, dtype=np.complex64)
    assert _roll_tail(tail, np.ones(4, dtype=np.complex64)) is tail


# -- pipeline ----------------------------------------------------------------

def test_pipeline_repr_describe_and_stats_str():
    src = ArraySource(_tone(1e3, 1000), FS, block_size=250)
    taps = design_lowpass(100e3, FS, num_taps=15)
    seen = []
    pipe = (Pipeline(src)
            .add(StreamingFIR(taps), "lpf")
            .tap(lambda b: seen.append(len(b)), "count"))
    assert "Pipeline(2 stages" in repr(pipe)
    desc = pipe.describe()
    assert "step lpf" in desc and "tap  count" in desc
    results, stats = pipe.run(profile=True)
    assert len(results) == 4 and seen == [250, 250, 250, 250]
    text = str(stats)
    assert text.startswith("4 blocks, 1,000 input samples")
    assert "lpf" in text and "ms total" in text
    assert isinstance(stats, PipelineStats)


def test_pipeline_tap_observes_without_replacing():
    src = ArraySource(np.ones(10, dtype=np.complex64), FS, block_size=10)
    pipe = Pipeline(src).tap(lambda b: None)
    out = pipe.process_block(np.full(10, 3 + 0j, dtype=np.complex64))
    assert np.all(out == 3)                          # untouched by the tap


# -- sources -----------------------------------------------------------------

def test_arraysource_read_truncates():
    src = ArraySource(np.arange(10, dtype=np.complex64), FS)
    assert len(src.read(4)) == 4
    assert len(src.read(100)) == 10


def test_filesource_repr_and_short_tail(tmp_path):
    iq = _tone(1e3, 1000)
    save_iq(tmp_path / "r.iq", iq, FS, center_freq=100e6)
    src = FileSource(tmp_path / "r.sigmf-meta", block_size=300)
    text = repr(src)
    assert "1000 samples" in text and "1 Msps" in text and "100 MHz" in text
    blocks = list(src.blocks())
    assert [len(b) for b in blocks] == [300, 300, 300, 100]
    assert np.allclose(np.concatenate(blocks), iq)


def test_filesource_blocks_stop_when_file_shrinks_underneath(tmp_path):
    # n_samples is sized from the data file when the source is opened. If the
    # file is truncated afterwards (a capture being re-recorded under a live
    # reader), blocks() must stop at the data that exists rather than loop on
    # empty reads until the stale total is reached.
    iq = _tone(1e3, 1000)
    save_iq(tmp_path / "t.iq", iq, FS)
    src = FileSource(tmp_path / "t.sigmf-meta", block_size=300)
    assert src.n_samples == 1000
    data = tmp_path / "t.sigmf-data"
    data.write_bytes(data.read_bytes()[: 400 * 8])  # keep 400 cf32 samples
    got = sum(len(b) for b in src.blocks())
    assert got == 400


# -- sinks -------------------------------------------------------------------

def test_loopback_sink_repr():
    sink = LoopbackSink(FS)
    sink.transmit(np.ones(5, dtype=np.complex64))
    assert "5 samples buffered" in repr(sink) and "1 transmits" in repr(sink)


def test_plot_spectrogram_rejects_too_short_input():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    from sdr_dsp.sinks import plot_spectrogram
    with pytest.raises(ValueError, match="not enough samples"):
        plot_spectrogram(_tone(1e3, 100), FS, nfft=1024)


def test_plot_spectrogram_show_calls_pyplot(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from sdr_dsp.sinks import plot_spectrogram
    called = []
    monkeypatch.setattr(plt, "show", lambda: called.append(True))
    fig, ax = plot_spectrogram(_tone(1e3, 8192), FS, nfft=256, show=True)
    assert called == [True]
    plt.close(fig)
