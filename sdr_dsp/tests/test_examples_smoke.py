"""Guards against the two example-layer regressions found during polish.

1. Peak normalization destroyed by a startup transient. A causal FIR, a phase
   discriminator, and a zero-initialized IIR all produce a brief burst far
   above the program material. Scaling by the raw maximum let one such sample
   set the file's scale, pushing real audio toward zero -- a click, then
   silence. write_wav now scales to a high percentile instead.

2. Example scripts that never imported. Several used
   `sys.path.insert(0, "src")` together with `from src.sdr_dsp import ...`,
   which cannot resolve: running `python examples/foo.py` puts examples/ on
   sys.path[0], not the cwd, so `src` is not a package from there. The
   scripts failed for everyone, and nothing in the suite noticed.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

from sdr_dsp.sinks import write_wav

EXAMPLES = sorted((Path(__file__).resolve().parent.parent / "examples").glob("*.py"))


# --------------------------------------------------------------------------
# 1. write_wav must survive a startup transient
# --------------------------------------------------------------------------
def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def test_write_wav_survives_startup_transient(tmp_path):
    """A single huge leading sample must not bury the program material."""
    rng = np.random.default_rng(0)
    audio = 0.05 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
    audio += 0.001 * rng.standard_normal(48000)
    audio[0] = 50.0          # the transient: 1000x the program material

    out = tmp_path / "t.wav"
    write_wav(out, audio, 48000)
    pcm = _read_wav(out).astype(float)

    body_rms = np.sqrt(np.mean(pcm[100:] ** 2))
    # With peak normalization this lands near 33 counts (-60 dBFS). With
    # percentile normalization the tone should sit in a usable range.
    assert body_rms > 3000, f"program material buried at RMS {body_rms:.0f}"


def test_write_wav_normal_audio_unchanged(tmp_path):
    """Without a transient, normalization still fills the range as before."""
    audio = 0.3 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000)
    out = tmp_path / "t.wav"
    write_wav(out, audio, 48000)
    pcm = _read_wav(out).astype(float)
    assert 0.80 * 32767 < np.max(np.abs(pcm)) <= 32767


def test_write_wav_all_zero_audio(tmp_path):
    """A silent buffer must not divide by zero."""
    out = tmp_path / "t.wav"
    write_wav(out, np.zeros(1000), 48000)
    assert np.all(_read_wav(out) == 0)


@pytest.mark.parametrize("bad", [
    dict(headroom=0.0), dict(headroom=1.5), dict(percentile=0.0),
    dict(percentile=101.0), dict(sample_rate=0),
])
def test_write_wav_rejects_bad_params(tmp_path, bad):
    kwargs = dict(sample_rate=48000)
    kwargs.update(bad)
    rate = kwargs.pop("sample_rate")
    with pytest.raises(ValueError):
        write_wav(tmp_path / "t.wav", np.ones(100), rate, **kwargs)


def test_write_wav_rejects_empty_and_nonfinite(tmp_path):
    with pytest.raises(ValueError):
        write_wav(tmp_path / "a.wav", np.array([]), 48000)
    with pytest.raises(ValueError):
        write_wav(tmp_path / "b.wav", np.array([1.0, np.nan]), 48000)


# --------------------------------------------------------------------------
# 2. every example script must at least parse and resolve its imports
# --------------------------------------------------------------------------
def test_examples_found():
    assert len(EXAMPLES) > 30, "examples/ did not glob as expected"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_parses(path):
    ast.parse(path.read_text(), filename=str(path))


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_imports_the_installed_package(path):
    """Examples import `sdr_dsp`; they never manipulate sys.path.

    The project installs as a package (`uv sync`), so examples should import
    it the same way any downstream user would. Path shims are banned: they
    hide a missing install behind a layout assumption, and the cwd-relative
    form used previously silently broke whenever an example was run from
    anywhere other than the project directory.
    """
    text = path.read_text()
    assert "from src.sdr_dsp" not in text, (
        f"{path.name} imports 'src.sdr_dsp', which never resolves when run as "
        "a script; import 'sdr_dsp' instead"
    )
    assert "sys.path" not in text, (
        f"{path.name} manipulates sys.path; examples run against the installed "
        "package (`uv sync`), so the import should be a plain 'sdr_dsp' import"
    )


OPTIONAL_MODULES = ("hackrfpy", "sounddevice", "matplotlib")


def _missing_optional_deps(path):
    """Optional extras an example imports that aren't installed here.

    Determined by reading the example's imports rather than by matching text
    on stderr. A script that guards its own optional import and exits with a
    friendly message -- which the collection scripts do, because a raw
    ModuleNotFoundError is a poor thing to hand someone who just wanted to
    record a capture -- never produces the string a stderr check looks for,
    so the skip silently turns into a failure.
    """
    text = path.read_text()
    missing = []
    for mod in OPTIONAL_MODULES:
        pattern = rf"^\s*(?:import {mod}\b|from {mod}[\s.])"
        if re.search(pattern, text, re.MULTILINE):
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
    return missing


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_imports_cleanly(path, tmp_path):
    """Import each example under a non-main run name, from a foreign cwd.

    Catches broken import lines without executing main(). Run from tmp_path so
    a cwd-relative sys.path entry cannot accidentally rescue a bad import.
    """
    missing = _missing_optional_deps(path)
    if missing:
        pytest.skip(f"needs optional extra(s): {', '.join(missing)}")

    code = (
        "import runpy, sys\n"
        f"sys.argv = ['{path.name}']\n"
        f"runpy.run_path({str(path)!r}, run_name='_smoke_')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        err = r.stderr.strip().splitlines()
        pytest.fail(f"{path.name} failed to import:\n" + "\n".join(err[-6:]))


# --------------------------------------------------------------------------
# 3. capture_health must distinguish a real capture from an empty one
# --------------------------------------------------------------------------
def _fm_capture(amp_counts, n=200_000, fs=2e6, noise=0.4, seed=0):
    """Synthesize an FM capture at a given ADC level, quantized to ci8."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    msg = np.sin(2 * np.pi * 440 * t)
    z = np.exp(1j * 2 * np.pi * 75e3 * np.cumsum(msg) / fs) * amp_counts
    z = z + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * noise
    i8 = np.clip(np.round(z.real), -127, 127)
    q8 = np.clip(np.round(z.imag), -127, 127)
    return ((i8 + 1j * q8) / 127.0).astype(np.complex64)


def test_capture_health_accepts_a_real_capture():
    from sdr_dsp.core import capture_health
    h = capture_health(_fm_capture(100), 2e6, channel_bw=100e3)
    assert h["ok"], h["reasons"]
    assert 80 < h["adc_counts"] <= 127
    assert h["channel_excess_db"] > 10


def test_capture_health_rejects_a_noise_floor_capture():
    """The exact failure mode of the originally-shipped sample file."""
    from sdr_dsp.core import capture_health
    rng = np.random.default_rng(1)
    n = 200_000
    z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1.2
    iq = ((np.round(z.real) + 1j * np.round(z.imag)) / 127.0).astype(np.complex64)
    h = capture_health(iq, 2e6, channel_bw=100e3)
    assert not h["ok"]
    assert h["adc_counts"] < 10
    assert len(h["reasons"]) >= 1


def test_capture_health_flags_a_signal_free_but_well_scaled_capture():
    """Plenty of ADC range in use, but no carrier: still not demodulable."""
    from sdr_dsp.core import capture_health
    rng = np.random.default_rng(2)
    n = 200_000
    z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 30.0
    iq = ((np.round(z.real) + 1j * np.round(z.imag)) / 127.0).astype(np.complex64)
    h = capture_health(iq, 2e6, channel_bw=100e3)
    assert h["adc_counts"] > 50, "this capture uses plenty of ADC range"
    assert not h["ok"], "but there is no carrier in it"
    assert any("carrier" in r for r in h["reasons"])


def test_capture_health_handles_empty_and_missing_channel_bw():
    from sdr_dsp.core import capture_health
    empty = capture_health(np.zeros(0, dtype=np.complex64), 2e6)
    assert not empty["ok"] and empty["adc_counts"] == 0.0
    no_bw = capture_health(_fm_capture(100), 2e6, channel_bw=None)
    assert no_bw["channel_excess_db"] is None


def test_capture_health_measures_level_per_component():
    """I and Q are separate converters; level is max(|I|,|Q|), not |I+jQ|.

    A fully clipped sample is I=127, Q=127. Its complex magnitude is 179.6,
    which is not a level any 8-bit converter can produce -- reporting it
    would put "ADC counts" above full scale and make a clipping threshold
    fire on signals that are merely loud.
    """
    from sdr_dsp.core import capture_health
    # full-scale on the int8 grid after the loader's /128 normalization
    clipped = np.full(4096, (127 + 127j) / 128.0, dtype=np.complex64)
    h = capture_health(clipped, 2e6)
    assert h["adc_counts"] == pytest.approx(127.0, abs=0.5), (
        f"clipped capture reported {h['adc_counts']:.1f} counts; the complex "
        "magnitude would give ~179.6")


def test_capture_health_counts_track_a_known_amplitude():
    from sdr_dsp.core import capture_health
    for counts in (4.0, 30.0, 100.0):
        n = 4096
        z = counts * np.exp(1j * np.linspace(0, 40 * np.pi, n)) / 127.0
        h = capture_health(z.astype(np.complex64), 2e6)
        assert h["adc_counts"] == pytest.approx(counts, rel=0.02)


# --------------------------------------------------------------------------
# 4. the development corpus plan must be internally consistent
# --------------------------------------------------------------------------
DEV_TOOL = Path(__file__).resolve().parent.parent / "tools" / "collect_dev_data.py"


def _load_dev_tool():
    pytest.importorskip("hackrfpy",
                        reason="optional extra: uv sync --extra examples-hackrf")
    import importlib.util
    spec = importlib.util.spec_from_file_location("_collect_dev_data", DEV_TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_collect_dev_data"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_dev_tool_exists():
    assert DEV_TOOL.exists(), "tools/collect_dev_data.py is missing"


def test_dev_plan_is_well_formed():
    dev = _load_dev_tool()
    plan = dev.build_plan()
    assert len(plan) >= 5, "expected at least five datasets"
    seen_names = set()
    for n, ds in plan.items():
        assert {"slug", "title", "why", "captures"} <= set(ds), f"data_{n}"
        assert ds["captures"], f"data_{n} has no captures"
        for cap in ds["captures"]:
            assert {"name", "rate", "secs", "expect", "purpose"} <= set(cap)
            assert cap["expect"] in ("signal", "weak", "empty")
            assert dev.LNA_STEPS[0] <= 0
            # every capture name must be unique across the whole corpus, since
            # they become filenames that may be copied into one folder
            assert cap["name"] not in seen_names, f"duplicate {cap['name']}"
            seen_names.add(cap["name"])


def test_dev_plan_rates_are_within_hackrf_limits():
    dev = _load_dev_tool()
    for n, ds in dev.build_plan().items():
        for cap in ds["captures"]:
            assert 2e6 <= cap["rate"] <= 20e6, (
                f"data_{n}/{cap['name']}: {cap['rate']} sps is outside the "
                "HackRF's 2-20 Msps range")


def test_dev_plan_offsets_fit_inside_the_captured_span():
    """An offset capture is useless if the station falls outside the span."""
    dev = _load_dev_tool()
    for n, ds in dev.build_plan().items():
        for cap in ds["captures"]:
            off = abs(cap.get("freq_offset", 0))
            if off:
                assert off < cap["rate"] / 2, (
                    f"data_{n}/{cap['name']}: station at {off/1e3:g} kHz is "
                    f"outside the {cap['rate']/2e3:g} kHz half-span")


def test_dev_gain_offsets_snap_to_the_hardware_grid():
    dev = _load_dev_tool()
    ref = {"lna": 16, "vga": 20, "amp": True, "total_db": 36,
           "counts": 90.0}
    for offset in (-99, -30, -12, 0, 6, 12, 24):
        lna, vga, amp, applied = dev.gain_for(ref, offset)
        assert lna in dev.LNA_STEPS and vga in dev.VGA_STEPS, (
            f"offset {offset} gave lna={lna} vga={vga}, off grid")
        assert 0 <= lna <= 40 and 0 <= vga <= 62
        # minimum gain means the amp comes off too; otherwise the amp
        # setting is part of the calibrated reference and must not drift
        assert amp is (False if offset <= -99 else True)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_does_not_peak_normalize_audio(path):
    """No example should scale audio by its raw maximum.

    A single transient -- a filter delay line settling, a discriminator fed a
    near-zero-magnitude sample -- then sets the scale for the whole file and
    buries the program material. write_wav handles this with a percentile;
    examples should route through it rather than reimplement the fragile
    version. In a streaming loop the same idiom also swings the gain once per
    block, which is audible as pumping.
    """
    text = path.read_text()
    assert "peak = np.max(np.abs(" not in text, (
        f"{path.name} normalizes by the raw peak; use write_wav, or a fixed "
        "scale for streaming output")
    assert "wave.open" not in text, (
        f"{path.name} inlines wave-file plumbing; use "
        "sdr_dsp.sinks.write_wav instead")


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_has_no_undefined_names(path):
    """Catch references to names that do not exist.

    The import smoke test above loads each module but never calls main(), so
    a stale reference inside a function body -- a variable left behind after
    an edit, say -- survives import and only fails when someone actually runs
    the example. pyflakes finds those statically. Unused imports are not
    policed here; an example may import something to show it exists.
    """
    pyflakes = pytest.importorskip("pyflakes.api",
                                   reason="pyflakes not installed")
    from pyflakes.reporter import Reporter
    import io
    out, err = io.StringIO(), io.StringIO()
    pyflakes.checkPath(str(path), Reporter(out, err))
    bad = [ln for ln in out.getvalue().splitlines()
           if "undefined name" in ln or "local variable" in ln
           and "referenced before assignment" in ln]
    assert not bad, f"{path.name}:\n" + "\n".join(bad)

def test_fm_receiver_decimates_before_demod(tmp_path):
    """fm_receiver decimates the channel to FM_RATE before the discriminator
    (mirrors the live path). Verify the WAV comes out at 48 kHz with real
    program audio and the 19 kHz pilot suppressed by the audio lowpass --
    i.e. the faster path did not change the output's character."""
    import json
    import wave

    fs = 2_000_000
    n = int(fs * 1.0)
    t = np.arange(n) / fs
    # 440 Hz tone + the 19 kHz stereo pilot in the composite
    msg = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.09 * np.sin(2 * np.pi * 19_000 * t)
    z = 0.4 * np.exp(1j * 2 * np.pi * 75e3 * np.cumsum(msg) / fs)
    i8 = np.empty(2 * n, dtype=np.int8)
    i8[0::2] = np.clip(np.round(z.real * 128), -128, 127).astype(np.int8)
    i8[1::2] = np.clip(np.round(z.imag * 128), -128, 127).astype(np.int8)
    iqp = tmp_path / "fm.iq"
    i8.tofile(iqp)
    (tmp_path / "fm.sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "ci8", "core:sample_rate": fs},
        "captures": [{"core:sample_start": 0, "core:frequency": 98e6}],
        "annotations": []}))

    out = tmp_path / "out.wav"
    example = (Path(__file__).resolve().parent.parent / "examples") / "fm_receiver.py"
    r = subprocess.run([sys.executable, str(example), str(iqp),
                        "--out", str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "for demod" in r.stdout, "decimate/resample-before-demod step missing"

    w = wave.open(str(out))
    assert w.getframerate() == 48_000
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16) / 32768
    spec = np.abs(np.fft.rfft(pcm * np.hanning(len(pcm))))
    f = np.fft.rfftfreq(len(pcm), 1 / 48_000)

    def band(lo, hi):
        m = (f > lo) & (f < hi)
        return 10 * np.log10(np.mean(spec[m] ** 2) + 1e-20)

    # 440 Hz program present; 19 kHz pilot cut well below it by the 15 kHz LPF
    assert band(300, 600) > band(17_000, 21_000) + 30, "pilot not suppressed"


def test_fm_receiver_integer_decimation_on_common_rate(tmp_path):
    """At 2 Msps the intermediate 250 kHz rate divides evenly, so the log
    should say 'decimated' (pure stride), not 'resampled'."""
    import json

    fs = 2_000_000
    n = int(fs * 0.5)
    t = np.arange(n) / fs
    msg = 0.6 * np.sin(2 * np.pi * 440 * t) + 0.09 * np.sin(2 * np.pi * 19_000 * t)
    z = 0.4 * np.exp(1j * 2 * np.pi * 75e3 * np.cumsum(msg) / fs)
    i8 = np.empty(2 * n, dtype=np.int8)
    i8[0::2] = np.clip(np.round(z.real * 128), -128, 127).astype(np.int8)
    i8[1::2] = np.clip(np.round(z.imag * 128), -128, 127).astype(np.int8)
    iqp = tmp_path / "fm.iq"
    i8.tofile(iqp)
    (tmp_path / "fm.sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "ci8", "core:sample_rate": fs},
        "captures": [{"core:sample_start": 0, "core:frequency": 98e6}],
        "annotations": []}))
    out = tmp_path / "out.wav"
    r = subprocess.run([sys.executable, str((Path(__file__).resolve().parent.parent / "examples") / "fm_receiver.py"),
                        str(iqp), "--out", str(out)],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "decimated 2 Msps -> 250 kHz" in r.stdout, r.stdout
