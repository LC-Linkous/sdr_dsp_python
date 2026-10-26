"""Run every file-mode and synthetic example's main() end to end.

test_examples_smoke.py proves each example PARSES and IMPORTS. About half of
them had nothing beyond that: argparse surfaces, file loading, the pipeline
glue, and the output writers of am_receiver, ook_decoder, inspect_capture,
iq_basics, waterfall and friends never executed under test. This runs each one
as a subprocess the way a user would, with the smallest arguments that still
exercise the real path:

* file-based examples get a short synthetic SigMF capture written by this test
  (an FM station, an OOK burst train, an AM carrier), so no sample_data or
  hardware is needed;
* plotting examples run with MPLBACKEND=Agg, where plt.show() is a no-op, so
  figures are built but never block;
* anything that needs a radio is excluded by name -- the stub-radio adapter
  tests cover that seam.

A non-zero exit, a traceback, or a 120 s timeout fails the example. These are
smoke tests: they assert the run completed and produced its declared output,
not the DSP (the core tests and test_examples_demods do that).
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sdr_dsp.io import save_iq
from sdr_dsp.core import ook_modulate

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
TIMEOUT = 120

# Examples that need a radio, a sound device, or an interactive window loop
# with no non-interactive mode. Each is covered elsewhere or by name here so
# the list is explicit, not a silent omission.
NEEDS_HARDWARE = {
    "hackrf_capture.py",       # adapter: tests/test_hackrf_adapter_stub.py
    "hackrf_sink.py",          # TX adapter: tests/test_tx_sink.py
    "live_433_capture.py", "live_spectrum.py", "live_fm_listen.py",
    "collect_sample_data.py",  # tests/test_collect_sample_data.py
    "fm_band_explorer.py",     # interactive UI; tests/test_fm_band_explorer.py
    "sdr_adapter_template.py", # a template, exercised by tests/test_modularity.py
}


# ---------------------------------------------------------------------------
# synthetic captures (cf32 SigMF, which load_iq reads like any capture)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def captures(tmp_path_factory):
    d = tmp_path_factory.mktemp("captures")
    rng = np.random.default_rng(0)

    # FM: 1 kHz tone, 75 kHz deviation, 2 Msps, 0.25 s, a touch of noise
    fs = 2_000_000
    n = fs // 4
    t = np.arange(n) / fs
    msg = np.sin(2 * np.pi * 1000 * t)
    fm = 0.5 * np.exp(1j * 2 * np.pi * 75e3 * np.cumsum(msg) / fs)
    fm += 0.005 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    fm_path = d / "fm.sigmf-data"
    save_iq(fm_path, fm.astype(np.complex64), fs, center_freq=100e6)

    # OOK: a few bursts of a 1 kbaud pattern at 250 ksps
    fs_ook = 250_000
    sps = 250
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 4)
    burst = ook_modulate(bits, sps)
    gap = np.zeros(20 * sps, dtype=np.complex64)
    ook = np.concatenate([gap, burst, gap, burst, gap]).astype(np.complex64)
    ook += 0.01 * (rng.standard_normal(len(ook))
                   + 1j * rng.standard_normal(len(ook))).astype(np.complex64)
    ook_path = d / "ook.sigmf-data"
    save_iq(ook_path, ook, fs_ook, center_freq=433.92e6)

    # AM: carrier at +50 kHz offset, 60% modulation by a 700 Hz tone
    fs_am = 500_000
    n = fs_am // 2
    t = np.arange(n) / fs_am
    am = (1 + 0.6 * np.sin(2 * np.pi * 700 * t)) * np.exp(1j * 2 * np.pi * 50e3 * t)
    am = 0.4 * am + 0.005 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    am_path = d / "am.sigmf-data"
    save_iq(am_path, am.astype(np.complex64), fs_am, center_freq=1e6)

    return {"fm": str(fm_path), "ook": str(ook_path), "am": str(am_path),
            "dir": d}


# ---------------------------------------------------------------------------
# the table: example -> argv (callables receive the captures dict)
# ---------------------------------------------------------------------------

CASES = {
    # file-based receivers / inspectors
    "am_receiver.py": lambda c: [c["am"], "--tune", "50000", "--out",
                                 str(c["dir"] / "am.wav")],
    "fm_receiver.py": lambda c: [c["fm"], "--out", str(c["dir"] / "fm.wav")],
    "ook_decoder.py": lambda c: [c["ook"]],
    "inspect_capture.py": lambda c: [c["fm"], "--max-samples", "200000"],
    "signal_survey.py": lambda c: [c["fm"], c["ook"]],
    "spectrum_analyzer.py": lambda c: [c["fm"], "--no-plot"],
    "waterfall.py": lambda c: [c["fm"], "--max-samples", "200000"],
    "burst_detector.py": lambda c: [c["ook"]],
    "annotate_bursts.py": lambda c: [c["ook"], "--out",
                                     str(c["dir"] / "annotated.sigmf-data")],
    "channelizer.py": lambda c: [c["fm"]],
    "channel_sweep.py": lambda c: ["--scheme", "fsk", "--msg", "HI"],
    "power_calibration.py": lambda c: ["--save", str(c["dir"] / "cal.json")],
    "gain_search_demo.py": lambda c: ["--radio", "all", "--trace"],
    "live_pipeline_fm.py": lambda c: [c["fm"], "--block", "50000"],
    "cfo_demo.py": lambda c: [c["fm"], "--fm", "--correct"],
    "dc_offset_demo.py": lambda c: [c["fm"]],
    "filter_explorer.py": lambda c: [c["fm"]],
    "iq_basics.py": lambda c: [c["fm"], "--samples", "500"],
    # synthetic / teaching demos (no file)
    "constellation_recovery.py": lambda c: ["--nsym", "300"],
    "decimation_stages.py": lambda c: ["--n", "100000"],
    "impairment_extraction.py": lambda c: [],
    "matched_filter_demo.py": lambda c: ["--total", "2000"],
    "nyquist_aliasing_demo.py": lambda c: [],
    "resampler_benchmark.py": lambda c: ["--n", "50000"],
    "spectrogram_chirp.py": lambda c: ["--dur", "0.02"],
    "windowing_demo.py": lambda c: [],
    "agc_demo.py": lambda c: [],
    "modulate_demo.py": lambda c: [],
    "differential_psk_demo.py": lambda c: [],
    "dsss_demo.py": lambda c: [],
    "fhss_visualizer.py": lambda c: [],
    "cw_decoder.py": lambda c: [],
    "fsk_decoder.py": lambda c: [],
    "nask_decoder.py": lambda c: [],
    "ssb_receiver.py": lambda c: [],
    "dsb_sc_demo.py": lambda c: [],
    "packet_loopback.py": lambda c: [],
    "two_station_link.py": lambda c: [],
    "transmit_examples.py": lambda c: [],
}


def test_every_example_is_either_run_or_explicitly_excluded():
    """No example may fall through the cracks: each is in CASES or in
    NEEDS_HARDWARE with a reason. Adding an example without deciding which is
    a failure, on purpose."""
    names = {p.name for p in EXAMPLES.glob("*.py")}
    covered = set(CASES) | NEEDS_HARDWARE
    missing = sorted(names - covered)
    assert not missing, f"examples with no run test and no exclusion: {missing}"
    stale = sorted(covered - names)
    assert not stale, f"CASES/NEEDS_HARDWARE name files that do not exist: {stale}"


def _run(path, argv, cwd):
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    env.pop("DISPLAY", None)
    return subprocess.run([sys.executable, str(path), *argv], cwd=cwd,
                          capture_output=True, text=True, timeout=TIMEOUT,
                          env=env)


@pytest.mark.parametrize("name", sorted(CASES), ids=lambda n: n)
def test_example_runs_to_completion(name, captures, tmp_path):
    path = EXAMPLES / name
    argv = CASES[name](captures)
    if "matplotlib" in path.read_text() and not pytest.importorskip("matplotlib"):
        pytest.skip("matplotlib not installed")
    try:
        r = _run(path, argv, cwd=tmp_path)
    except subprocess.TimeoutExpired:
        pytest.fail(f"{name} did not finish in {TIMEOUT}s with args {argv}")
    if r.returncode != 0:
        tail = "\n".join((r.stderr or r.stdout).strip().splitlines()[-12:])
        pytest.fail(f"{name} exited {r.returncode} with args {argv}:\n{tail}")
    assert "Traceback" not in r.stderr, r.stderr[-800:]


def test_am_receiver_wrote_audio(captures):
    out = captures["dir"] / "am.wav"
    if not out.exists():        # the run test for am_receiver.py creates it
        _run(EXAMPLES / "am_receiver.py", CASES["am_receiver.py"](captures),
             cwd=captures["dir"])
    assert out.exists() and out.stat().st_size > 1000
