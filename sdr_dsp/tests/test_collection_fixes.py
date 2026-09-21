"""The three fixes from the first real corpus run, pinned.

1. The gain-ladder verdict flagged a legitimately-weak capture (5 counts vs
   a 3-count target) as UNEXPECTEDLY STRONG, because a ratio tolerance is
   wrong down where the ~2-count quantization floor dominates.
2. The quiet-reference search only looked inside the broadcast band, but at
   the test location the in-band "empty" spots were occupied; the emptiest
   real estate is the guard band just below it.
3. Every probe re-printed hackrfpy's identical sub-8Msps warning, ~16 times
   per station, burying the results.
"""

import importlib.util
import io
import logging
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from sdr_dsp.sources.probe import dedup_warnings

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(f"_{name}", TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. verdict tolerance at low counts
# ---------------------------------------------------------------------------
dev = _load("collect_dev_data")


def _health(ok, counts):
    return {"ok": ok, "adc_counts": counts, "channel_excess_db": 20.0,
            "reasons": []}


def test_weak_capture_at_quantization_floor_is_ok():
    """The exact case from the run: target 3.0, measured 5.0, health still
    'ok' because a strong station can't be turned down far enough. Must NOT
    be flagged."""
    v = dev.judge_capture("weak", _health(True, 5.0), at_min_gain=False,
                          target_counts=3.0)
    assert v.startswith("ok"), v


def test_weak_capture_genuinely_too_strong_is_flagged():
    """Well above the absolute floor and not at min gain -> still caught."""
    v = dev.judge_capture("weak", _health(True, 40.0), at_min_gain=False,
                          target_counts=3.0)
    assert "UNEXPECTEDLY STRONG" in v


def test_weak_capture_below_health_is_always_ok():
    v = dev.judge_capture("weak", _health(False, 5.0), at_min_gain=False,
                          target_counts=3.0)
    assert v.startswith("ok (intentionally weak)")


def test_weak_tolerance_uses_ratio_when_target_is_large():
    """For a comfortable target the 2.5x ratio governs, not the +3 floor."""
    assert dev.judge_capture("weak", _health(True, 45.0), False,
                             target_counts=20.0).startswith("ok")   # <= 50
    assert "UNEXPECTEDLY" in dev.judge_capture(
        "weak", _health(True, 60.0), False, target_counts=20.0)     # > 50


def test_empty_verdicts_unchanged():
    assert dev.judge_capture("empty", _health(False, 2.0), False).startswith(
        "ok (intentionally empty)")
    assert "HAS SIGNAL" in dev.judge_capture("empty", _health(True, 30.0),
                                             at_min_gain=False)
    assert dev.judge_capture("empty", _health(True, 30.0),
                             at_min_gain=True).startswith("ok")


def test_signal_verdicts_unchanged():
    assert dev.judge_capture("signal", _health(True, 70.0), False) == "ok"
    assert dev.judge_capture("signal", _health(False, 70.0), False) == \
        "UNEXPECTEDLY EMPTY"


# ---------------------------------------------------------------------------
# 2. quiet search reaches below the broadcast band
# ---------------------------------------------------------------------------
pf = _load("preflight_collection")


def test_quiet_can_be_found_below_the_band():
    """Every in-band spot occupied; the only quiet air is below 88 MHz.
    The preflight must sweep down there and find it."""
    # stations fill the band; nothing below 88
    stations = {f: 12.0 for f in np.arange(88.3e6, 108e6, 0.5e6)}
    radio = pf.SimulatedRadio(stations)
    out = io.StringIO()
    ok, report = pf.run_preflight(radio, (88e6, 108e6), out=out)
    assert report["quiet_hz"] is not None, out.getvalue()
    assert report["quiet_hz"] < 88e6, (
        f"quiet at {report['quiet_hz'] / 1e6} MHz -- should be below the band")


def test_below_band_carrier_not_ranked_as_a_station():
    """A carrier below 88 MHz must not become a station candidate (ranking
    stays in-band) but must still block a quiet pick on top of it."""
    radio = pf.SimulatedRadio({87.0e6: 20.0, 98.5e6: 25.0})
    out = io.StringIO()
    ok, report = pf.run_preflight(radio, (88e6, 108e6), out=out)
    for s in report["stations"]:
        assert s["freq"] >= 88e6, f"ranked an out-of-band signal: {s['freq']}"
    if report["quiet_hz"] is not None:
        assert abs(report["quiet_hz"] - 87.0e6) > 400e3


# ---------------------------------------------------------------------------
# 3. warning dedupe
# ---------------------------------------------------------------------------
def test_repeated_bang_warnings_collapse_to_one():
    buf = io.StringIO()
    with redirect_stdout(buf):
        for _ in range(16):
            with dedup_warnings():
                print("[!] sample_rate=2e+06 is below the recommended 8e+06")
                print("    probe lna=8 -> 70 counts")
    text = buf.getvalue()
    assert text.count("[!]") == 1
    assert text.count("probe lna=8") == 16


def test_distinct_warnings_each_shown_once():
    buf = io.StringIO()
    with redirect_stdout(buf):
        with dedup_warnings():
            print("[!] warning A")
            print("[!] warning B")
            print("[!] warning A")
    text = buf.getvalue()
    assert text.count("warning A") == 1
    assert text.count("warning B") == 1


def test_non_bang_lines_never_filtered():
    """Only [!]-prefixed lines dedupe; identical normal lines all pass."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        with dedup_warnings():
            print("same line")
            print("same line")
    assert buf.getvalue().count("same line") == 2


def test_logging_warnings_dedupe_through_bound_handler():
    """The real path: hackrfpy warns via logging, not a direct print.

    A StreamHandler bound to the true stderr (what any logging.basicConfig or
    hackrfpy's own setup installs) bypasses redirect_stderr entirely, so the
    stream wrapper alone let the same warning print once per probe. The
    logging-layer filter must collapse it to one across many probes.
    """
    log = logging.getLogger("hackrfpy")
    sink = io.StringIO()
    handler = logging.StreamHandler(sink)  # bound to sink now, before redirect
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    prev_propagate = log.propagate
    prev_level = log.level
    log.propagate = False
    log.setLevel(logging.INFO)  # let INFO through the level gate so the
    #                             filter's level scoping is what's under test
    try:
        for _ in range(16):  # ~one probe each
            with dedup_warnings():
                log.warning("[!] sample_rate=2e+06 is below the recommended 8e+06")
                log.info("probe lna=8 -> 70 counts")  # INFO must NOT dedupe
    finally:
        log.removeHandler(handler)
        log.propagate = prev_propagate
        log.setLevel(prev_level)
    text = sink.getvalue()
    assert text.count("[!]") == 1, "warning should collapse to one across probes"
    assert text.count("probe lna=8") == 16, "INFO progress must not be filtered"


def test_dedup_filter_removed_after_context():
    """dedup_warnings must not leave its filter attached to the logger."""
    log = logging.getLogger("hackrfpy")
    before = list(log.filters)
    with dedup_warnings():
        pass
    assert list(log.filters) == before
