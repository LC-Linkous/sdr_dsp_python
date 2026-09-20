"""Promotion from dev_data/ into sample_data/ must not launder a bad capture.

The whole point of scripting this step rather than copying by hand is that
the copy re-validates. A capture declared as containing signal that turns out
to be empty has to be refused at the boundary, because sample_data/ is what
every clone gets and what the examples default to -- which is exactly how a
noise-floor recording came to ship in the first place.
"""

from __future__ import annotations

import hashlib

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "promote_to_sample_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("_promote", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_promote"] = mod
    spec.loader.exec_module(mod)
    return mod


promote = _load()


# --------------------------------------------------------------------------
# fixture corpus
# --------------------------------------------------------------------------
def _write_capture(path, kind, rate=2e6, secs=0.25, offset_hz=0):
    """Write a ci8 capture: a real FM signal, a weak one, or pure noise."""
    n = int(rate * secs)
    t = np.arange(n) / rate
    # STABLE seed from the capture's DEFINING inputs (kind, offset, rate),
    # not from `path`: the path is a per-run pytest tmp dir, so seeding on it
    # -- or on the per-process-randomized builtin hash() -- reseeds the noise
    # every run and occasionally pushes a weak/noise fixture across the
    # health threshold, flipping its verdict (~1 run in 6). hashlib.sha256
    # gives a process-independent seed; the same fixture is byte-identical
    # every run.
    key = f"{kind}|{offset_hz}|{rate}|{secs}".encode()
    seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    if kind == "noise":
        z = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1.2
    else:
        # "weak" must land BELOW the health floor (min_counts=4) so it is
        # correctly a "genuinely bad capture" that fails re-validation --
        # that is the fixture's whole purpose. amp=1.5 sat right at counts=4
        # and flipped with the noise draw; 0.7 gives counts~3 with margin.
        amp = 100.0 if kind == "signal" else 0.7
        msg = np.sin(2 * np.pi * 440 * t)
        z = np.exp(1j * 2 * np.pi * 75e3 * np.cumsum(msg) / rate)
        z = z * np.exp(2j * np.pi * offset_hz * t) * amp
        z = z + (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 0.4
    i8 = np.clip(np.round(z.real), -127, 127).astype(np.int8)
    q8 = np.clip(np.round(z.imag), -127, 127).astype(np.int8)
    inter = np.empty(2 * n, dtype=np.int8)
    inter[0::2], inter[1::2] = i8, q8
    inter.tofile(path)
    Path(str(path).rsplit(".", 1)[0] + ".sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "ci8", "core:sample_rate": rate,
                   "core:version": "1.0.0"},
        "captures": [{"core:sample_start": 0, "core:frequency": 98.5e6}],
        "annotations": []}, indent=2))
    return n


@pytest.fixture
def corpus(tmp_path):
    """A miniature dev_data/ with one good, one offset, one weak capture."""
    dev = tmp_path / "dev_data"
    specs = [
        (1, "reference", "fm_reference", "signal", 0, "signal"),
        (2, "tuning", "fm_offset_p250k", "signal", 250_000, "signal"),
        (4, "gain", "fm_gain_low", "weak", 0, "weak"),
        (5, "negative", "quiet_channel", "noise", 0, "empty"),
    ]
    for num, slug, name, kind, offset, expect in specs:
        folder = dev / f"data_{num}_{slug}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.iq"
        n = _write_capture(path, kind, offset_hz=offset)
        (folder / "manifest.json").write_text(json.dumps({
            "dataset": num, "slug": slug, "title": f"test {slug}",
            "why": "fixture", "station_name": "test", "station_hz": 98.5e6,
            "reference_gain": {"lna": 16, "vga": 20, "counts": 100.0},
            "captures": [{
                "file": path.name, "sigmf": f"{name}.sigmf-meta",
                "purpose": "fixture", "frequency_hz": 98.5e6 - offset,
                "station_offset_hz": offset, "sample_rate": 2e6,
                "seconds": 0.25, "lna_db": 16, "vga_db": 20,
                "samples": n, "size_bytes": path.stat().st_size,
                "expect": expect,
                "health": {"ok": kind == "signal", "adc_counts": 1.0,
                           "channel_excess_db": 1.0, "reasons": []},
                "verdict": "ok",
            }]}, indent=2))
    return dev


@pytest.fixture
def sample_dir(tmp_path, monkeypatch):
    d = tmp_path / "sample_data"
    d.mkdir()
    monkeypatch.setattr(promote, "SAMPLE_DIR", d)
    return d


def _run(dev, argv=()):
    old = sys.argv
    sys.argv = ["promote", "--dev-dir", str(dev), *argv]
    try:
        return promote.main()
    finally:
        sys.argv = old


# --------------------------------------------------------------------------
def test_promotes_the_curated_set(corpus, sample_dir):
    assert _run(corpus) == 0
    for stem in ("fm_2Msps", "fm_offset_250k", "fm_weak", "quiet_2Msps"):
        assert (sample_dir / f"{stem}.iq").exists(), f"{stem} not promoted"
        assert (sample_dir / f"{stem}.sigmf-meta").exists(), (
            f"{stem} promoted without its sidecar; the capture would lose "
            "its sample rate and frequency")


def test_manifest_records_provenance(corpus, sample_dir):
    _run(corpus)
    mani = json.loads((sample_dir / "manifest.json").read_text())
    by_dest = {c["dest"]: c for c in mani["captures"]}
    ref = by_dest["fm_2Msps"]
    assert ref["source"].startswith("dev_data/data_1")
    assert ref["lna_db"] == 16 and ref["vga_db"] == 20
    assert "health_at_capture" in ref and "health_now" in ref
    assert ref["health_now"]["adc_counts"] > 50


def test_offset_capture_is_validated_where_the_station_actually_is(corpus,
                                                                   sample_dir):
    """Checking at DC would wrongly reject a deliberately-tuned-away capture."""
    _run(corpus)
    mani = json.loads((sample_dir / "manifest.json").read_text())
    off = {c["dest"]: c for c in mani["captures"]}["fm_offset_250k"]
    assert off["station_offset_hz"] == 250_000
    assert off["health_now"]["ok"], (
        "offset capture was judged empty; it should be mixed down by its "
        "known offset before the carrier check")


def test_a_capture_that_lost_its_signal_is_rejected(corpus, sample_dir):
    """The failure this whole script exists to prevent."""
    bad = corpus / "data_1_reference" / "fm_reference.iq"
    _write_capture(bad, "noise")
    rc = _run(corpus)
    assert rc == 1, "promotion should report failure"
    assert not (sample_dir / "fm_2Msps.iq").exists(), (
        "an empty capture was promoted into sample_data")
    mani = json.loads((sample_dir / "manifest.json").read_text())
    assert "fm_2Msps" not in {c["dest"] for c in mani["captures"]}


def test_force_overrides_rejection(corpus, sample_dir):
    bad = corpus / "data_1_reference" / "fm_reference.iq"
    _write_capture(bad, "noise")
    _run(corpus, ["--force"])
    assert (sample_dir / "fm_2Msps.iq").exists()


def test_intentionally_empty_fixtures_are_not_rejected(corpus, sample_dir):
    """quiet_channel and fm_weak are meant to fail capture_health."""
    _run(corpus)
    mani = json.loads((sample_dir / "manifest.json").read_text())
    by_dest = {c["dest"]: c for c in mani["captures"]}
    assert not by_dest["quiet_channel" if "quiet_channel" in by_dest
                       else "quiet_2Msps"]["health_now"]["ok"]
    assert by_dest["quiet_2Msps"]["verdict"].startswith("intentionally")
    assert by_dest["fm_weak"]["verdict"].startswith("intentionally")


def test_dry_run_writes_nothing(corpus, sample_dir):
    assert _run(corpus, ["--dry-run"]) == 0
    assert not list(sample_dir.iterdir()), "dry run wrote files"


def test_missing_dataset_is_reported_not_crashed(corpus, sample_dir):
    import shutil
    shutil.rmtree(corpus / "data_4_gain")
    rc = _run(corpus)
    assert rc == 1
    assert (sample_dir / "fm_2Msps.iq").exists(), (
        "one missing dataset should not block the others")


def test_empty_dev_dir_is_reported(tmp_path, sample_dir):
    assert _run(tmp_path / "nothing_here") == 1


def test_readme_is_regenerated_with_the_promoted_files(corpus, sample_dir):
    _run(corpus)
    text = (sample_dir / "README.md").read_text()
    assert "fm_2Msps.iq" in text
    assert "--tune 250000" in text or "--tune 250" in text, (
        "the offset capture's recovery command should be documented")
    assert "on purpose" in text, (
        "the deliberately-empty fixtures need explaining, or someone will "
        "file a bug against them")


def test_promote_list_targets_are_unique():
    dests = [dest for _, _, dest, _ in promote.PROMOTE]
    assert len(dests) == len(set(dests)), "two entries write the same file"
