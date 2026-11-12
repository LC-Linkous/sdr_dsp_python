"""tools/fetch_sample_data.py: download release-asset payloads next to their
sidecars and verify them against the sidecar's sdr_dsp:sha256.

Exercised entirely through file:// URLs and a temp directory, so no network
and no 80 MB files: a "release" is a directory of small fake payloads.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "fetch_sample_data.py"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_sample_data", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _release(tmp_path: Path, payloads: dict[str, bytes]):
    """Make a fake release dir + a sample dir with sidecars carrying the hashes."""
    rel = tmp_path / "release"
    rel.mkdir()
    sample = tmp_path / "sample"
    sample.mkdir()
    for stem, data in payloads.items():
        (rel / f"{stem}.iq").write_bytes(data)
        meta = {"global": {"core:datatype": "ci8", "core:sample_rate": 8e6,
                           "sdr_dsp:sha256": hashlib.sha256(data).hexdigest()},
                "captures": [{"core:sample_start": 0, "core:frequency": 98.1e6}],
                "annotations": []}
        (sample / f"{stem}.sigmf-meta").write_text(json.dumps(meta))
    return rel.as_uri() + "/", sample


def test_fetches_missing_verifies_present_and_rejects_bad(tmp_path, capsys):
    tool = _load()
    base, sample = _release(tmp_path, {"a": b"\x01\x02" * 5000, "b": b"\x7f\x80" * 7000})
    # first run: both missing -> fetched and verified
    assert tool.main(["--base-url", base, "--sample-dir", str(sample), "-q"]) == 0
    assert (sample / "a.iq").read_bytes() == b"\x01\x02" * 5000
    assert not list(sample.glob("*.part-*"))                 # temp files cleaned up
    # second run: nothing to do
    assert tool.main(["--base-url", base, "--sample-dir", str(sample)]) == 0
    assert capsys.readouterr().out.count("ok ") == 2
    # corrupt one payload: --check reports it and exits 1 without touching it
    (sample / "b.iq").write_bytes(b"garbage")
    assert tool.main(["--base-url", base, "--sample-dir", str(sample), "--check"]) == 1
    assert (sample / "b.iq").read_bytes() == b"garbage"
    # a normal run re-downloads it
    assert tool.main(["--base-url", base, "--sample-dir", str(sample), "-q"]) == 0
    assert (sample / "b.iq").read_bytes() == b"\x7f\x80" * 7000


def test_mismatched_asset_is_not_installed(tmp_path):
    # the release serves different bytes than the sidecar hash: refuse, and
    # leave no payload behind -- a wrong-tag download must not look like success
    tool = _load()
    base, sample = _release(tmp_path, {"a": b"good" * 1000})
    (tmp_path / "release" / "a.iq").write_bytes(b"evil" * 1000)
    with pytest.raises(SystemExit, match="sha256 mismatch"):
        tool.main(["--base-url", base, "--sample-dir", str(sample), "-q"])
    assert not (sample / "a.iq").exists()
    assert not list(sample.glob("*.part-*"))


def test_only_and_missing_url_and_sidecar_without_hash(tmp_path):
    tool = _load()
    base, sample = _release(tmp_path, {"a": b"x" * 100, "b": b"y" * 100})
    assert tool.main(["--base-url", base, "--sample-dir", str(sample), "--only", "a", "-q"]) == 0
    assert (sample / "a.iq").exists() and not (sample / "b.iq").exists()
    with pytest.raises(SystemExit, match="no sidecar named"):
        tool.main(["--base-url", base, "--sample-dir", str(sample), "--only", "zzz"])
    # unreachable asset: reported, exit 1, other work continues
    assert tool.main(["--base-url", (tmp_path / "nowhere").as_uri() + "/",
                      "--sample-dir", str(sample), "-q"]) == 1
    # a sidecar with no hash is refused rather than fetched blind
    meta = json.loads((sample / "b.sigmf-meta").read_text())
    del meta["global"]["sdr_dsp:sha256"]
    (sample / "b.sigmf-meta").write_text(json.dumps(meta))
    with pytest.raises(SystemExit, match="refusing to fetch blind"):
        tool.main(["--base-url", base, "--sample-dir", str(sample), "--only", "b"])


def _open_fds() -> int | None:
    for d in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(list(Path(d).iterdir()))
        except OSError:
            continue
    return None


@pytest.mark.parametrize("failure", ["missing_url", "hash_mismatch"])
def test_failed_download_closes_and_removes_temp_file(tmp_path, failure):
    """A failed download must leave no temp file AND no open descriptor.

    The descriptor half is what broke Windows CI: the temp fd was only wrapped
    after urlopen succeeded, so a missing URL leaked it and Windows refused
    the cleanup unlink. Linux deletes an open file without complaint, so on
    Linux/macOS the leak is caught by counting open descriptors instead.
    """
    tool = _load()
    dest = tmp_path / "out" / "x.iq"
    if failure == "missing_url":
        url, sha = (tmp_path / "nowhere.iq").as_uri(), "0" * 64
        expected = OSError            # URLError is an OSError
    else:
        src = tmp_path / "src.iq"
        src.write_bytes(b"\x01" * 100)
        url, sha = src.as_uri(), "0" * 64
        expected = SystemExit
    before = _open_fds()
    for _ in range(5):                # a leak of one fd per call shows as 5
        with pytest.raises(expected):
            tool.download(url, dest, sha, quiet=True)
    after = _open_fds()
    assert list(dest.parent.iterdir()) == [], "temp file left behind"
    assert not dest.exists()
    if before is not None:
        assert after == before, f"leaked {after - before} file descriptor(s)"


def test_real_sidecars_carry_hashes_and_check_mode_runs():
    # the shipped sidecars must be fetchable: each has a sha256. --check on the
    # real directory exits 0 or 1 depending on whether payloads are present,
    # but must never raise.
    tool = _load()
    metas = tool.sidecars(tool.SAMPLE_DIR)
    assert {m.stem for m in metas} == {"fm_98.1MHz_8Msps", "fm_103.7MHz_8Msps"}
    for m in metas:
        assert len(tool.expected_sha(m)) == 64
    assert tool.main(["--check", "-q"]) in (0, 1)
