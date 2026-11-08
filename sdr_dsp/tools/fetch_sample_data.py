"""Fetch the large raw captures that live in release assets, not in git.

`sample_data/hackrf_one/` holds SigMF sidecars for two 80 MB ci8 captures
whose payloads are distributed as GitHub release assets (see
sample_data/README.md). This script downloads each payload next to its sidecar
and verifies it against the `sdr_dsp:sha256` the sidecar already carries, so a
truncated or substituted download cannot pass. A payload that is present and
verifies is left alone.

Usage (from sdr_dsp/):
    uv run python tools/fetch_sample_data.py                 # everything in hackrf_one/
    uv run python tools/fetch_sample_data.py --only fm_98.1MHz_8Msps
    uv run python tools/fetch_sample_data.py --release v0.3.0-captures
    uv run python tools/fetch_sample_data.py --base-url file:///mnt/mirror/   # any URL prefix
    uv run python tools/fetch_sample_data.py --check         # verify only, download nothing

Exit status 0 when every requested payload is present and verified, 1 otherwise.
Standard library only: no requests, no hackrfpy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SAMPLE_DIR = HERE.parent / "sample_data" / "hackrf_one"

# Where the payloads live. The release tag is the one knob that changes when
# captures are re-published; the pattern is GitHub's for release assets.
REPO = "LC-Linkous/sdr_dsp_python"
DEFAULT_RELEASE = "v0.3.0-captures"
RELEASE_URL = "https://github.com/{repo}/releases/download/{release}/"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sidecars(sample_dir: Path, only: str | None = None) -> list[Path]:
    metas = sorted(sample_dir.glob("*.sigmf-meta"))
    if only is not None:
        metas = [m for m in metas if m.stem == only]
        if not metas:
            raise SystemExit(f"no sidecar named {only}.sigmf-meta in {sample_dir}")
    return metas


def expected_sha(meta_path: Path) -> str:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sha = meta.get("global", {}).get("sdr_dsp:sha256")
    if not sha:
        raise SystemExit(f"{meta_path.name}: no sdr_dsp:sha256 in the sidecar; "
                         "cannot verify a download, refusing to fetch blind")
    return sha.lower()


def verify(payload: Path, sha: str) -> bool:
    return payload.exists() and sha256_of(payload) == sha


def download(url: str, dest: Path, sha: str, quiet: bool = False) -> None:
    """Stream url into dest via a temp file; verify before moving into place."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".part-")
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(url) as resp, open(tmp_fd, "wb") as out:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            done = 0
            while True:
                block = resp.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if not quiet and total:
                    sys.stderr.write(f"\r  {dest.name}: {done / 1e6:7.1f} / {total / 1e6:.1f} MB")
            if not quiet and total:
                sys.stderr.write("\n")
        got = sha256_of(tmp)
        if got != sha:
            raise SystemExit(f"{dest.name}: sha256 mismatch after download\n"
                             f"  expected {sha}\n  got      {got}\n"
                             "  (wrong release tag, or a changed asset); nothing installed")
        shutil.move(str(tmp), str(dest))
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", default=DEFAULT_RELEASE,
                    help=f"GitHub release tag holding the assets (default {DEFAULT_RELEASE})")
    ap.add_argument("--base-url", default=None,
                    help="URL prefix the payload file names are appended to; "
                         "overrides --release (any scheme urllib accepts, incl. file://)")
    ap.add_argument("--only", default=None, metavar="STEM",
                    help="fetch one capture by sidecar stem, e.g. fm_98.1MHz_8Msps")
    ap.add_argument("--sample-dir", type=Path, default=SAMPLE_DIR,
                    help="directory holding the sidecars (default sample_data/hackrf_one)")
    ap.add_argument("--check", action="store_true",
                    help="verify what is present; download nothing")
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args(argv)

    base = args.base_url or RELEASE_URL.format(repo=REPO, release=args.release)
    if not base.endswith("/"):
        base += "/"

    failures = 0
    for meta in sidecars(args.sample_dir, args.only):
        payload = meta.with_suffix(".iq")
        sha = expected_sha(meta)
        if verify(payload, sha):
            if not args.quiet:
                print(f"ok        {payload.name}")
            continue
        if payload.exists():
            print(f"BAD       {payload.name}: present but sha256 does not match the sidecar"
                  + ("" if args.check else "; re-downloading"))
        elif not args.quiet:
            print(f"missing   {payload.name}")
        if args.check:
            failures += 1
            continue
        url = base + payload.name
        try:
            download(url, payload, sha, quiet=args.quiet)
        except urllib.error.URLError as exc:
            print(f"FAILED    {payload.name}: {exc.reason} ({url})")
            failures += 1
            continue
        print(f"fetched   {payload.name}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
