"""The `@overload`s on the diagnostics/profile flags give callers precise
return types. Pinned by running mypy on a snippet that uses `reveal_type`,
so a regression to the bare union (or a wrong branch) fails here rather than
in a downstream type-check.

Skipped when mypy is not importable (it is a dev dependency, not a runtime
one), so the suite still runs from a bare `pip install sdr-dsp`.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("mypy")

SRC = Path(__file__).resolve().parents[1] / "src"

SNIPPET = """
import numpy as np
from sdr_dsp import carrier_recovery, symbol_sync, Pipeline, ArraySource

iq = np.zeros(8, dtype=np.complex64)
reveal_type(carrier_recovery(iq))
reveal_type(carrier_recovery(iq, diagnostics=True))
reveal_type(symbol_sync(iq, 4))
reveal_type(symbol_sync(iq, 4, diagnostics=True))

p = Pipeline(ArraySource(iq, sample_rate=1.0))
reveal_type(p.run())
reveal_type(p.run(sink=print))
reveal_type(p.run(profile=True))
reveal_type(p.run(sink=print, profile=True))
"""

# numpy's stubs spell the array type differently across versions
# (`ndarray[Any, Any]`, `ndarray[tuple[Any, ...], dtype[Any]]`), so the
# parameters are stripped before comparing; `builtins.` prefixes vary too and
# are removed.
EXPECTED = [
    "numpy.ndarray",
    "tuple[numpy.ndarray, sdr_dsp.core.sync.LoopDiagnostics]",
    "numpy.ndarray",
    "tuple[numpy.ndarray, sdr_dsp.core.sync.LoopDiagnostics]",
    "list[numpy.ndarray]",
    "None",
    "tuple[list[numpy.ndarray], sdr_dsp.stream.pipeline.PipelineStats]",
    "tuple[None, sdr_dsp.stream.pipeline.PipelineStats]",
]


def _strip_ndarray_params(text: str) -> str:
    out, i = [], 0
    while i < len(text):
        j = text.find("numpy.ndarray[", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j] + "numpy.ndarray")
        depth, k = 0, j + len("numpy.ndarray")
        while k < len(text):
            depth += text[k] == "["
            depth -= text[k] == "]"
            k += 1
            if depth == 0:
                break
        i = k
    return "".join(out).replace("builtins.", "")


def test_overloads_narrow_return_types(tmp_path):
    snippet = tmp_path / "snippet.py"
    snippet.write_text(textwrap.dedent(SNIPPET))
    proc = subprocess.run(
        [sys.executable, "-m", "mypy", "--no-error-summary", "--no-incremental",
         "--cache-dir", str(tmp_path / "cache"), str(snippet)],
        capture_output=True, text=True, cwd=tmp_path,
        env={**os.environ, "MYPYPATH": str(SRC), "PYTHONPATH": str(SRC)},
        check=False,
    )
    notes = [ln.split("Revealed type is ", 1)[1].strip('"')
             for ln in proc.stdout.splitlines() if "Revealed type is" in ln]
    assert len(notes) == len(EXPECTED), proc.stdout + proc.stderr
    for got, want in zip(notes, EXPECTED, strict=True):
        assert _strip_ndarray_params(got) == want, f"expected {want!r}, mypy says {got!r}"
