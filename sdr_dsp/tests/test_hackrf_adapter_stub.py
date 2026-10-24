"""HackRFCapture (the reference IQSource adapter) against a stub hackrf_transfer.

The adapter is what every live example streams through, and until now nothing
exercised it without a board. hackrfpy resolves its tool binaries from
``tools_dir``, so a tiny executable that writes a known int8 ramp to stdout
stands in for the radio: the adapter's context manager, blocks(), read() and
the hackrfpy stream path underneath all run for real, with no hardware.

The ramp has I even and Q odd, and the stub writes it in deliberately
odd-sized flushed pieces, so a single dropped byte anywhere shows up as every
later sample having I and Q exchanged -- a capture whose power statistics look
fine while its phase is garbage. hackrfpy < 1.1.1 does exactly that in
``PersistentReceiver.blocks()``; the xfail flips to a hard failure (strict)
when the fixed hackrfpy is installed, which is the signal to delete the mark.
"""

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

hackrfpy = pytest.importorskip("hackrfpy", reason="dev group installs hackrfpy")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
N = 200_000
RAW = bytes(i % 256 for i in range(2 * N))
EXPECT = np.arange(2 * N) % 256

_STUB = '''#!{python}
import signal, sys, time
signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
raw = bytes(i % 256 for i in range({nbytes}))
sizes = [7, 65537, 3, 131071, 1]
pos = k = 0
while pos < len(raw):
    n = sizes[k % len(sizes)]; k += 1
    sys.stdout.buffer.write(raw[pos:pos + n]); pos += n
    sys.stdout.buffer.flush()
time.sleep(10)        # stay alive until the parent interrupts us
'''


@pytest.fixture
def stub_tools_dir(tmp_path):
    body = _STUB.format(python=sys.executable, nbytes=len(RAW))
    if os.name == "nt":
        py = tmp_path / "hackrf_transfer.py"
        py.write_text(body)
        (tmp_path / "hackrf_transfer.bat").write_text(
            f'@echo off\r\n"{sys.executable}" "{py}" %*\r\n')
    else:
        exe = tmp_path / "hackrf_transfer"
        exe.write_text(body)
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(tmp_path)


@pytest.fixture
def HackRFCapture():
    sys.path.insert(0, str(EXAMPLES))
    try:
        from hackrf_capture import HackRFCapture
    finally:
        sys.path.pop(0)
    return HackRFCapture


def _check_ramp(x):
    i = np.rint(x.real * 128).astype(int) % 256
    q = np.rint(x.imag * 128).astype(int) % 256
    assert np.array_equal(i, EXPECT[0::2][:len(x)]), "I channel corrupted"
    assert np.array_equal(q, EXPECT[1::2][:len(x)]), "Q channel corrupted"


def _version_tuple(v):
    try:
        return tuple(int(p) for p in v.split("+")[0].split(".")[:3])
    except ValueError:
        return (0, 0, 0)


needs_fixed_hackrfpy = pytest.mark.xfail(
    _version_tuple(hackrfpy.__version__) < (1, 1, 1), strict=True,
    reason="hackrfpy < 1.1.1: PersistentReceiver.blocks() drops an odd byte "
           "and swaps I/Q for the rest of the stream")


def test_adapter_satisfies_iqsource(stub_tools_dir, HackRFCapture):
    from sdr_dsp.sources.base import IQSource
    src = HackRFCapture(100e6, 2e6, tools_dir=stub_tools_dir)
    assert isinstance(src, IQSource)
    assert src.sample_rate == 2e6 and src.center_freq == 100e6


def test_adapter_requires_context_manager(stub_tools_dir, HackRFCapture):
    src = HackRFCapture(100e6, 2e6, tools_dir=stub_tools_dir)
    with pytest.raises(RuntimeError):
        next(src.blocks())
    with pytest.raises(RuntimeError):
        src.read(10)


def test_adapter_read_is_exact_and_pair_aligned(stub_tools_dir, HackRFCapture):
    with HackRFCapture(100e6, 2e6, tools_dir=stub_tools_dir) as src:
        a = src.read(1000)
        b = src.read(70_000)
    assert len(a) == 1000 and len(b) == 70_000
    assert a.dtype == np.complex64
    _check_ramp(np.concatenate([a, b]))


@needs_fixed_hackrfpy
def test_adapter_blocks_are_pair_aligned_across_odd_pipe_chunks(stub_tools_dir,
                                                                 HackRFCapture):
    got = []
    with HackRFCapture(100e6, 2e6, tools_dir=stub_tools_dir) as src:
        for blk in src.blocks():
            assert blk.dtype == np.complex64
            got.append(blk)
            if sum(len(g) for g in got) >= N // 2:
                break
    _check_ramp(np.concatenate(got)[:N // 2])


def test_adapter_capture_array_convenience(stub_tools_dir, HackRFCapture):
    src = HackRFCapture(100e6, 2e6, tools_dir=stub_tools_dir)
    x = src.capture_array(5000)
    assert len(x) == 5000
    _check_ramp(x)


def test_adapter_drives_a_streaming_pipeline(stub_tools_dir, HackRFCapture):
    """The point of the seam: the same Pipeline that runs on a file runs on
    the adapter. Stage output must be what the core gives on the same samples."""
    from sdr_dsp import Pipeline
    from sdr_dsp.core import design_lowpass, fir_apply
    from sdr_dsp.stream import StreamingFIR
    taps = design_lowpass(200e3, 2e6, num_taps=33)
    seen = []
    with HackRFCapture(100e6, 2e6, tools_dir=stub_tools_dir, block_size=4096) as src:
        class _Bounded:
            sample_rate, center_freq = src.sample_rate, src.center_freq
            def blocks(self):
                for b in src.blocks():
                    seen.append(b)
                    yield b
                    if sum(len(s) for s in seen) >= 40_000:
                        return
        out = Pipeline(_Bounded()).add(StreamingFIR(taps), "lp").run()
    raw = np.concatenate(seen)
    np.testing.assert_allclose(np.concatenate(out), fir_apply(raw, taps), atol=1e-5)
