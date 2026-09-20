"""sdr_dsp.sources.probe: probes must go through a file, and prove it.

Why: the pipe-based capture_array path drops samples on Windows at high
rates, which corrupts every phase-based measurement while leaving power
statistics looking healthy (full story in the module docstring). These
tests pin the file contract: what h.capture wrote to disk is exactly, to
the int8 grid, what the probe hands to the measurement -- no pipe, no
resampling, no silent truncation -- and the temp file never outlives the
probe.
"""

import glob
import os

import numpy as np

from sdr_dsp.sources.probe import load_ci8, make_prober, probe_capture


class FileRadio:
    """Speaks exactly the h.capture file contract; records what was asked."""

    def __init__(self, seed=0):
        self.calls = []
        self._rng = np.random.default_rng(seed)
        self.last_iq = None

    def capture(self, freq, sample_rate, *, num_samples=None, out=None,
                lna=16, vga=20, amp=False, sigmf=False, **k):
        self.calls.append(dict(freq=freq, rate=sample_rate, n=num_samples,
                               out=out, lna=lna, vga=vga, amp=amp,
                               sigmf=sigmf))
        n = int(num_samples)
        iq = (self._rng.standard_normal(n)
              + 1j * self._rng.standard_normal(n)) * 0.2
        i8 = np.empty(2 * n, dtype=np.int8)
        i8[0::2] = np.clip(np.round(iq.real * 128), -128, 127).astype(np.int8)
        i8[1::2] = np.clip(np.round(iq.imag * 128), -128, 127).astype(np.int8)
        i8.tofile(out)
        self.last_iq = (i8[0::2].astype(np.float32)
                        + 1j * i8[1::2].astype(np.float32)) / 128.0


def test_probe_round_trips_the_file_exactly(tmp_path):
    h = FileRadio()
    iq = probe_capture(h, 98.5e6, 2e6, 5000, lna=24, vga=8, amp=True,
                       tmp_dir=tmp_path)
    assert iq.dtype == np.complex64 and len(iq) == 5000
    np.testing.assert_array_equal(iq, h.last_iq.astype(np.complex64))
    call = h.calls[0]
    assert (call["lna"], call["vga"], call["amp"]) == (24, 8, True)
    assert call["sigmf"] is False, "probe files don't need sidecars"


def test_probe_cleans_up_its_temp_file(tmp_path):
    h = FileRadio()
    probe_capture(h, 98.5e6, 2e6, 1000, tmp_dir=tmp_path)
    assert glob.glob(str(tmp_path / "*")) == [], "temp capture left behind"


def test_probe_cleans_up_even_when_capture_fails(tmp_path):
    class BrokenRadio:
        def capture(self, *a, **k):
            raise RuntimeError("device busy")

    try:
        probe_capture(BrokenRadio(), 98.5e6, 2e6, 1000, tmp_dir=tmp_path)
    except RuntimeError:
        pass
    assert glob.glob(str(tmp_path / "*")) == []


def test_make_prober_binds_frequency_and_signature(tmp_path):
    h = FileRadio()
    probe = make_prober(h, 2e6, 4096, 103.7e6, tmp_dir=tmp_path)
    iq = probe(32, 20, False)                    # the search_gain signature
    assert len(iq) == 4096
    assert h.calls[0]["freq"] == 103.7e6
    assert (h.calls[0]["lna"], h.calls[0]["vga"]) == (32, 20)


def test_load_ci8_handles_odd_trailing_byte(tmp_path):
    p = tmp_path / "odd.iq"
    np.array([64, -64, 32, -32, 127], dtype=np.int8).tofile(p)   # 2.5 pairs
    iq = load_ci8(p)
    assert len(iq) == 2, "a split I/Q pair must be dropped, not misaligned"
    assert iq[0] == np.complex64(0.5 - 0.5j)
