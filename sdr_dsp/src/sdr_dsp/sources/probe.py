"""Probe captures that go through a FILE, never a stdout pipe. OUR code.

Why this module exists -- the short version of a real debugging session:
hackrfpy's ``capture_array`` streams raw IQ from hackrf_transfer's stdout
through a subprocess pipe into Python. On Windows at 8 Msps that is 16 MB/s
through a pipe with a Python reader in the loop; reader stalls back the pipe
up and samples get dropped inside hackrf_transfer. Dropped samples are phase
discontinuities, and FM lives in phase: the demodulated noise floor rises
~15-20 dB, the 19 kHz stereo pilot drowns, and every quality measurement
lies -- while counts, tuning, and the power spectrum all still look healthy.

Measured on the same board, antenna, station, gain, minutes apart:

    capture_array (pipe):    channel  +7.2 dB   pilot  +1.4 dB
    capture to file:         channel +17.5 dB   pilot +19.4 dB
    reference file (by ear): channel +20.6 dB   pilot +26.5 dB

So every probe that feeds a measurement -- gain search, preflight scoring,
health checks -- must capture the way the verified reference recordings were
made: hackrf_transfer writing straight to disk, loaded afterwards. That is
what ``probe_capture`` does. The corpus recordings in the collection tools
always used the file path; only the probes went through the pipe.

The handle ``h`` just needs hackrfpy's ``capture`` signature; nothing here
imports hackrfpy, so the library core stays device-free.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import sys
import tempfile

import numpy as np

# Warning lines hackrfpy has already printed once this process. A gain
# search makes ~16 probes, and identical "[!] sample_rate=... below the
# recommended..." lines on every one of them bury the actual probe results
# (a real preflight printed it 50+ times). First occurrence passes through;
# repeats are dropped. Only lines starting with "[!]" are ever filtered --
# everything else streams through untouched.
_seen_warnings: set = set()


class _DedupWarnings(io.TextIOBase):
    def __init__(self, real):
        self._real = real
        self._buf = ""

    def write(self, text):
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.startswith("[!]"):
                if line in _seen_warnings:
                    continue
                _seen_warnings.add(line)
            self._real.write(line + "\n")
        return len(text)

    def flush(self):
        if self._buf:
            self._real.write(self._buf)
            self._buf = ""
        self._real.flush()


class _WarnDedupFilter(logging.Filter):
    """Drop repeated WARNING+ records from a logger, once per process.

    Attached to hackrfpy's logger by ``dedup_warnings``. Runs at record-
    dispatch time, before handlers, so it collapses duplicates regardless of
    how the record eventually reaches stderr. Keeps its own memory (separate
    from the stream wrapper's ``_seen_warnings``) so the first, legitimate
    occurrence is never mistaken for a repeat by the other layer.
    """

    def __init__(self) -> None:
        super().__init__()
        self._seen: set = set()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True  # leave INFO/progress chatter untouched
        msg = record.getMessage()
        if msg in self._seen:
            return False
        self._seen.add(msg)
        return True


_HACKRFPY_LOG = logging.getLogger("hackrfpy")
_hackrfpy_warn_dedup = _WarnDedupFilter()


@contextlib.contextmanager
def dedup_warnings():
    """Suppress repeated "[!] ..." warning lines on stdout/stderr.

    Warnings reach the console two different ways, so this collapses both:

    * hackrfpy emits its "[!] ..." notices through
      ``logging.getLogger("hackrfpy").warning(...)``. A logging ``Filter`` on
      that logger drops repeat records *before they reach any handler*, so it
      works whether the record is written by a ``StreamHandler`` bound to the
      real ``sys.stderr`` or by logging's last-resort handler. The stream
      redirect below only ever caught the last-resort case -- once anything
      installs a bound handler (a plain ``logging.basicConfig`` in a tool is
      enough), ``redirect_stderr`` no longer sees the record and the same
      sub-8-Msps line printed once per probe (~16x per station in preflight).
      The filter is scoped to WARNING+ so INFO progress chatter on the same
      logger (which legitimately repeats per probe) is left alone.
    * Anything written straight to stdout/stderr (direct ``print`` of a
      "[!] " line) is line-filtered by the stream wrapper, preserving that
      contract for callers that don't go through logging.
    """
    out, err = _DedupWarnings(sys.stdout), _DedupWarnings(sys.stderr)
    _HACKRFPY_LOG.addFilter(_hackrfpy_warn_dedup)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            yield
        finally:
            out.flush()
            err.flush()
            _HACKRFPY_LOG.removeFilter(_hackrfpy_warn_dedup)


def load_ci8(path, max_samples=None):
    """Load an interleaved ci8 IQ file as complex64 normalized by 128."""
    count = -1 if max_samples is None else 2 * int(max_samples)
    raw = np.fromfile(path, dtype=np.int8, count=count)
    n = raw.size - (raw.size % 2)
    return ((raw[0:n:2].astype(np.float32)
             + 1j * raw[1:n:2].astype(np.float32)) / 128.0)


def probe_capture(h, freq, sample_rate, num_samples, *, lna=16, vga=20,
                  amp=False, tmp_dir=None):
    """Capture num_samples of IQ via the FILE path and return complex64.

    Drop-in replacement for ``h.capture_array(...)`` wherever the samples
    feed a measurement. Slightly slower per probe (a file lands on disk and
    is deleted), and worth every millisecond: the numbers it returns are the
    ones the verified reference captures were judged by.
    """
    num_samples = int(num_samples)
    fd, path = tempfile.mkstemp(suffix=".iq", dir=tmp_dir)
    os.close(fd)
    try:
        with dedup_warnings():
            h.capture(freq, sample_rate, num_samples=num_samples, out=path,
                      lna=lna, vga=vga, amp=amp, sigmf=False)
        return load_ci8(path, max_samples=num_samples)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def make_prober(h, sample_rate, num_samples, freq, *, tmp_dir=None):
    """A ``probe(lna, vga, amp)`` callable for ``search_gain``, bound to one
    frequency, capturing via the file path."""
    def probe(lna, vga, amp):
        return probe_capture(h, freq, sample_rate, num_samples,
                             lna=lna, vga=vga, amp=amp, tmp_dir=tmp_dir)
    return probe
