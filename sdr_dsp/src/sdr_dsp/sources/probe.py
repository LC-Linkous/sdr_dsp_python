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

import os
import tempfile

import numpy as np


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
