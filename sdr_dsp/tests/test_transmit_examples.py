"""Offline tests for examples/transmit_examples.py (TX Phase C).

The point of these is the round trip demod(modulate(x)) == x driven through the
TXSink seam, in software, for every scheme the example ships -- the last
no-hardware step before the two-SDR bench test. They exercise the example's own
modulate/demod chains and its key_out() driver, not just the core primitives.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from sdr_dsp.sinks import LoopbackSink, TXSink

# examples aren't a package; add the dir so we can import the example module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
import transmit_examples as tx        # noqa: E402


ALL_SCHEMES = list(tx.SCHEMES)


# --------------------------------------------------------------------------
# the round-trip proof: every scheme recovers the exact payload with a good CRC
# --------------------------------------------------------------------------
@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_scheme_round_trips_through_loopback(scheme):
    payload = b"CQ DE SDR"
    sink = LoopbackSink(tx.FS, center_freq=tx.DEFAULT_CENTER_HZ)
    frames, n = tx.key_out(scheme, payload, sink)
    good = [f for f in frames if f["crc_ok"] and f["payload"] == payload]
    assert len(good) == 1, f"{scheme}: {len(frames)} frames, no clean match"
    assert n > 0


@pytest.mark.parametrize("scheme", ALL_SCHEMES)
@pytest.mark.parametrize("payload", [b"A", b"HELLO WORLD",
                                     bytes(range(32)), b"\x00\xff\x00\xff"])
def test_scheme_round_trips_varied_payloads(scheme, payload):
    sink = LoopbackSink(tx.FS)
    frames, _ = tx.key_out(scheme, payload, sink)
    good = [f for f in frames if f["crc_ok"] and f["payload"] == payload]
    assert len(good) == 1, f"{scheme} lost payload {payload!r}"


# --------------------------------------------------------------------------
# key_out really drives the sink protocol (not a hidden shortcut)
# --------------------------------------------------------------------------
def test_key_out_drives_the_sink():
    sink = LoopbackSink(tx.FS)
    tx.key_out("fsk", b"x", sink)
    assert sink.transmit_count == 1
    assert len(sink.buffer) > 0


def test_key_out_accepts_any_txsink_and_handles_non_buffering():
    """A real device sink transmits instead of buffering, so key_out can't read
    it back -- it must return no frames rather than crash."""
    class DeviceLikeSink:                 # satisfies TXSink, no .buffer
        sample_rate = tx.FS
        center_freq = tx.DEFAULT_CENTER_HZ

        def __init__(self):
            self.sent = 0

        def transmit(self, iq):
            self.sent += 1

    sink = DeviceLikeSink()
    assert isinstance(sink, TXSink)
    frames, n = tx.key_out("ook", b"hi", sink)
    assert sink.sent == 1
    assert frames == [] and n > 0


# --------------------------------------------------------------------------
# the self-check entry point and the guarded hardware seam
# --------------------------------------------------------------------------
def test_run_offline_reports_all_pass():
    assert tx.run_offline(b"73 DE SDR", ALL_SCHEMES) is True


def test_hackrf_sink_seam_is_guarded():
    """The device adapter satisfies TXSink, constructs safely (armed=False), and
    refuses to transmit -- so key_out reaches the seam and stops."""
    path = Path(__file__).resolve().parent.parent / "examples" / "hackrf_sink.py"
    spec = importlib.util.spec_from_file_location("hackrf_sink", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    sink = mod.HackRFSink(tx.DEFAULT_CENTER_HZ, tx.FS)
    assert isinstance(sink, TXSink)
    assert sink.armed is False
    with pytest.raises(RuntimeError):
        tx.key_out("bpsk", b"x", sink)
