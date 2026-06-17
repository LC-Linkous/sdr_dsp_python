"""Tests for the transmit-side device seam (TX Phase E).

What CAN be tested in software: the TXSink protocol, the LoopbackSink, the
LiveLink wiring (protocol genuinely drives a sink), and the device-adapter
safety guard. What CANNOT: actual over-the-air transmission, timing, and
regulatory behavior -- those need a bench and are not asserted here.
"""

import numpy as np
import pytest

from sdr_dsp.sinks import TXSink, LoopbackSink
from sdr_dsp.core.link import ARQ, LiveLink, EventLog
from sdr_dsp.core import build_frame, find_frames, fsk_modulate, fsk_demod

FS = 1e6
SPS = 20


def _modulate(payload):
    return fsk_modulate(build_frame(payload), SPS, 50e3, FS)


def _demodulate(iq):
    bits = fsk_demod(iq, FS)[SPS // 2::SPS]
    return find_frames(np.asarray(bits, dtype=np.uint8))


# -- protocol conformance ---------------------------------------------------

def test_loopback_satisfies_txsink():
    assert isinstance(LoopbackSink(FS), TXSink)


def test_loopback_buffers_transmitted_iq():
    sink = LoopbackSink(FS)
    sink.transmit(np.ones(100, dtype=np.complex64))
    sink.transmit(np.ones(50, dtype=np.complex64))
    assert len(sink.buffer) == 150
    assert sink.transmit_count == 2


def test_loopback_clear():
    sink = LoopbackSink(FS)
    sink.transmit(np.ones(10, dtype=np.complex64))
    sink.clear()
    assert len(sink.buffer) == 0 and sink.transmit_count == 0


# -- LiveLink wiring (the seam genuinely drives a sink) --------------------

def test_livelink_transmits_into_sink():
    eng = ARQ(window_size=1)
    sink = LoopbackSink(FS)
    link = LiveLink(eng, sink, _modulate, _demodulate, station="A")
    link.send(b"HI")
    # sending a message should have produced a real transmission into the sink
    assert sink.transmit_count == 1
    assert len(sink.buffer) > 0


def test_livelink_full_exchange_through_loopback():
    # two stations, each a LiveLink over its own LoopbackSink; shuttle the IQ
    a_eng = ARQ(window_size=1, timeout_ticks=3, max_retries=10)
    b_eng = ARQ(window_size=1, timeout_ticks=3, max_retries=10)
    sink_a, sink_b = LoopbackSink(FS), LoopbackSink(FS)
    A = LiveLink(a_eng, sink_a, _modulate, _demodulate, station="A")
    B = LiveLink(b_eng, sink_b, _modulate, _demodulate, station="B")

    A.send(b"HELLO")
    delivered = []
    for _ in range(30):
        if len(sink_a.buffer):
            iq = sink_a.buffer
            sink_a.clear()
            delivered += [o[1] for o in B.on_rx_iq(iq) if o[0] == "deliver"]
        if len(sink_b.buffer):
            iq = sink_b.buffer
            sink_b.clear()
            A.on_rx_iq(iq)
        if a_eng.idle and b_eng.idle:
            break
        A.tick()
        B.tick()
    assert delivered == [b"HELLO"]
    assert a_eng.idle                      # A got its ACK


def test_livelink_records_log():
    eng = ARQ(window_size=1)
    sink = LoopbackSink(FS)
    log = EventLog()
    link = LiveLink(eng, sink, _modulate, _demodulate, log=log, station="A")
    link.send(b"X")
    # the transmission was logged as a structured record
    tx_recs = [r for r in log.records if r["dir"] == "tx"]
    assert len(tx_recs) == 1 and tx_recs[0]["type"] == "DATA"


# -- device-adapter safety guard -------------------------------------------

def test_hackrf_sink_guard_prevents_accidental_tx():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))
    from hackrf_sink import HackRFSink
    sink = HackRFSink(433.92e6, FS)         # not armed
    assert isinstance(sink, TXSink)
    with pytest.raises(RuntimeError):
        sink.transmit(np.ones(10, dtype=np.complex64))   # refuses when unarmed


# -- hardware test (auto-skips without a device) ---------------------------

@pytest.mark.hardware
def test_real_hackrf_transmit():
    # only runs with a connected, armed HackRF on a legal/wired setup.
    pytest.skip("requires connected HackRF + explicit arming; bench-only")
