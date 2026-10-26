"""Edge-case coverage for sdr_dsp.io and sdr_dsp.link.

SigMF path resolution (meta given, data given, neither on disk), datatype
rejection, the real-valued datatype branch, extra_global metadata; and the
protocol / ARQ / driver guards: short payloads, bad engine arguments, unknown
event kinds, undecodable payloads in the log, and LiveLink's tick + logging
paths.
"""

import json

import numpy as np
import pytest

from sdr_dsp.io.sigmf import (save_iq, load_iq, read_meta, _meta_path,
                              _data_path)
from sdr_dsp.link.protocol import unpack_payload, pack_payload, TYPE_DATA
from sdr_dsp.link import ARQ, EventLog, LiveLink, replay
from sdr_dsp.sinks import LoopbackSink

FS = 1_000_000.0


# -- sigmf: path resolution --------------------------------------------------

def test_meta_path_resolution(tmp_path):
    stem = tmp_path / "rec"
    assert _meta_path(stem.with_suffix(".sigmf-meta")) == stem.with_suffix(".sigmf-meta")
    assert _meta_path(stem.with_suffix(".iq")) == stem.with_suffix(".sigmf-meta")
    assert _meta_path(stem.with_suffix(".sigmf-data")) == stem.with_suffix(".sigmf-meta")
    # any other suffix: same stem, meta extension
    assert _meta_path(stem.with_suffix(".bin")) == stem.with_suffix(".sigmf-meta")


def test_data_path_from_meta_prefers_existing_sibling(tmp_path):
    stem = tmp_path / "rec"
    meta = stem.with_suffix(".sigmf-meta")
    # nothing on disk: the canonical .sigmf-data name is returned anyway
    assert _data_path(meta) == stem.with_suffix(".sigmf-data")
    # an .iq sibling exists and .sigmf-data does not: .iq wins
    stem.with_suffix(".iq").write_bytes(b"")
    assert _data_path(meta) == stem.with_suffix(".iq")


def test_data_path_from_missing_data_name_falls_back(tmp_path):
    stem = tmp_path / "rec"
    # asked for rec.iq, only rec.sigmf-data exists -> use it (save_iq writes
    # .sigmf-data even when handed an .iq name)
    stem.with_suffix(".sigmf-data").write_bytes(b"")
    assert _data_path(stem.with_suffix(".iq")) == stem.with_suffix(".sigmf-data")
    # neither exists: hand back what was asked for and let the caller fail
    missing = tmp_path / "nothing.iq"
    assert _data_path(missing) == missing


# -- sigmf: datatypes and metadata ------------------------------------------

def test_load_iq_rejects_unknown_datatype(tmp_path):
    stem = tmp_path / "odd"
    stem.with_suffix(".sigmf-data").write_bytes(b"\x00" * 16)
    stem.with_suffix(".sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "cu4", "core:sample_rate": FS},
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }))
    with pytest.raises(ValueError, match="unsupported SigMF datatype"):
        load_iq(stem.with_suffix(".sigmf-meta"))


def test_load_iq_real_datatype_promotes_to_complex(tmp_path):
    stem = tmp_path / "real"
    samples = np.array([1.0, -0.5, 0.25, 0.0], dtype=np.float32)
    samples.tofile(stem.with_suffix(".sigmf-data"))
    stem.with_suffix(".sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "rf32_le", "core:sample_rate": FS},
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }))
    iq, meta = load_iq(stem.with_suffix(".sigmf-meta"))
    assert iq.dtype == np.complex64 and len(iq) == 4
    assert np.allclose(iq.real, samples) and np.all(iq.imag == 0)


def test_save_iq_extra_global_is_merged(tmp_path):
    iq = np.zeros(8, dtype=np.complex64)
    save_iq(tmp_path / "x.iq", iq, FS, extra_global={"core:author": "test",
                                                     "core:description": "d"})
    meta = read_meta(tmp_path / "x.sigmf-meta")
    assert meta["global"]["core:author"] == "test"
    assert meta["global"]["core:datatype"] == "cf32_le"      # defaults kept


# -- protocol ----------------------------------------------------------------

def test_unpack_payload_rejects_short_input():
    with pytest.raises(ValueError, match="too short"):
        unpack_payload(b"\x01")
    with pytest.raises(ValueError, match="too short"):
        unpack_payload(b"")


# -- ARQ ---------------------------------------------------------------------

def test_arq_rejects_aliasing_seq_mod():
    with pytest.raises(ValueError, match="seq_mod must be >= 2"):
        ARQ(window_size=4, seq_mod=6)


def test_arq_rejects_unknown_event_kind():
    eng = ARQ()
    with pytest.raises(ValueError, match="unknown event kind"):
        eng.on_event(("bogus",))


def test_arq_ignores_undecodable_rx_payload():
    eng = ARQ()
    eng.on_event(("rx", b"\x07", True))          # one byte: no header
    assert eng.poll() == []                      # nothing emitted, no raise


def test_arq_duplicate_detection_window_edge():
    # A windowed receiver sees a seq that is neither expected nor one of the
    # last `window_size` delivered: the duplicate check must say "no".
    eng = ARQ(window_size=2, seq_mod=8)
    assert eng._is_rx_duplicate(5) is False


# -- EventLog ----------------------------------------------------------------

def test_event_log_records_undecodable_payload_as_unknown():
    log = EventLog()
    log.record(0, "A", "rx", payload=b"\x01")    # too short to unpack
    rec = log.records[-1]
    assert rec["type"] == "?" and rec["payload_hex"] == "01"
    # a well-formed one decodes
    log.record(1, "A", "rx", payload=pack_payload(TYPE_DATA, 3, b"hi"))
    assert log.records[-1]["seq"] == 3 and log.records[-1]["payload_hex"] == "6869"


def test_replay_feeds_ticks_to_the_engine():
    log = EventLog()
    log.record(0, "A", "tick")
    log.record(1, "B", "tick")                   # other station: ignored
    eng = ARQ(timeout_ticks=1)
    eng.send(b"x")
    eng.poll()                                   # consume the initial tx
    produced = replay(log, eng, "A")
    # one tick with timeout_ticks=1 triggers a retransmit intention
    assert any(p[0] == "tx" for p in produced)


# -- LiveLink ----------------------------------------------------------------

def _modulate(payload):
    # "modulation" that just encodes bytes as +-1 symbols; enough for the seam
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    return (2.0 * bits - 1.0).astype(np.complex64)


def _demodulate(iq):
    bits = (np.real(iq) > 0).astype(np.uint8)
    if bits.size % 8:
        return []
    return [{"payload": np.packbits(bits).tobytes(), "crc_ok": True}]


def test_livelink_tick_advances_and_logs(tmp_path):
    log = EventLog()
    eng = ARQ(window_size=1, timeout_ticks=1, max_retries=1)
    sink = LoopbackSink(FS)
    link = LiveLink(eng, sink, _modulate, _demodulate, log=log, station="A")
    link.send(b"Q")
    assert sink.transmit_count == 1
    out1 = link.tick()                           # timeout -> retransmit
    assert sink.transmit_count == 2
    out2 = link.tick()                           # retries exhausted -> failed
    outputs = out1 + out2
    assert any(o[0] == "failed" for o in outputs)
    # the app-level outcome was recorded in the log with a note
    app = [r for r in log.records if r["dir"] == "app"]
    assert app and "failed" in app[-1]["note"]


def test_livelink_logs_received_frames():
    log = EventLog()
    eng = ARQ(window_size=1)
    sink = LoopbackSink(FS)
    link = LiveLink(eng, sink, _modulate, _demodulate, log=log, station="B")
    payload = pack_payload(TYPE_DATA, 0, b"hello")
    link.on_rx_iq(_modulate(payload))
    rx = [r for r in log.records if r["dir"] == "rx"]
    assert rx and rx[-1]["type"] == "DATA" and rx[-1]["crc_ok"] is True


def test_iq_info_rejects_unknown_datatype(tmp_path):
    from sdr_dsp.io.sigmf import iq_info
    stem = tmp_path / "odd2"
    stem.with_suffix(".sigmf-data").write_bytes(b"\x00" * 16)
    stem.with_suffix(".sigmf-meta").write_text(json.dumps({
        "global": {"core:datatype": "cu4", "core:sample_rate": FS},
        "captures": [{"core:sample_start": 0}],
        "annotations": [],
    }))
    with pytest.raises(ValueError, match="unsupported SigMF datatype"):
        iq_info(stem.with_suffix(".sigmf-meta"))


def test_channel_transport_reports_loss_when_no_frame_is_found():
    from sdr_dsp.link.drivers import make_channel_transport
    transport = make_channel_transport(modulate=_modulate,
                                       demodulate=lambda iq: [],
                                       channel=lambda iq: iq)
    assert transport(pack_payload(TYPE_DATA, 0, b"x")) == (False, None)
