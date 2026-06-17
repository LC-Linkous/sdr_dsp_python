"""Tests for the ARQ link protocol (TX Phase D).

Covers the pure state machine (clean exchange, retransmit on loss, duplicate
handling, retry-limit give-up), both window modes, record/replay exactness, and
the protocol running over the real modulate->channel->demod DSP chain.
"""

import functools
import os
import tempfile

import numpy as np
import pytest

from sdr_dsp.core.link import (ARQ, run_link, run_sim, replay, EventLog,
                               make_channel_transport, unpack_payload,
                               TYPE_ACK, TYPE_DATA)
from sdr_dsp.core import (build_frame, find_frames, apply_channel,
                          fsk_modulate, fsk_demod)


# -- pure state machine -----------------------------------------------------

def test_clean_stop_and_wait():
    received, log = run_link([b"HELLO"])
    assert received == [b"HELLO"]


def test_multiple_messages_seq_reuse():
    # more messages than the sequence space (mod 2) -- exercises seq wraparound
    msgs = [b"a", b"b", b"c", b"d", b"e"]
    received, _ = run_link(msgs)
    assert received == msgs


def test_sliding_window_delivers_in_order():
    msgs = [bytes([i]) for i in range(10)]
    received, _ = run_link(msgs, window_size=4)
    assert received == msgs


def test_retransmit_on_lost_data():
    A = ARQ(window_size=1, timeout_ticks=3, max_retries=5)
    A.send(b"X")
    first = [i for i in A.poll() if i[0] == "tx"]
    assert len(first) == 1                       # sent once
    for _ in range(3):
        A.on_event(("tick",))
    retx = [i for i in A.poll() if i[0] == "tx"]
    assert len(retx) == 1                        # retransmitted after timeout


def test_duplicate_not_redelivered():
    A = ARQ(window_size=1, timeout_ticks=3)
    B = ARQ(window_size=1)
    A.send(b"Y")
    data = [i for i in A.poll() if i[0] == "tx"][0][1]
    B.on_event(("rx", data, True))
    first = B.poll()
    assert sum(1 for i in first if i[0] == "deliver") == 1
    # same frame again (a retransmission): re-ACK, do NOT redeliver
    B.on_event(("rx", data, True))
    second = B.poll()
    assert sum(1 for i in second if i[0] == "deliver") == 0
    acks = [i for i in second if i[0] == "tx"
            and unpack_payload(i[1])[0] == TYPE_ACK]
    assert len(acks) == 1


def test_bad_crc_dropped_silently():
    B = ARQ(window_size=1)
    B.on_event(("rx", b"\x00\x00garbage", False))   # crc_ok=False
    assert B.poll() == []                           # nothing delivered or acked


def test_give_up_after_max_retries():
    A = ARQ(window_size=1, timeout_ticks=2, max_retries=3)
    A.send(b"Z")
    A.poll()
    failed = False
    for _ in range(2 * (3 + 2)):                    # enough ticks to exhaust
        A.on_event(("tick",))
        for intent in A.poll():
            if intent[0] == "failed":
                failed = True
    assert failed
    assert A.idle                                   # gave up, nothing in flight


def test_invalid_window_size():
    with pytest.raises(ValueError):
        ARQ(window_size=0)


# -- lossy link -------------------------------------------------------------

class _Lossy:
    def __init__(self, drop_every):
        self.n = 0
        self.k = drop_every

    def __call__(self, payload):
        self.n += 1
        return (False, None) if self.n % self.k == 0 else (True, payload)


def test_delivers_through_loss():
    msgs = [b"one", b"two", b"three", b"four"]
    received, _ = run_link(msgs, transport=_Lossy(3),
                           timeout_ticks=2, max_retries=20)
    assert received == msgs


# -- record / replay --------------------------------------------------------

def test_record_replay_reproduces():
    received, log = run_link([b"AAA", b"BBB"], transport=_Lossy(4),
                             timeout_ticks=2, max_retries=20)
    d = tempfile.mkdtemp()
    path = os.path.join(d, "x.json")
    log.save(path)
    log2 = EventLog.load(path)
    B = ARQ(window_size=1)
    produced = replay(log2, B, "B")
    delivered = [i[1] for i in produced if i[0] == "deliver"]
    assert delivered == received


def test_event_log_is_structured_json():
    _, log = run_link([b"hi"])
    # records have the named, tool-friendly fields
    rec = next(r for r in log.records if r["dir"] == "tx")
    assert "tick" in rec and "type" in rec and "seq" in rec
    assert "payload_hex" in rec


def test_event_log_roundtrip_file():
    _, log = run_link([b"data"])
    d = tempfile.mkdtemp()
    path = os.path.join(d, "log.json")
    log.save(path)
    assert len(EventLog.load(path)) == len(log)


# -- over the real DSP chain ------------------------------------------------

def test_protocol_over_fsk_chain():
    fs = 1e6
    sps = 20

    def modulate(payload):
        return fsk_modulate(build_frame(payload), sps, 50e3, fs)

    def demodulate(iq):
        bits = fsk_demod(iq, fs)[sps // 2::sps]
        return find_frames(np.asarray(bits, dtype=np.uint8))

    channel = functools.partial(apply_channel, sample_rate=fs, snr_db=25, seed=7)
    transport = make_channel_transport(modulate, demodulate, channel)
    msgs = [b"CQ", b"DE", b"SDR"]
    received, _ = run_link(msgs, transport=transport,
                           timeout_ticks=3, max_retries=10)
    assert received == msgs
