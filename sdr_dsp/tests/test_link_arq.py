"""Tests for the ARQ link protocol (TX Phase D).

Covers the pure state machine (clean exchange, retransmit on loss, duplicate
handling, retry-limit give-up), both window modes, record/replay exactness, and
the protocol running over the real modulate->channel->demod DSP chain.
"""

import functools
import os
import random
import tempfile

import numpy as np
import pytest

from sdr_dsp.link import (ARQ, run_link, run_sim, replay, EventLog,
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


# -- sequence-number width guard (the silent-aliasing landmine) -------------

def test_seq_width_guard_rejects_oversize_window():
    # the header packs seq in one byte; a window forcing seq_mod > 256 must raise
    with pytest.raises(ValueError):
        ARQ(window_size=200)               # seq_mod would be 400


def test_seq_width_allows_max_safe_window():
    a = ARQ(window_size=128)               # seq_mod exactly 256
    assert a.seq_mod == 256


def test_explicit_seq_mod_over_limit_rejected():
    with pytest.raises(ValueError):
        ARQ(window_size=4, seq_mod=512)


# -- windowed hardening: burst loss and reordering --------------------------

class _BurstLoss:
    def __init__(self, lose_at, count):
        self.n = 0
        self.lose_at = lose_at
        self.count = count

    def __call__(self, payload):
        self.n += 1
        if self.lose_at <= self.n < self.lose_at + self.count:
            return False, None
        return True, payload


@pytest.mark.parametrize("n,lose_at,count", [
    (4, 3, 3), (4, 1, 2), (8, 5, 4), (2, 2, 1),
    (4, 2, 5), (8, 1, 7), (3, 4, 2), (16, 8, 10),
])
def test_windowed_survives_burst_loss(n, lose_at, count):
    msgs = [bytes([i % 256]) for i in range(20)]
    received, _ = run_link(msgs, window_size=n,
                           transport=_BurstLoss(lose_at, count),
                           timeout_ticks=3, max_retries=60, max_ticks=8000)
    assert received == msgs                 # complete and in order


def test_windowed_random_loss_stress():
    import random
    for trial in range(20):
        rng = random.Random(trial)

        class _RandomLoss:
            def __call__(self, payload):
                return (False, None) if rng.random() < 0.3 else (True, payload)

        n = rng.choice([1, 2, 4, 8])
        nmsg = rng.randint(5, 25)
        msgs = [bytes([i % 256, (i * 7) % 256]) for i in range(nmsg)]
        received, _ = run_link(msgs, window_size=n, transport=_RandomLoss(),
                               timeout_ticks=2, max_retries=200,
                               max_ticks=20000)
        assert received == msgs, f"trial {trial}: N={n}"


def test_windowed_send_base_advances():
    # after a clean windowed run the base catches up to next_seq (window empty)
    msgs = [bytes([i]) for i in range(8)]
    A = ARQ(window_size=4)
    B = ARQ(window_size=4)
    for m in msgs:
        A.send(m)
    run_sim(A, B, max_ticks=200)
    assert A._send_base == A._next_seq      # base fully caught up
    assert not A._outstanding               # nothing left in flight


# -- cumulative ACK (opt-in) ------------------------------------------------

class _RandomLoss:
    def __init__(self, seed, p):
        self.rng = random.Random(seed)
        self.p = p

    def __call__(self, payload):
        return (False, None) if self.rng.random() < self.p else (True, payload)


@pytest.mark.parametrize("window", [2, 4, 8])
def test_cumulative_ack_correct_under_random_loss(window):
    for trial in range(15):
        loss = _RandomLoss(trial + 1000, 0.3)
        nmsg = 5 + (trial % 15)
        msgs = [bytes([(i >> 8) & 0xFF, i & 0xFF]) for i in range(nmsg)]
        A = ARQ(window_size=window, cumulative_ack=True,
                timeout_ticks=2, max_retries=999)
        B = ARQ(window_size=window, cumulative_ack=True,
                timeout_ticks=2, max_retries=999)
        for m in msgs:
            A.send(m)
        _, rec, _ = run_sim(A, B, transport=loss, max_ticks=200000)
        assert rec == msgs, f"window={window} trial={trial}"


def test_cumulative_ack_survives_heavy_ack_loss():
    # the scenario that originally broke cumulative ACK: ACKs dropped heavily
    from sdr_dsp.link.protocol import unpack_payload, TYPE_ACK
    rng = random.Random(7)

    def transport(payload):
        ftype, _, _ = unpack_payload(payload)
        thresh = 0.4 if ftype == TYPE_ACK else 0.15
        return (False, None) if rng.random() < thresh else (True, payload)

    msgs = [bytes([i]) for i in range(20)]
    A = ARQ(window_size=4, cumulative_ack=True, timeout_ticks=2, max_retries=999)
    B = ARQ(window_size=4, cumulative_ack=True, timeout_ticks=2, max_retries=999)
    for m in msgs:
        A.send(m)
    _, rec, _ = run_sim(A, B, transport=transport, max_ticks=200000)
    assert rec == msgs


def test_cumulative_ack_never_acks_out_of_order_seq():
    # the specific bug that was fixed: a cumulative ACK must name the contiguous
    # high-water mark, never an out-of-order frame's seq (which would tell the
    # sender a gap frame arrived when it hadn't -> seq aliasing).
    from sdr_dsp.link.protocol import (pack_payload, unpack_payload,
                                       TYPE_DATA, TYPE_ACK)
    B = ARQ(window_size=4, cumulative_ack=True)

    # out-of-order seq 2 with nothing contiguous yet -> NO ack
    B.on_event(("rx", pack_payload(TYPE_DATA, 2, b"XX"), True))
    acks = [i for i in B.poll() if i[0] == "tx"]
    assert acks == []

    # seq 0 arrives -> deliver 0, cumulative-ACK 0
    B.on_event(("rx", pack_payload(TYPE_DATA, 0, b"AA"), True))
    seqs = [unpack_payload(i[1])[1] for i in B.poll()
            if i[0] == "tx" and unpack_payload(i[1])[0] == TYPE_ACK]
    assert seqs == [0]

    # seq 1 arrives -> drains 1 and buffered 2, cumulative-ACK 2
    B.on_event(("rx", pack_payload(TYPE_DATA, 1, b"BB"), True))
    seqs = [unpack_payload(i[1])[1] for i in B.poll()
            if i[0] == "tx" and unpack_payload(i[1])[0] == TYPE_ACK]
    assert seqs == [2]


def test_cumulative_and_selective_both_deliver_clean():
    # parity: both ACK modes deliver a clean exchange identically
    for cumulative in (False, True):
        msgs = [bytes([i]) for i in range(10)]
        A = ARQ(window_size=4, cumulative_ack=cumulative)
        B = ARQ(window_size=4, cumulative_ack=cumulative)
        for m in msgs:
            A.send(m)
        _, received, _ = run_sim(A, B, max_ticks=500)
        assert received == msgs
