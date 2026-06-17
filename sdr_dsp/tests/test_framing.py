"""Tests for packet framing (TX Phase B).

Covers the CRC, the build->find round-trip, frame detection in a noisy/junk
stream, corruption detection, and the full chain through modulation/demod
(frame -> modulate -> demod -> find_frames), which is what the ACK protocol
will rest on.
"""

import numpy as np
import pytest

from sdr_dsp.core import (build_frame, find_frames, crc16,
                          ook_modulate, ook_envelope, ook_slice,
                          fsk_modulate, fsk_demod,
                          bpsk_modulate, bpsk_demod)


# -- CRC --------------------------------------------------------------------

def test_crc16_deterministic():
    assert crc16(b"hello") == crc16(b"hello")


def test_crc16_changes_with_data():
    assert crc16(b"hello") != crc16(b"hellp")


def test_crc16_known_vector():
    # CRC-16/CCITT-FALSE of "123456789" is 0x29B1
    assert crc16(b"123456789") == 0x29B1


# -- build / find round-trip -----------------------------------------------

def test_frame_roundtrip():
    frame = build_frame(b"HELLO")
    found = find_frames(frame)
    assert len(found) == 1
    assert found[0]["payload"] == b"HELLO"
    assert found[0]["crc_ok"]


def test_empty_payload():
    frame = build_frame(b"")
    found = find_frames(frame)
    assert len(found) == 1
    assert found[0]["payload"] == b""
    assert found[0]["crc_ok"]


def test_max_payload_ok():
    payload = bytes(range(256))[:255]
    found = find_frames(build_frame(payload))
    assert found[0]["payload"] == payload and found[0]["crc_ok"]


def test_oversize_payload_raises():
    with pytest.raises(ValueError):
        build_frame(bytes(256))


def test_two_frames_with_junk():
    rng = np.random.default_rng(0)
    junk = rng.integers(0, 2, 50).astype(np.uint8)
    stream = np.concatenate([junk, build_frame(b"first"), junk,
                             build_frame(b"second"), junk])
    found = find_frames(stream)
    assert [f["payload"] for f in found] == [b"first", b"second"]
    assert all(f["crc_ok"] for f in found)


def test_no_frame_in_random_bits():
    rng = np.random.default_rng(1)
    # random bits should rarely produce a valid CRC; allow detection but not crc_ok
    bits = rng.integers(0, 2, 2000).astype(np.uint8)
    found = find_frames(bits)
    assert all(not f["crc_ok"] for f in found) or len(found) == 0


# -- corruption detection ---------------------------------------------------

def test_crc_catches_payload_corruption():
    frame = build_frame(b"DATA").copy()
    # payload starts after preamble(32)+sync(16)+length(8)=56; flip a payload bit
    frame[60] ^= 1
    found = find_frames(frame)
    assert len(found) == 1
    assert not found[0]["crc_ok"]


def test_sync_tolerates_small_errors():
    # a couple of bit errors in the sync word should still be found
    frame = build_frame(b"OK").copy()
    frame[33] ^= 1                      # one bit in the sync region
    found = find_frames(frame, max_sync_errors=2)
    assert len(found) == 1 and found[0]["payload"] == b"OK"


# -- full chain through modulation -----------------------------------------

@pytest.mark.parametrize("scheme", ["ook", "fsk", "bpsk"])
def test_full_chain_through_modulation(scheme):
    payload = b"ACK 42"
    frame_bits = build_frame(payload)
    sps = 20
    if scheme == "ook":
        iq = ook_modulate(frame_bits, sps)
        rec = ook_slice(ook_envelope(iq))[::sps][:len(frame_bits)]
    elif scheme == "fsk":
        iq = fsk_modulate(frame_bits, sps, 50e3, 1e6)
        rec = fsk_demod(iq, 1e6)[sps // 2::sps][:len(frame_bits)]
    else:  # bpsk
        iq = bpsk_modulate(frame_bits, 1)
        rec, _ = bpsk_demod(iq)
        rec = rec[:len(frame_bits)]
    found = find_frames(np.asarray(rec, dtype=np.uint8))
    assert len(found) == 1
    assert found[0]["payload"] == payload
    assert found[0]["crc_ok"]


def test_bit_offset_reported():
    # bit_offset is where the SYNC WORD was found (after pad + preamble),
    # which is where the receiver locks -- not the start of the preamble.
    from sdr_dsp.core.framing import DEFAULT_PREAMBLE_BITS
    rng = np.random.default_rng(2)
    pad = rng.integers(0, 2, 40).astype(np.uint8)
    stream = np.concatenate([pad, build_frame(b"X")])
    found = find_frames(stream)
    assert found[0]["bit_offset"] == 40 + DEFAULT_PREAMBLE_BITS
