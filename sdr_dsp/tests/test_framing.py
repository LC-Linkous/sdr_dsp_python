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
from sdr_dsp.core.framing import DEFAULT_SYNC, DEFAULT_PREAMBLE_BITS


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


@pytest.mark.parametrize("tolerance", [0, 1, 2, 3])
def test_sync_tolerance_is_inclusive_at_the_boundary(tolerance):
    # max_sync_errors is an inclusive bound: EXACTLY that many errors must
    # still match, one more must not. (The test above uses 1 error against a
    # tolerance of 2, so an off-by-one in the comparison -- `<` for `<=` --
    # passed it.) Errors are placed inside the sync word, which build_frame
    # puts at the head of the frame after the preamble.
    frame = build_frame(b"OK")
    sync_start = DEFAULT_PREAMBLE_BITS
    assert np.array_equal(frame[sync_start:sync_start + len(DEFAULT_SYNC)],
                          DEFAULT_SYNC)

    at_limit = frame.copy()
    for k in range(tolerance):
        at_limit[sync_start + 2 * k] ^= 1
    found = find_frames(at_limit, max_sync_errors=tolerance)
    assert len(found) == 1 and found[0]["payload"] == b"OK", \
        f"{tolerance} errors rejected at tolerance {tolerance}"

    over = at_limit.copy()
    over[sync_start + 2 * tolerance] ^= 1       # one more error
    assert find_frames(over, max_sync_errors=tolerance) == [], \
        f"{tolerance + 1} errors accepted at tolerance {tolerance}"


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


def test_near_miss_sync_in_noise_does_not_swallow_the_real_frame():
    # 200 random bits before a valid frame: at max_sync_errors=2 a random
    # window matches the sync ~1 time in 100, and if its length byte happens
    # to fit the buffer the old scanner jumped past the claimed body -- over
    # the real frame -- 2% of the time. The real frame must be found in every
    # one of 300 trials, with its CRC intact.
    frame = build_frame(b"hello")
    missed = []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        bits = np.concatenate([rng.integers(0, 2, 200, dtype=np.uint8), frame])
        good = [f for f in find_frames(bits) if f["crc_ok"]]
        want = 200 + DEFAULT_PREAMBLE_BITS                # bit_offset is the SYNC position
        if not (good and good[-1]["payload"] == b"hello" and good[-1]["bit_offset"] == want):
            missed.append(seed)
    assert not missed, f"real frame missed after noise for seeds {missed}"


def test_corrupted_frame_still_reported_and_following_frame_found():
    # a frame with a flipped length-byte bit is reported crc_ok=False, and the
    # good frame right behind it is still found (the scan did not jump past it)
    bad = build_frame(b"first").copy()
    bad[32 + 16 + 3] ^= 1                            # a bit of the length byte
    bits = np.concatenate([bad, build_frame(b"second")])
    found = find_frames(bits)
    assert any(f["crc_ok"] and f["payload"] == b"second" for f in found)
    assert not any(f["crc_ok"] and f["payload"] == b"first" for f in found)


def test_find_frames_rejects_a_demod_result_and_non_bits():
    from sdr_dsp.core import bpsk_demod, DemodResult
    res = bpsk_demod(np.array([1, -1, 1], dtype=np.complex64))
    assert isinstance(res, DemodResult)
    bits, soft = res                                   # tuple unpacking still works
    assert bits.dtype == np.uint8 and res.bits is bits
    with pytest.raises(TypeError, match=r"\.bits"):
        find_frames(res)                               # used to return [] silently
    with pytest.raises(ValueError, match="1-D"):
        find_frames(np.zeros((2, 40), dtype=np.uint8))
    with pytest.raises(ValueError, match="bits"):
        find_frames(np.array([0, 1, 2, 1]))
    assert find_frames(res.bits) == []                 # the right call, no frame in 3 bits
