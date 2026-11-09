"""Framing: turn a payload into a findable, verifiable packet, and back.

A modulator turns bits into a waveform, but a receiver needs to find where a
packet starts in a stream of recovered bits and know whether it arrived intact.
A frame provides that structure:

    [preamble][sync word][length][payload][CRC]

  - preamble  : an alternating 1010... run so the receiver can detect the
                packet and settle its timing recovery before the real data.
  - sync word : a fixed, known bit pattern marking exactly where the payload
                begins. The receiver correlates against it to find the frame.
  - length    : how many payload bytes follow (so the receiver knows where the
                payload ends).
  - payload   : the actual data bytes.
  - CRC       : a checksum over length+payload. The receiver recomputes it; a
                match means the payload is intact. This is the foundation ACKs
                are built on -- you ACK a frame whose CRC checks out.

This module works on BITS (numpy arrays of 0/1), the layer between your message
and the modulator: build_frame(payload) -> bits -> modulate -> ... -> demod ->
bits -> find_frames -> payloads. It is pure logic + a correlation search; no DSP
beyond what core.detect already provides, and no hardware.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from ._validate import byteslike, positive_int

# default sync word: a 16-bit pattern with good autocorrelation (low side-lobes)
# so the correlator finds it cleanly. This is a Barker-like / well-known marker.
DEFAULT_SYNC = np.array(
    [1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1], dtype=np.uint8)
DEFAULT_PREAMBLE_BITS = 32


def _bytes_to_bits(data: Any) -> np.ndarray:
    """Bytes -> bit array, MSB first."""
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    return np.unpackbits(arr)


def _bits_to_bytes(bits: Any) -> bytes:
    """Bit array (MSB first, length multiple of 8) -> bytes."""
    bits = np.asarray(bits, dtype=np.uint8)
    n = (len(bits) // 8) * 8
    return np.packbits(bits[:n]).tobytes()


def crc16(data: Any) -> int:
    """CRC-16/CCITT-FALSE over a bytes-like input. OUR code.

    A standard 16-bit CRC (poly 0x1021, init 0xFFFF). Used to detect whether a
    received payload is intact. Not cryptographic -- just error detection.
    """
    crc = 0xFFFF
    for byte in byteslike("data", data):
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def build_frame(
    payload: Any, sync: Any | None = None, preamble_bits: int = DEFAULT_PREAMBLE_BITS
) -> np.ndarray:
    """Build a complete frame from a payload. Returns a bit array (0/1). OUR code.

    Layout: [preamble][sync][length:1 byte][payload][crc16:2 bytes].
    The length byte limits a single frame to 255 payload bytes; split larger
    messages across frames yourself.

    payload:        bytes-like (bytes, bytearray, or a list of ints 0..255).
    sync:           the sync word bits (default DEFAULT_SYNC).
    preamble_bits:  number of alternating preamble bits.

    The CRC covers the length byte and the payload, so a receiver validates both.
    """
    payload = byteslike("payload", payload)
    if len(payload) > 255:
        raise ValueError(f"payload exceeds 255 bytes ({len(payload)}); split across frames")
    preamble_bits = positive_int("preamble_bits", preamble_bits, minimum=0)
    sync = DEFAULT_SYNC if sync is None else np.asarray(sync, dtype=np.uint8)

    preamble = np.tile([1, 0], preamble_bits // 2 + 1)[:preamble_bits].astype(
        np.uint8)
    header_and_payload = bytes([len(payload)]) + payload
    crc = crc16(header_and_payload)
    crc_bytes = bytes([(crc >> 8) & 0xFF, crc & 0xFF])

    body_bits = _bytes_to_bits(header_and_payload + crc_bytes)
    return np.concatenate([preamble, sync, body_bits]).astype(np.uint8)


def find_frames(
    bits: Any, sync: Any | None = None, max_sync_errors: int = 2
) -> list[dict[str, Any]]:
    """Find and validate frames in a recovered bit stream. OUR code.

    Searches for the sync word (allowing up to max_sync_errors bit mismatches,
    since recovered bits may have errors), then reads the length byte, payload,
    and CRC after each match. Returns a list of dicts, one per frame found:

        {"payload": bytes, "crc_ok": bool, "bit_offset": int}

    crc_ok tells you whether the payload survived intact -- the basis for an ACK.
    Frames with a bad length read or running off the end of the buffer are
    skipped. After a frame whose CRC VERIFIES, the scan steps past its whole
    body so a sync-like pattern inside the payload is not reported again.
    After a match whose CRC FAILS, the scan advances one bit: the match may
    have been a near-miss on random bits with a plausible length byte, and the
    real frame may begin inside the body it claimed. (Stepping past every
    claimed body, as an earlier version did, lost 2% of real frames at the
    default tolerance when preceded by noise.) A consequence: one corrupted
    real frame can be reported more than once at nearby offsets, all with
    crc_ok False -- consumers act on crc_ok, so this costs a duplicate NAK
    at most.
    """
    if isinstance(bits, tuple):
        # a DemodResult (or a (bits, soft) pair) is iterable, so this used to
        # be accepted and silently find nothing
        raise TypeError("find_frames wants a 1-D bit array, got a tuple -- "
                        "pass demod_result.bits (the symbol-domain demods return "
                        "DemodResult(bits, soft))")
    bits = np.asarray(bits)
    if bits.ndim != 1:
        raise ValueError(f"find_frames wants a 1-D bit array, got shape {bits.shape}")
    if bits.size and not np.all((bits == 0) | (bits == 1)):
        raise ValueError("find_frames wants bits (0/1); got other values -- slice or "
                         "threshold first")
    bits = bits.astype(np.uint8)
    sync = DEFAULT_SYNC if sync is None else np.asarray(sync, dtype=np.uint8)
    slen = len(sync)
    n = len(bits)
    frames: list[dict[str, Any]] = []

    i = 0
    while i <= n - slen:
        # match the sync word at position i (Hamming distance within tolerance)
        window = bits[i:i + slen]
        if np.count_nonzero(window != sync) <= max_sync_errors:
            start = i + slen
            # need at least the length byte
            if start + 8 > n:
                break
            length = int(_bits_to_bytes(bits[start:start + 8])[0])
            body_bytes = 1 + length + 2          # length + payload + crc16
            body_bits = body_bytes * 8
            if start + body_bits > n:
                # not enough bits left for the claimed frame; skip this match
                i += 1
                continue
            chunk = _bits_to_bytes(bits[start:start + body_bits])
            payload = chunk[1:1 + length]
            rx_crc = (chunk[1 + length] << 8) | chunk[1 + length + 1]
            crc_ok = (rx_crc == crc16(chunk[:1 + length]))
            frames.append({
                "payload": payload,
                "crc_ok": crc_ok,
                "bit_offset": i,
            })
            # a verified frame owns its bits: step past them. A failed one
            # may be a false sync in front of the real frame: step one bit.
            i = start + body_bits if crc_ok else i + 1
        else:
            i += 1
    return frames
