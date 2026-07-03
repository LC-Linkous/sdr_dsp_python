"""Protocol header that rides inside a Phase B frame payload.

A frame's payload carries a tiny protocol header so each frame announces what it
is and which sequence number it belongs to:

    [type:1 byte][seq:1 byte][ app-data... ]

  type : DATA or ACK (room left for NAK etc.)
  seq  : sequence number (mod 256 here; the engine uses it mod 2 for
         stop-and-wait or mod 2*window for sliding window)
  app-data : the message bytes (empty for an ACK)

The Phase B CRC already covers the whole payload, so this header is
CRC-protected for free. These helpers just pack/unpack the header; building the
findable frame (preamble/sync/length/CRC) is still build_frame from framing.py.
"""

from __future__ import annotations

# frame types
TYPE_DATA = 0x00
TYPE_ACK = 0x01
TYPE_NAK = 0x02

# the seq field in the header is ONE byte, so sequence numbers are mod 256.
# This is the governing limit on seq_mod (and hence window size) -- the ARQ
# engine validates against it so a too-large window can't silently alias.
SEQ_MOD_MAX = 256

_TYPE_NAMES = {TYPE_DATA: "DATA", TYPE_ACK: "ACK", TYPE_NAK: "NAK"}


def type_name(t):
    """Human name for a frame type byte."""
    return _TYPE_NAMES.get(t, f"0x{t:02x}")


def pack_payload(frame_type, seq, data=b""):
    """Build a protocol payload: [type][seq][data]. Returns bytes.

    This goes inside build_frame() as the payload. seq is taken mod 256.
    """
    return bytes([frame_type & 0xFF, seq & 0xFF]) + bytes(data)


def unpack_payload(payload):
    """Parse a protocol payload. Returns (frame_type, seq, data).

    Raises ValueError if too short to hold the 2-byte header.
    """
    payload = bytes(payload)
    if len(payload) < 2:
        raise ValueError("payload too short for a protocol header")
    return payload[0], payload[1], payload[2:]
