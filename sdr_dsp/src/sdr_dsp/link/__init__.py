"""Link-layer ARQ protocol: reliable acknowledged messaging (TX Phase D).

The ARQ engine is a pure event-driven state machine (stop-and-wait at
window_size=1, sliding window for N>1). Drivers turn its intentions into I/O:
sim (over a transport), replay (from a recorded log, zero TX), and the live seam
for real radio (Phase E). run_link() is the convenience entry point.
"""

from .arq import ARQ
from .protocol import (TYPE_DATA, TYPE_ACK, TYPE_NAK, type_name,
                       pack_payload, unpack_payload)
from .drivers import (EventLog, run_sim, run_link, replay, LiveLink,
                      perfect_transport, make_channel_transport)

__all__ = [
    "ARQ",
    "TYPE_DATA", "TYPE_ACK", "TYPE_NAK", "type_name",
    "pack_payload", "unpack_payload",
    "EventLog", "run_sim", "run_link", "replay", "LiveLink",
    "perfect_transport", "make_channel_transport",
]
