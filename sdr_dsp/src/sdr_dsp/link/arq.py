"""The ARQ state machine: reliable, acknowledged message exchange.

A pure, event-driven state machine. It never touches a radio, a file, or a
clock -- it only consumes EVENTS (things that happened) and emits INTENTIONS
(things it wants done). A driver decides what an intention becomes: a real
transmission, a simulated channel hop, or a log line. Because behavior is purely
a function of the events fed in, a recorded event sequence replays exactly --
which is what lets a real exchange be recorded once and replayed for demos with
zero transmission.

One engine covers both ARQ variants via `window_size`:
  - window_size = 1  -> stop-and-wait (one frame in flight)
  - window_size = N  -> sliding window (up to N frames in flight)

EVENTS in (via on_event):
  ("send", data_bytes)          the application wants to send a message
  ("rx", payload_bytes, crc_ok) a frame arrived (payload = the protocol payload)
  ("tick",)                     one unit of logical time passed

INTENTIONS out (from poll()):
  ("tx", payload_bytes)         transmit this protocol payload
  ("deliver", data_bytes)       a message arrived intact; hand it to the app
  ("done", seq)                 a sent message was acknowledged
  ("failed", seq)               a sent message gave up after max retries

Time is LOGICAL TICKS, not seconds. Timeouts are counted in ticks. The engine is
deterministic: same events in, same intentions out.
"""

from __future__ import annotations

from collections import deque

from .protocol import (TYPE_DATA, TYPE_ACK, pack_payload, unpack_payload)


class ARQ:
    """Event-driven ARQ engine. window_size=1 is stop-and-wait; N is windowed.

    Args:
        window_size:  max frames in flight (1 = stop-and-wait).
        timeout_ticks: ticks to wait for an ACK before retransmitting.
        max_retries:  retransmit attempts before giving up on a frame.
        seq_mod:      sequence-number modulus (must be >= 2*window_size so the
                      window never aliases; defaults to 2*window_size).
    """

    def __init__(self, window_size=1, timeout_ticks=10, max_retries=5,
                 seq_mod=None):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = int(window_size)
        self.timeout_ticks = int(timeout_ticks)
        self.max_retries = int(max_retries)
        self.seq_mod = int(seq_mod) if seq_mod else 2 * self.window_size
        if self.seq_mod < 2 * self.window_size:
            raise ValueError("seq_mod must be >= 2*window_size")

        # --- sender state ---
        self._send_queue = deque()        # app messages waiting to go out
        self._next_seq = 0                # seq to assign to the next new frame
        # outstanding frames: seq -> {"data", "age", "retries"}
        self._outstanding = {}

        # --- receiver state ---
        self._expected_rx = 0             # next seq we expect to deliver in order
        # buffered out-of-order frames (windowed mode): seq -> data
        self._rx_buffer = {}

        # --- output ---
        self._out = deque()               # pending intentions

    # -- application API --------------------------------------------------
    def send(self, data):
        """Queue a message for reliable delivery (an application event)."""
        self.on_event(("send", bytes(data)))

    def poll(self):
        """Return and clear the pending intentions emitted so far."""
        out = list(self._out)
        self._out.clear()
        return out

    @property
    def idle(self):
        """True when nothing is queued or in flight (sender side)."""
        return not self._send_queue and not self._outstanding

    # -- event intake -----------------------------------------------------
    def on_event(self, event):
        """Feed one event to the machine. Updates state, may queue intentions."""
        kind = event[0]
        if kind == "send":
            self._send_queue.append(event[1])
            self._pump_window()
        elif kind == "rx":
            _, payload, crc_ok = event
            self._on_rx(payload, crc_ok)
        elif kind == "tick":
            self._on_tick()
        else:
            raise ValueError(f"unknown event kind: {kind!r}")

    # -- sender -----------------------------------------------------------
    def _pump_window(self):
        """Move queued messages into flight while the window has room."""
        while self._send_queue and len(self._outstanding) < self.window_size:
            data = self._send_queue.popleft()
            seq = self._next_seq
            self._next_seq = (self._next_seq + 1) % self.seq_mod
            self._outstanding[seq] = {"data": data, "age": 0, "retries": 0}
            self._emit_tx(TYPE_DATA, seq, data)

    def _emit_tx(self, frame_type, seq, data=b""):
        self._out.append(("tx", pack_payload(frame_type, seq, data)))

    def _on_tick(self):
        """Advance logical time; retransmit or give up on timed-out frames."""
        for seq in list(self._outstanding.keys()):
            info = self._outstanding[seq]
            info["age"] += 1
            if info["age"] >= self.timeout_ticks:
                if info["retries"] >= self.max_retries:
                    # give up on this frame
                    del self._outstanding[seq]
                    self._out.append(("failed", seq))
                    # a gap frees window room; let queued messages flow
                    self._pump_window()
                else:
                    info["retries"] += 1
                    info["age"] = 0
                    self._emit_tx(TYPE_DATA, seq, info["data"])

    # -- receiver + ack handling ------------------------------------------
    def _on_rx(self, payload, crc_ok):
        """Handle an arrived frame. Bad CRC -> drop silently (sender will retry)."""
        if not crc_ok:
            return
        try:
            ftype, seq, data = unpack_payload(payload)
        except ValueError:
            return

        if ftype == TYPE_ACK:
            # an ACK acknowledges the frame with this seq
            if seq in self._outstanding:
                del self._outstanding[seq]
                self._out.append(("done", seq))
                self._pump_window()
            return

        if ftype == TYPE_DATA:
            # always ACK a well-formed DATA frame (even a duplicate, since the
            # sender retransmits when our earlier ACK was lost)
            self._emit_tx(TYPE_ACK, seq)
            if self.window_size == 1:
                # stop-and-wait: deliver only if this is the seq we expect.
                # a retransmission carries the previous seq -> re-ACKed above,
                # but not re-delivered.
                if seq == self._expected_rx:
                    self._out.append(("deliver", data))
                    self._expected_rx = (self._expected_rx + 1) % self.seq_mod
            else:
                # windowed: deliver in order, buffering gaps; ignore seqs already
                # passed (duplicates) and those outside the receive window.
                if self._in_rx_window(seq):
                    self._rx_buffer[seq] = data
                    self._drain_rx_in_order()

    def _drain_rx_in_order(self):
        """Deliver buffered frames in sequence order (windowed receiver)."""
        while self._expected_rx in self._rx_buffer:
            data = self._rx_buffer.pop(self._expected_rx)
            self._out.append(("deliver", data))
            self._expected_rx = (self._expected_rx + 1) % self.seq_mod

    def _in_rx_window(self, seq):
        """True if seq is within the current receive window (windowed mode).

        Accepts the next `window_size` sequence numbers starting at the expected
        one, handling modular wraparound. A seq below the expected one is a
        duplicate (already delivered) and rejected.
        """
        for k in range(self.window_size):
            if (self._expected_rx + k) % self.seq_mod == seq:
                return True
        return False
