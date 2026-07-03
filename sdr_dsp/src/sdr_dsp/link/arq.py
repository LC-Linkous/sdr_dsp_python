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

from .protocol import (TYPE_DATA, TYPE_ACK, pack_payload, unpack_payload,
                       SEQ_MOD_MAX)


class ARQ:
    """Event-driven ARQ engine. window_size=1 is stop-and-wait; N is windowed.

    Args:
        window_size:  max frames in flight (1 = stop-and-wait).
        timeout_ticks: ticks to wait for an ACK before retransmitting.
        max_retries:  retransmit attempts before giving up on a frame.
        seq_mod:      sequence-number modulus (must be >= 2*window_size so the
                      window never aliases; defaults to 2*window_size).
        cumulative_ack: if True, the receiver acknowledges the highest
                      contiguously-received seq (one ACK confirms everything up
                      to it) instead of each frame individually. Correct under
                      loss; the traffic saving only appears with batched arrival.
                      Default False (Selective Repeat, per-frame ACK).
    """

    def __init__(self, window_size=1, timeout_ticks=10, max_retries=5,
                 seq_mod=None, cumulative_ack=False):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = int(window_size)
        self.timeout_ticks = int(timeout_ticks)
        self.max_retries = int(max_retries)
        self.cumulative_ack = bool(cumulative_ack)
        self.seq_mod = int(seq_mod) if seq_mod else 2 * self.window_size
        if self.seq_mod < 2 * self.window_size:
            raise ValueError("seq_mod must be >= 2*window_size")
        if self.seq_mod > SEQ_MOD_MAX:
            # the protocol header packs seq in one byte; a seq_mod above 256
            # would alias silently on the wire. Reject it explicitly rather than
            # corrupt. (For a stop-and-wait or modest window this never trips;
            # it guards the large-window case.)
            raise ValueError(
                f"seq_mod={self.seq_mod} exceeds the header limit "
                f"({SEQ_MOD_MAX}); window_size at most {SEQ_MOD_MAX // 2} with "
                f"the default seq_mod. The 1-byte seq field cannot represent "
                f"more than {SEQ_MOD_MAX} sequence numbers.")

        # --- sender state ---
        self._send_queue = deque()        # app messages waiting to go out
        self._next_seq = 0                # seq to assign to the next new frame
        self._send_base = 0               # oldest unacked seq (window left edge)
        # outstanding frames: seq -> {"data", "age", "retries", "acked"}
        self._outstanding = {}

        # --- receiver state ---
        self._expected_rx = 0             # next seq we expect to deliver in order
        # buffered out-of-order frames (windowed mode): seq -> data
        self._rx_buffer = {}
        # whether any frame has been delivered in order yet -- so a cumulative ACK
        # of (expected_rx - 1) is meaningful even after expected_rx wraps to 0.
        self._rx_delivered_any = False

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
        """Move queued messages into flight while the window has room.

        The window is a contiguous range starting at the oldest UNACKNOWLEDGED
        frame (the send base). A new frame may go out only if its sequence number
        is within window_size of that base -- otherwise the sender's window would
        slide past frames the receiver hasn't accepted yet, desynchronizing the
        two windows (which silently drops frames under burst loss). So the gate
        is the distance from the base, not merely the count of outstanding
        frames.
        """
        while self._send_queue and self._window_has_room():
            data = self._send_queue.popleft()
            seq = self._next_seq
            self._next_seq = (self._next_seq + 1) % self.seq_mod
            self._outstanding[seq] = {"data": data, "age": 0, "retries": 0}
            self._emit_tx(TYPE_DATA, seq, data)

    def _window_has_room(self):
        """True if another frame fits without sliding past the send base.

        The window spans [send_base, send_base + window_size). The next seq to
        assign must stay inside it. Distance is modular.
        """
        distance = (self._next_seq - self._send_base) % self.seq_mod
        return distance < self.window_size

    def _advance_send_base(self):
        """Slide the base forward past contiguous acknowledged frames.

        A frame is removed from _outstanding when acked. The base advances over
        any seq no longer outstanding (i.e. acked) until it reaches one still in
        flight -- the Selective-Repeat window slide.
        """
        while (self._send_base != self._next_seq
               and self._send_base not in self._outstanding):
            self._send_base = (self._send_base + 1) % self.seq_mod

    def _emit_tx(self, frame_type, seq, data=b""):
        self._out.append(("tx", pack_payload(frame_type, seq, data)))

    def _emit_data_ack(self, seq):
        """Emit the ACK for a received DATA frame.

        Selective (default): ACK exactly this seq -- it stands alone.

        Cumulative: ACK the highest CONTIGUOUSLY-received seq, i.e.
        (expected_rx - 1). A cumulative ACK for N means "I have everything
        through N in order", so it must NEVER name an out-of-order seq -- doing
        so would tell the sender a gap frame arrived when it didn't, letting the
        sender clear and eventually reuse that sequence number while the receiver
        still owes it (the aliasing bug). If nothing contiguous has been received
        yet (expected_rx has not advanced from the receive base), there is
        nothing to cumulatively acknowledge, so we emit no ACK and let the
        sender's timeout drive a retransmit of the gap frame.
        """
        if not self.cumulative_ack:
            self._emit_tx(TYPE_ACK, seq)            # selective: this frame
            return
        # cumulative: only ever ACK the contiguous high-water mark
        if self._rx_delivered_any:
            self._emit_tx(TYPE_ACK, (self._expected_rx - 1) % self.seq_mod)
        # else: nothing contiguous yet -> no cumulative ACK to give (sender retries)

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
                    # a gap frees window room; slide the base and let queued
                    # messages flow
                    self._advance_send_base()
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
            if self.cumulative_ack:
                # a cumulative ACK for seq N acknowledges every outstanding frame
                # from the base through N -- BUT only if N is actually within the
                # live window ahead of the base. A stale/duplicate ACK for a seq
                # at or below the base (already acknowledged) must be IGNORED, not
                # walked around the modulus (which would clear the whole window).
                dist = (seq - self._send_base) % self.seq_mod
                if self._outstanding and dist < self.window_size:
                    acked_any = False
                    for k in range(dist + 1):       # base .. seq inclusive
                        s = (self._send_base + k) % self.seq_mod
                        if s in self._outstanding:
                            del self._outstanding[s]
                            self._out.append(("done", s))
                            acked_any = True
                    if acked_any:
                        self._advance_send_base()
                        self._pump_window()
                # else: stale/out-of-window cumulative ACK -> ignore
            else:
                # selective ACK: acknowledges exactly this frame
                if seq in self._outstanding:
                    del self._outstanding[seq]
                    self._out.append(("done", seq))
                    self._advance_send_base()
                    self._pump_window()
            return

        if ftype == TYPE_DATA:
            if self.window_size == 1:
                # stop-and-wait: ACK any well-formed frame (a retransmission of
                # the previous seq must be re-ACKed), but deliver only the
                # expected one.
                self._emit_tx(TYPE_ACK, seq)
                if seq == self._expected_rx:
                    self._out.append(("deliver", data))
                    self._expected_rx = (self._expected_rx + 1) % self.seq_mod
            else:
                # windowed (Selective Repeat): only ACK frames we can actually
                # account for -- those within the receive window (accepted now)
                # or already-delivered duplicates (below the window). A frame
                # ABOVE the window is NOT ACKed, so the sender won't mark it done
                # while the receiver can't yet hold it -- which is what kept the
                # two windows from desynchronizing under burst loss.
                if self._in_rx_window(seq):
                    self._rx_buffer[seq] = data
                    self._drain_rx_in_order()
                    self._emit_data_ack(seq)
                elif self._is_rx_duplicate(seq):
                    self._emit_data_ack(seq)   # re-ACK so a lost-ACK sender moves on
                # else: above the window -> drop silently, do not ACK

    def _drain_rx_in_order(self):
        """Deliver buffered frames in sequence order (windowed receiver)."""
        while self._expected_rx in self._rx_buffer:
            data = self._rx_buffer.pop(self._expected_rx)
            self._out.append(("deliver", data))
            self._expected_rx = (self._expected_rx + 1) % self.seq_mod
            self._rx_delivered_any = True

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

    def _is_rx_duplicate(self, seq):
        """True if seq was already delivered (lies in the window_size range just
        below expected_rx). Such a frame is re-ACKed but not re-delivered, so a
        sender whose earlier ACK was lost can make progress."""
        for k in range(1, self.window_size + 1):
            if (self._expected_rx - k) % self.seq_mod == seq:
                return True
        return False
