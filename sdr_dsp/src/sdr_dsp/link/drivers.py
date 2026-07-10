"""Drivers: turn the pure ARQ engine's intentions into actual I/O.

The ARQ state machine only emits intentions and consumes events. A driver
decides what an intention becomes. Three are provided / seamed here:

  - sim driver       : runs two engines over a transport (e.g. the Phase C
                       channel or a perfect link), no hardware.
  - replay driver    : feeds a recorded event log to one engine, zero TX.
  - live driver seam : a transport interface a real-radio implementation fills
                       in at Phase E (modulate + transmit + receive).

Plus an EventLog (structured, JSON-serializable, tool-friendly) and a
convenience run_link() for the common simulated case.
"""

from __future__ import annotations

import json

import numpy as np
from dataclasses import dataclass, field, asdict

from .arq import ARQ
from .protocol import unpack_payload, type_name


# ===========================================================================
# Event log -- structured, JSON, designed to convert toward existing tools
# ===========================================================================
@dataclass
class EventLog:
    """A structured, replayable record of a protocol exchange.

    Each record is a flat dict with named fields (tick, station, dir, type, seq,
    crc_ok, payload_hex, note) -- deliberately tool-friendly, so the JSON can be
    transformed into pcap/Wireshark, dropped into pandas, etc., rather than being
    an insular format. Replay only needs the inbound events; the full log
    (including emitted intentions) is kept for inspection.
    """
    records: list = field(default_factory=list)

    def record(self, tick, station, direction, payload=None, crc_ok=None,
               note=""):
        """Append one structured event record. payload is the protocol payload
        bytes (decoded into type/seq for readability)."""
        rec = {"tick": tick, "station": station, "dir": direction, "note": note}
        if payload is not None:
            try:
                ftype, seq, data = unpack_payload(payload)
                rec["type"] = type_name(ftype)
                rec["seq"] = seq
                rec["payload_hex"] = bytes(data).hex()
            except ValueError:
                rec["type"] = "?"
                rec["payload_hex"] = bytes(payload).hex()
        if crc_ok is not None:
            rec["crc_ok"] = bool(crc_ok)
        self.records.append(rec)

    def save(self, path):
        """Write the log as indented JSON."""
        with open(path, "w") as f:
            json.dump({"records": self.records}, f, indent=2)
        return str(path)

    @classmethod
    def load(cls, path):
        """Load a log saved by save()."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls(records=d.get("records", []))

    def __len__(self):
        return len(self.records)


# ===========================================================================
# Sim driver: two engines over a transport
# ===========================================================================
def _deliver(intents, src_station, dst_engine, tick, log, transport):
    """Route one engine's tx intentions to the other engine as rx events,
    passing each through `transport` (which may drop/corrupt). Non-tx intentions
    are logged/collected. Returns (delivered_payloads, app_outputs)."""
    app_outputs = []
    for intent in intents:
        kind = intent[0]
        if kind == "tx":
            payload = intent[1]
            log.record(tick, src_station, "tx", payload=payload)
            ok, out_payload = transport(payload)
            if out_payload is not None:
                # arrives at the destination as an rx event
                log.record(tick, _other(src_station), "rx",
                           payload=out_payload, crc_ok=ok)
                dst_engine.on_event(("rx", out_payload, ok))
            else:
                log.record(tick, src_station, "drop", payload=payload,
                           note="dropped in transport")
        else:
            app_outputs.append((src_station, intent))
            log.record(tick, src_station, "app", note=str(intent))
    return app_outputs


def _other(station):
    return "B" if station == "A" else "A"


def perfect_transport(payload):
    """A lossless transport: every frame arrives, CRC ok."""
    return True, payload


def make_channel_transport(modulate, demodulate, channel, drop_predicate=None):
    """Build a transport that runs a payload through modulate -> channel ->
    demodulate -> framing, returning (crc_ok, recovered_payload | None).

    modulate:   payload_bytes -> iq      (build_frame + a modulator)
    demodulate: iq -> [frames]           (a demod + find_frames)
    channel:    iq -> iq                 (apply_channel, partial-applied)
    drop_predicate: optional callable() -> bool to force a drop (for demos).

    This is the bridge from the abstract protocol to the real DSP chain. It's how
    the sim driver exercises the protocol over the Phase C channel.
    """
    def transport(payload):
        if drop_predicate and drop_predicate():
            return False, None
        iq = modulate(payload)
        iq = channel(iq)
        frames = demodulate(iq)
        if not frames:
            return False, None
        f = frames[0]
        return f["crc_ok"], (f["payload"] if f["crc_ok"] else None)
    return transport


def run_sim(station_a, station_b, max_ticks=200, transport=None, log=None):
    """Drive two ARQ engines until both are idle or max_ticks is reached.

    station_a, station_b: ARQ engines (already given their send() messages).
    transport: a transport callable (default: perfect_transport).
    log: an EventLog to record into (created if None).

    Returns (delivered_to_a, delivered_to_b, log) where delivered_* are the
    payloads each station's app received (in order).
    """
    transport = transport or perfect_transport
    log = log if log is not None else EventLog()
    delivered = {"A": [], "B": []}

    def collect(app_outputs):
        for station, intent in app_outputs:
            if intent[0] == "deliver":
                # a deliver at station X means X RECEIVED a message
                delivered[station].append(intent[1])

    engines = {"A": station_a, "B": station_b}
    for tick in range(max_ticks):
        # route A's intentions to B, and B's to A
        collect(_deliver(station_a.poll(), "A", station_b, tick, log, transport))
        collect(_deliver(station_b.poll(), "B", station_a, tick, log, transport))
        # advance logical time on both
        if station_a.idle and station_b.idle:
            break
        station_a.on_event(("tick",))
        station_b.on_event(("tick",))
    return delivered["A"], delivered["B"], log


# ===========================================================================
# Replay driver: reproduce an exchange from a recorded log, zero TX
# ===========================================================================
def replay(log, engine, station):
    """Feed a recorded log's inbound events for `station` to a fresh engine and
    return the intentions it produces -- reproducing the original run with no
    transmission.

    Only the rx events (and ticks) drive the engine; tx records in the log were
    the *output* of the original run and are recomputed here. This is the
    zero-TX demo path: load a saved exchange and watch the protocol re-derive it.
    """
    produced = []
    for rec in log.records:
        if rec["dir"] == "rx" and rec["station"] == station:
            payload = _payload_from_record(rec)
            engine.on_event(("rx", payload, rec.get("crc_ok", True)))
            produced.extend(engine.poll())
        elif rec["dir"] == "tick" and rec.get("station") == station:
            engine.on_event(("tick",))
            produced.extend(engine.poll())
    return produced


def _payload_from_record(rec):
    """Reconstruct the protocol payload bytes from a log record."""
    from .protocol import pack_payload, TYPE_DATA, TYPE_ACK, TYPE_NAK
    name_to_type = {"DATA": TYPE_DATA, "ACK": TYPE_ACK, "NAK": TYPE_NAK}
    ftype = name_to_type.get(rec.get("type", "DATA"), TYPE_DATA)
    seq = rec.get("seq", 0)
    data = bytes.fromhex(rec.get("payload_hex", ""))
    return pack_payload(ftype, seq, data)


# ===========================================================================
# Live driver: wire one ARQ engine to a real TXSink + IQSource (Phase E)
# ===========================================================================
class LiveLink:
    """Drive one ARQ engine over real radio: a TXSink to transmit, an IQSource
    to receive. The Phase E seam.

    This is the structural bridge from the pure protocol to hardware. It turns
    the engine's ("tx", payload) intentions into modulate -> sink.transmit(), and
    turns received IQ into ("rx", payload, crc_ok) events fed back to the engine.
    It does NOT implement a radio -- it drives whatever TXSink/IQSource you give
    it, so a LoopbackSink tests the wiring and a real device adapter (Phase E on
    the bench) does the real thing.

    HONESTY: half-duplex turnaround timing, real device latency, and whether the
    tick-based timeouts map onto hardware are exactly the things that can only be
    settled on a bench. This class makes the wiring correct and testable; it does
    not and cannot prove over-the-air behavior in software.

    Args:
        engine:     an ARQ instance.
        sink:       a TXSink (transmit side).
        modulate:   payload_bytes -> iq  (build_frame + a modulator).
        demodulate: iq -> [frames]       (a demod + find_frames), used on RX.
        log:        optional EventLog to record the exchange (for later replay).
        station:    label for the log ("A"/"B").
        carry_samples: how many trailing IQ samples to carry from each
            on_rx_iq() call into the next. A streaming receiver delivers
            blocks at arbitrary boundaries, and a frame split across two
            blocks is invisible to both halves' demod -- silently lost. Set
            this to at least one full frame's length in samples (frame bits x
            samples_per_symbol, plus padding) and the overlap guarantees every
            frame lands whole in some window. Default 0 preserves the old
            per-block behavior, which is only correct when the caller
            delivers burst-aligned segments (e.g. via find_bursts, or the
            whole-buffer LoopbackSink loop). A frame that falls entirely
            inside the overlap can be found twice; that is safe here -- ARQ
            duplicate detection (sequence numbers) exists for exactly this.
    """

    def __init__(self, engine, sink, modulate, demodulate, log=None,
                 station="A", carry_samples=0):
        self.engine = engine
        self.sink = sink
        self.modulate = modulate
        self.demodulate = demodulate
        self.log = log
        self.station = station
        self.carry_samples = int(carry_samples)
        self._carry = np.zeros(0, dtype=np.complex64)
        self._tick = 0

    def pump(self):
        """Flush the engine's pending intentions: transmit any tx, return the
        app-level outputs (deliver/done/failed). Call after feeding events."""
        app_outputs = []
        for intent in self.engine.poll():
            if intent[0] == "tx":
                payload = intent[1]
                if self.log is not None:
                    self.log.record(self._tick, self.station, "tx",
                                    payload=payload)
                iq = self.modulate(payload)
                self.sink.transmit(iq)
            else:
                app_outputs.append(intent)
                if self.log is not None:
                    self.log.record(self._tick, self.station, "app",
                                    note=str(intent))
        return app_outputs

    def on_rx_iq(self, iq):
        """Feed received IQ: demodulate, find frames, deliver each as an rx
        event. Returns the app outputs produced.

        With carry_samples > 0, the tail of the previous call's IQ is
        prepended so frames straddling a block boundary are still found."""
        iq = np.asarray(iq, dtype=np.complex64)
        if self.carry_samples > 0:
            iq = np.concatenate([self._carry, iq])
            self._carry = iq[-self.carry_samples:].copy()
        frames = self.demodulate(iq)
        for f in frames:
            payload = f["payload"]
            crc_ok = f["crc_ok"]
            if self.log is not None:
                self.log.record(self._tick, self.station, "rx",
                                payload=payload if crc_ok else None,
                                crc_ok=crc_ok)
            if crc_ok and len(payload) >= 2:
                self.engine.on_event(("rx", payload, True))
        return self.pump()

    def tick(self):
        """Advance one logical tick (drives timeouts/retransmits)."""
        self._tick += 1
        self.engine.on_event(("tick",))
        return self.pump()

    def send(self, data):
        """Queue an application message and flush."""
        self.engine.send(data)
        return self.pump()


# ===========================================================================
# Convenience wrapper for the common simulated case
# ===========================================================================
def run_link(messages, window_size=1, transport=None, timeout_ticks=10,
             max_retries=5, max_ticks=500):
    """Send a list of messages from station A to station B over a sim link.

    The concrete, readable entry point: hides the event plumbing for the common
    "run it in simulation" case. Returns (received_by_b, log).

        received, log = run_link([b"hello", b"world"])
        log.save("exchange.json")          # then replay later with zero TX

    transport defaults to a perfect link; pass a channel transport (see
    make_channel_transport) to run over the Phase C DSP chain.
    """
    A = ARQ(window_size=window_size, timeout_ticks=timeout_ticks,
            max_retries=max_retries)
    B = ARQ(window_size=window_size, timeout_ticks=timeout_ticks,
            max_retries=max_retries)
    for m in messages:
        A.send(m)
    _, received_b, log = run_sim(A, B, max_ticks=max_ticks, transport=transport)
    return received_b, log
