# Design Note — EventLog → pcap/Wireshark Converter

A design note, not an implementation. Captures how the ARQ `EventLog` JSON would
be converted to pcap for inspection in Wireshark, what (small) logging additions
would help, and why this stays a standalone tool that never touches the protocol.

> Status: design only. Build later. The point of writing it now is so the log
> format isn't a liability — we know what the converter needs and that the
> current format is close.

---

## 1. Goal

Turn a recorded protocol exchange (`EventLog`, JSON) into a `.pcap`/`.pcapng`
file that Wireshark can open, with a dissector that decodes the ARQ header
(type, seq, crc_ok) so the exchange is browsable like any captured protocol —
filterable, colorable, timeline-able. This is the "don't be insular, ride
existing tools" path: the log already exists for replay; this makes it legible to
the standard network-analysis toolchain.

---

## 2. What the log has vs. what pcap wants

Each `EventLog` record currently has: `tick`, `station`, `dir`, `type`, `seq`,
`crc_ok`, `payload_hex`, `note`. That is ~80% of what a pcap needs.

| pcap/Wireshark wants | EventLog has? | Gap / plan |
|---|---|---|
| per-packet timestamp | `tick` (logical) | synthesize from tick for sim/replay; add a real timestamp for live runs (see §3) |
| src → dst | `station` + `dir` | derive in the converter (A↔B from station+dir) |
| frame bytes to dissect | `payload_hex` (type+seq+data) | sufficient — dissect the protocol header + data; the PHY frame (preamble/sync/CRC) is intentionally not logged |
| link-layer type | — | converter assigns a custom `DLT_USER` link type with a small synthetic header |
| frame length | derivable from `payload_hex` | optional explicit field (see §3) |

The deliberate non-goal: the event log does NOT carry raw IQ or the full PHY
frame. IQ belongs in SigMF captures; the event log is the protocol layer. The
dissector decodes type/seq/crc/data — exactly the protocol view you'd want in
Wireshark — not the waveform.

---

## 3. Small logging additions worth making first

Additive only; they don't change replay (which uses tick + rx records):

1. **Optional real timestamp** — for *live* runs (Phase E), record wall-clock
   time alongside `tick`. For sim/replay there's no real time, so the converter
   synthesizes monotonic timestamps from `tick` (e.g. tick × a nominal interval).
   Field: `time` (epoch seconds, float), present on live records, absent on sim.
2. **Optional frame length** — `len` (bytes of the protocol payload). Trivially
   derivable from `payload_hex`, but explicit is friendlier for the converter
   and for pandas. Low priority.

Neither is required to build the converter; the timestamp is the only one that
materially improves the Wireshark view (a real timeline for live captures).

---

## 4. Converter shape (when built)

A standalone tool — `examples/eventlog_to_pcap.py` or a small `tools/` script —
that does NOT import the protocol engine (it only reads JSON):

```
load EventLog JSON
  -> for each record:
       build a synthetic link-layer header (src/dst from station+dir,
         type, seq, crc_ok flag)
       + the protocol payload bytes (from payload_hex)
       -> write a pcap packet with timestamp (real or tick-derived)
  -> emit a .pcap with a DLT_USER link type
```

Plus a **Lua dissector** (`sdr_dsp_arq.lua`) for Wireshark that decodes the
synthetic header: a `type` field (DATA/ACK/NAK), a `seq` field, a `crc_ok`
boolean, and the data payload. Drop it in Wireshark's plugin dir; it makes the
custom link type human-readable and filterable (`sdr_dsp.type == ACK`,
`sdr_dsp.seq == 5`, etc.).

Two output options to weigh at build time:
- **pcap via `text2pcap`** — emit a hex+offset text format and shell out to
  `text2pcap`. Zero Python pcap deps; relies on an external tool.
- **pcapng directly** — write the binary with a tiny pcap writer (no heavy dep
  needed; the format is simple). Self-contained, slightly more code.

Either is fine; `text2pcap` is the faster path to a first result.

---

## 5. Why this is safe to defer

- The `EventLog` format is already structured for it (flat, named fields) — the
  decision that made this cheap was made when the log was designed.
- The converter touches nothing in the library: it reads JSON and writes pcap.
  No protocol risk, no test surface in the core.
- The only thing that would be annoying to add *later* is the real timestamp for
  live runs — but that's a one-field addition to the live driver's logging when
  Phase E happens, naturally alongside enabling real TX.

So: no liability in the current format. Build the converter when there's a real
exchange worth inspecting in Wireshark (likely once Phase E produces live logs).
