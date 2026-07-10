

LOG CREATION: July 2026
Modernization, standardation and a lot of spell checking have made major changes to 
the structure of the codebase. There's been a lot of improved modularity to eventually
expand to one or two other SDRs on my benchtop, but that means this codebase is now much
harder to read than the original version that was posted to GitHub. The following log has 
been implemented to track major updates and fixes. It's not complete because this is currently
pre-release (there is no Version 1 yet) and heavily under development. There's also features 
that have hooks, but have not been implemented yet. 



## 2026-07-10 — Pre-bench hardware-readiness fixes

Code review before first hardware TX found that the digital RX chain was
sim-calibrated: it assumed zero delay, zero carrier offset, and burst-aligned
blocks — the three things a real capture never gives you. All fixes are
opt-in; default outputs are byte-identical to before (pinned by test).

**Measured, full FSK packet chain:** delay sweep 12/20 → 20/20 (30 dB);
17 kHz CFO ~5/10 → 10/10 (20 dB); 8 kHz CFO at 8 dB SNR 0/30 → 30/30.

- **`pad_symbols=` on the digital modulators** (`ook/fsk/bpsk/qpsk_modulate`).
  Unpadded bursts put the first/last symbols at the buffer edges, where
  instantaneous-frequency edge effects + any delay corrupt them. Loopbacks at
  delay 0 hid this. Use ≥ 4 for anything leaving a same-buffer loopback.
- **`fsk_demod(threshold_hz="auto", smooth_samples=N)`.** "auto" thresholds at
  the amplitude²-weighted mean of the instantaneous frequency, self-centering
  under crystal CFO between two radios (±20 ppm ≈ ±17 kHz at 433 MHz).
  `smooth_samples` (~sps/2) is a cheap matched-filter stand-in. Docstring warns
  that `estimate_cfo` is the WRONG corrector for FSK (it finds the ±deviation
  tone, not the offset); `estimate_cfo`'s docstring says the same.
- **New timing primitive `sample_symbols(bits, sps, active=None)`.** Decimates
  a per-sample bit stream at the symbol-center phase estimated from transition
  positions (circular mean), with an optional envelope mask so silence flicker
  doesn't pollute the estimate. Delay-safe where the fixed stride
  `bits[sps//2::sps]` was not; glitch-tolerant where `slice_to_symbols` is not.
- **`LiveLink(carry_samples=N)`.** Frames split across two `on_rx_iq()` blocks
  were silently lost (streaming RX delivers arbitrary boundaries). Carries the
  previous block's tail forward; size ≥ one frame in samples. Overlap can
  re-find a frame — safe, ARQ sequence dedup exists for exactly this (tested
  end-to-end).
- **`find_bursts` auto-threshold floor: median → 1st percentile.** The median
  is only the noise floor when the record is mostly noise; on a
  burst-dominated capture it IS the signal level and one frame fragmented into
  ~5. Docstring now states the regimes and when to set `threshold` explicitly.
- **`remove_dc` caveat**: whole-record mean is biased on burst-dominated
  records — same trap. See `docs/DC_SPIKE.md` (new: DC spike origin, offset
  tuning recipe, per-device table).
- **Flattened the packet workflow API**: `build_frame`, `find_frames`, `crc16`,
  `apply_channel` (+ noise/cfo/delay), all modulators, pulse shaping, and
  `sample_symbols` now import from top-level `sdr_dsp`.
- **`LoopbackSink`**: internal chunk list, `.buffer` is a cached-concatenation
  property (was O(n²) growth per transmit).
- **`examples/two_station_link.py`** now uses the robust chain (padded TX;
  auto-threshold + smoothing + `sample_symbols` RX) — the template to copy for
  bench work.
- **New `tests/test_hardware_readiness.py` (44 tests)**, including the
  previously missing full-chain delay sweep — the closed-loop oracle had only
  ever run at delay 0, which is why 311 green tests coexisted with the delay
  bug. Suite: 355 passed.

Caveat: numbers are against `apply_channel` (AWGN + constant CFO + integer
delay). Fractional delay, drift, and real gain staging are what the wired
one-way bench test exists to prove.
