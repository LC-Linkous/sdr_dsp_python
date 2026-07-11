LOG CREATION: July 2026
Modernization, standardation and a lot of spell checking have made major changes to 
the structure of the codebase. There's been a lot of improved modularity to eventually
expand to one or two other SDRs on my benchtop, but that means this codebase is now much
harder to read than the original version that was posted to GitHub. The following log has 
been implemented to track major updates and fixes. It's not complete because this is currently
pre-release (there is no Version 1 yet) and heavily under development. There's also features 
that have hooks, but have not been implemented yet. 



## 2026-07-11 — Impairment modeling & feature extraction (fingerprint F0–F2/F4/F5)

New capability landed: the library can synthesize per-device analog-hardware
impairments and extract them back — the forward model as test oracle for the
estimators, same verify-against-truth discipline as the scipy filter tests. This
is the DSP layer only; turning feature vectors into a device *identity*
(classifier, labeled captures, trained model) is application logic that lives in
a separate consuming project, not here. The dividing line held: extractors are
general DSP, the classifier is not.

**Verified:** every estimator recovers its known impairment within tolerance
(imbalance gain < 0.05 dB, phase < 0.3° at the finite-sample floor); image ratio
rotation-invariant to 5 decimals; phase-noise step variance matches 2πΔν/fs to
3 figures. Separability guard (F5) reaches ~68% five-way on the two symbol-free
features alone (chance 20%). Suite: 355 → 379 passed, 1 skipped.

- **New `core/channel_impairments.py`** (forward model / oracle):
  `add_iq_imbalance` (α·s + β·s* conjugate-image), `add_pa_nonlinearity`
  (odd-order AM/AM+AM/PM polynomial), `add_phase_noise` (Wiener random-walk),
  and `DeviceImpairments`/`make_device_impairments`/`apply_device_impairments`
  (a frozen, seed-repeatable bundle = one virtual device). Sits beside
  `channel.py`, not inside it: that models propagation, this models the
  transmitter.
- **New `core/features/` package** (extractors): `impairments.py`
  (`iq_image_ratio` — the rotation-invariant, receiver-robust feature to
  classify on — plus split `estimate_iq_imbalance`, `estimate_cfo_ppm`,
  `estimate_phase_noise_variance`), `evm.py` (`decide_symbols`, `error_vector`,
  `evm_stats` — nine error-cloud moments), `fingerprint.py`
  (`fingerprint_vector` + stable `FEATURE_NAMES`, length 14; append-only).
- **Three physics guards baked into the code**, because the math constrains what
  is estimable and the library refuses to pretend otherwise: (1) properness —
  `estimate_iq_imbalance` warns when `|c|>0.5` (improper OOK/BPSK/DC reads
  modulation as device; clean OOK gives image ratio ~1.0); (2) constant-envelope
  PA blindness — `add_pa_nonlinearity` collapses to one complex constant on
  `|x|=A` signals (GFSK/FSK have no PA fingerprint, exact not approximate);
  (3) SNR trap — `evm_rms` is a distance-to-antenna thermometer, weight the
  shape moments. All three documented at the call site.
- **Flattened into `sdr_dsp.core`**: all synthesis + extraction functions import
  from the top level, house style.
- **New `tests/test_features.py` (24 tests)**: forward-model invariants,
  estimator round-trips against the oracle, rotation-invariance, the properness
  RuntimeWarning, and the F5 separability go/no-go.
- **`examples/impairment_extraction.py`**: synthesize known impairments, recover
  them, and the error-cloud demo (same 16-QAM + noise, four impairments — the
  geometry, not the size, is the signature). `--plot` shows the clouds. Scoped
  to the DSP boundary; no classification.
- **`sdr_dsp_REFERENCE.md` §13** (new; old §13–15 → §14–16) and an EXAMPLES.md
  section document the capability. API handbook regenerated to cover
  `core.features`.
- **License metadata fixed**: `pyproject.toml` declared MIT while both LICENSE
  files are GPL-2.0; toml now reads `license = "GPL-2.0-only"` (SPDX string),
  wheel metadata confirmed.

Caveat: F5 separability is proven only in simulation, against synthesized virtual
devices through `apply_channel`. Whether real same-model devices through a real
receiver satisfy S_B > S_W is an empirical measurement, not a theorem — the cheap
CFO-ppm capture experiment is the go/no-go before any classifier work. F3
(transient features) is scaffolded in the plan but deferred to the hardware era.

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