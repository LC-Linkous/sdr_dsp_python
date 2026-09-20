LOG CREATION: July 2026
Modernization, standardation and a lot of spell checking have made major changes to 
the structure of the codebase. There's been a lot of improved modularity to eventually
expand to one or two other SDRs on my benchtop, but that means this codebase is now much
harder to read than the original version that was posted to GitHub. The following log has 
been implemented to track major updates and fixes. It's not complete because this is currently
pre-release (there is no Version 1 yet) and heavily under development. There's also features 
that have hooks, but have not been implemented yet. 



## 2026-09-23 — CI pipeline, typing gate, RNG determinism (+ the real flake)

Made the pipeline trustworthy: a CI workflow, a green mypy gate, deterministic
tests, and -- while wiring the lint gate -- the actual root cause of the
long-standing ~1-in-6 flake. Suite: 744 passed, ruff + mypy clean.

- **The flake, finally caught.** `tests/test_promote_sample_data.py` seeded its
  synthetic fixtures from `abs(hash(str(path)))`. Python's builtin string hash
  is per-process randomized (no PYTHONHASHSEED), and the path is a per-run
  pytest tmp dir, so the noise was reseeded every run. The "weak" fixture sat
  right at the health floor (amp=1.5 -> counts=4, the min), so an unlucky draw
  pushed it over and flipped its verdict. Two fixes: seed from the capture's
  DEFINING inputs via hashlib (process-independent, byte-identical fixtures),
  and drop the weak amp to 0.7 so it lands clearly BELOW the floor (counts~3)
  as a "genuinely bad capture" should. Verified deterministic across 20 runs
  with varied PYTHONHASHSEED. This is almost certainly the flake first seen
  2026-09-21; the earlier conftest numpy-seed helped but couldn't fix a
  hash()-level nondeterminism upstream of numpy.
- **CI workflow** (`.github/workflows/tests.yml`, copied from hackrfpy's shape):
  pytest across {windows, ubuntu, macos} x {3.11, 3.12, 3.13} with hardware
  tests deselected and coverage, plus a ruff + mypy lint job. `working-directory:
  sdr_dsp` (git root is one level up); `PYTHONHASHSEED: 0` set workflow-wide as
  a determinism guard.
- **mypy gate is green.** Fixed 5 real type errors (sigmf dict value type;
  widened `add_pa_nonlinearity` to accept the tuple its caller passes) and added
  a scipy `ignore_missing_imports` override (scipy ships no stubs). Not the
  py.typed-strict gate yet -- that + full annotations stay a tracked follow-up --
  but mypy's default checks now block regressions. `Success: no issues found in
  51 source files`.
- **ruff gate is green.** Autofixed 7 (incl. an F811 duplicate-import in
  core/__init__), wrapped the over-long export lines in the package __init__s,
  renamed a `l` loop var to `tap` in channelize, and added targeted
  per-file-ignores for intentional patterns (E402 in script preambles, E741 `l`
  in DSP index loops -- matching hackrfpy's own config, E702/E731 in teaching
  examples). Also removed ~10 pre-existing dead imports/vars across the test
  suite that the gate would otherwise have failed on.
- **RNG migration:** the 3 legacy global-`np.random` tests (test_util,
  test_sinks, test_io_and_sources) now use explicit `default_rng(seed)`, so
  determinism is local and visible rather than relying only on the conftest
  safety net.

## 2026-09-22 — FM stereo decoder (pilot-locked L/R)

New core capability: `fm_stereo_decode(composite, sample_rate)` recovers left
and right from a demodulated FM stereo multiplex. Suite: 734 → 744 passed.

- **`core/demod/analog.py`: `fm_stereo_decode`** + a small `_analytic_phase`
  helper (analytic signal via numpy FFT, no scipy -- house style). Algorithm:
  bandpass the 19 kHz pilot, double its analytic phase to synthesize a
  unit-amplitude 38 kHz reference phase-locked to the L-R subcarrier,
  coherently detect the subcarrier (2·<mpx·cos2wt>) for (L-R), lowpass the
  composite for (L+R), then matrix L=(sum+dif)/2, R=(sum-dif)/2. Scale matches
  fm_demod's (unnormalized); de-emphasis stays a separate opt-in stage.
- **Centered (zero-delay) filtering is load-bearing here.** First attempt used
  causal `fir_apply`; the pilot bandpass's ~312-sample group delay shifted the
  38 kHz reference relative to the composite, which scrambled and INVERTED the
  L-R recovery (L-only test showed R louder than L). `fir_apply_centered` on
  all three paths keeps pilot/sum/difference on one time base. Lesson worth
  keeping: coherent detection needs the reference and the signal aligned.
- **Validated to the dB on synthetic ground truth** (71-79 dB channel
  separation on distinct L/R tones; mono reconstructs L==R at input amplitude;
  L-only stays >15 dB out of R; stereo sum == mono, so mono-compatible) and on
  the **real 103.7 MHz capture** (genuine L-R content recovered, side/mono
  ~0.4 pre-de-emphasis -- a mono station would be ~0).
- **`write_wav` extended to stereo**: accepts (N,2) with ONE shared
  normalization so the L/R balance isn't collapsed; mono path byte-identical
  (regression-tested). `fm_receiver.py --stereo` decodes and writes a 2-channel
  WAV, falling back to mono with a warning when no pilot is present.
- 10 new tests (6 stereo decode, 4 write_wav stereo).

## 2026-09-21 (later) — Real-data demod tests & the flaky-test fix

A critical pass on test realism, prompted by the observation that only 1 of 41
test files actually demodulates the real capture -- everything else is
synthetic self-consistency. Suite: 729 → 734 passed, and now deterministic.

- **`tests/test_real_demod_e2e.py` (5 tests)** runs the real `fm_receiver`
  chain on `sample_data/fm_2Msps.iq` (the ear-verified 103.7 MHz capture) and
  asserts on properties of genuine broadcast audio that synthesis can't vouch
  for: it demodulates to real program audio (not silence, no clipping, sane
  level), the spectrum has the broadcast low-frequency tilt (~12 dB, vs ~0 for
  noise -- with an explicit noise-contrast test so the tilt can't pass on
  hiss), the 19 kHz pilot stays >40 dB below program in the mono output, and
  the level is stable across the first vs second half of the capture.
  Tolerances are generous (measured with ~20 dB headroom) -- the point is
  "recovers real audio", not bit-exactness on a music broadcast.

- **Flaky-test fix (root cause + structural cure).** A rare (~1 in 6)
  full-suite failure that cleared on rerun. Three legacy tests (test_util,
  test_sinks, test_io_and_sources) draw from the global `np.random` without
  seeding; in isolation they pass, but in a full run the global state is
  whatever prior tests left it, so a tolerance-tight assertion could land on
  an unlucky draw -- an order-dependent flake. Fixed with an autouse conftest
  fixture that seeds `np.random` before every test, making runs reproducible
  regardless of order or selection. Verified: three consecutive full runs at
  734/734 where ~1-in-6 would previously fail. This cures the whole *class*;
  migrating the three tests to explicit `default_rng(seed)` generators is
  tracked in TODO as the local-intent follow-up.

## 2026-09-21 — N-level FSK smoothing & FM decimate-before-demod

Small, self-contained DSP/example patch (0011). Suite: 725 → 729 passed.

- **`fsk_demod_nlevel(smooth_samples=)`** added, matching `fsk_demod`'s
  2-level smoothing. N-level slicing packs decision bands close together, so
  per-sample discriminator noise scatters symbols across adjacent bands; a
  ~sps/2 moving average fixes it (4-FSK demo 33/40 → 40/40). Default 0 keeps
  the per-sample output byte-identical. `examples/fsk_decoder.py`'s 4-FSK
  branch now calls the library instead of hand-smoothing (the inline TODO
  from patch 0010 is closed); 2-FSK 40/40, 4-FSK 39/39 preserved.
- **`examples/fm_receiver.py` decimates the channel to 250 kHz before the
  discriminator**, mirroring the live path. 250 kHz divides every common
  HackRF rate (2/4/8/10/20 Msps) so capture→intermediate is pure decimation
  (2 Msps: ×8), then a small 24/125 resample to 48 kHz; odd rates fall back
  to a resample. Discriminator + de-emphasis now run on ~8× fewer samples;
  output audio spectrum unchanged (program in-band, pilot at −53 dB).
- **`.gitattributes` added** (LF policy) but normalization deliberately NOT
  run in this patch — a whole-tree `git add --renormalize` is its own commit
  so its whitespace diff never mixes with logic. Still pending.

Note: observed a rare (~1 in 6) flaky test failure during full-suite runs,
not from this patch (the new tests are seeded and hammer clean). Unidentified;
logged in TODO to catch with a seed sweep.

## 2026-09-20 — Capture validation, the dead-sample fix, and first real corpus

The shipped `sample_data/fm_2Msps.iq` was found to contain **no signal** — it
peaked at ~2 of 128 ADC counts (LO/DC only). Chasing that one bad file surfaced
a chain of related defects and produced the whole capture-validation layer.
Suite: 654 → 725 passed.

**Root causes found (not just the dead file):**
- **`capture_health` false positive.** The carrier check compared in-channel
  power against the *outer band edges* — exactly where the HackRF's baseband
  anti-alias filter rolls off — so receiver noise alone measured tens of dB of
  "excess" and passed. An empty capture at healthy gain looked fine. Fixed to
  reference a mid-band annulus, average in linear power, and use the loader's
  /128 full-scale.
- **Gain search optimized level, not signal.** All three collection tools
  walked the VGA (which amplifies signal and noise together) until the ADC
  level looked right, never touched the +14 dB RF amp, and never checked for a
  carrier — so on a weak antenna they amplified the noise floor into the target
  window and declared success. Replaced by `core.gain_search.search_gain`:
  front end (amp+LNA) chosen by a measured quality metric first, VGA for level
  second, and it refuses "ok" when the level converged on noise.
- **No "is this actually a station" check existed.** Added
  `fm_pilot_excess_db` — the 19 kHz stereo pilot standing above the demodulated
  noise floor. Amplified noise can fill the ADC and even hump the channel, but
  cannot manufacture a pilot. It **channelizes before discriminating**: at the
  full 8 Msps capture rate a real, by-ear-verified broadcast measured only
  +1.8 dB unfiltered vs +17–27 dB channelized (this bug would have *rejected*
  good captures — caught only because real data was on hand).
- **`capture_array` (hackrfpy) drops samples on Windows.** The stdout-pipe
  streaming path corrupts phase at 8 Msps while power stats stay healthy;
  every measurement probe was lying. Convicted by A/B against the file path on
  the same board/antenna/station minutes apart: pipe +7.2 dB channel / +1.4 dB
  pilot vs file +17.5 / +19.4 vs reference +20.6 / +26.5. All probes moved to a
  file-path helper (`sources/probe.py`); the corpus recordings always used the
  file path. Reported upstream.

**New capability:**
- `capture_health` (fixed), `fm_pilot_excess_db`, `search_gain`, `peak_counts`,
  `sources/probe.py` (file-path probes + warning dedupe), public `get_window`.
- `tools/preflight_collection.py` — board → band sweep → station scoring by
  pilot → verified-quiet-frequency preflight, with `--simulate`. Sweeps below
  87.5 MHz for the quiet reference.
- `tools/import_reference_capture.py` — import a validated capture as sample
  data with before/after validation and full provenance in the sidecar;
  refuses anything without a pilot.
- `tools/make_synthetic_sample.py` — fully-specified synthetic FM fixture
  (`fm_synthetic_2Msps.iq`).
- `deemphasis` vectorized (bit-identical); 15 kHz audio lowpass added to the FM
  receivers so the 19 kHz pilot no longer leaks into the WAV (−36 → −85 dB).

**First real corpus:** `tools/collect_dev_data.py` collected a 17-capture FM set
at 106.5 MHz (reference lna=40/vga=2/amp on, pilot +23–30 dB): clean reference,
tuning offsets, 2→10 Msps rate ladder, gain ladder (buried→clipping), negative
controls. Two captures correctly flagged by validation (a below-band-quiet
occupancy at 87.7 MHz; a single-digit-count tolerance nit, since fixed).

**Shipped sample replaced:** `fm_2Msps.iq` is now a real ear-verified 103.7 MHz
capture (pilot +27.3 dB) with sha256-stamped provenance, not a placeholder.

**Also:** pyproject aligned to the hackrfpy layout (uv_build backend; dev group
now carries matplotlib/sounddevice/hackrfpy so `uv sync` stops uninstalling
them); examples cleaned (fsk_decoder uses the library `fsk_demod` for 2-FSK,
two_station_link uses `run_link`, dead imports dropped, private `_get_window`
replaced by public API). Regression tests pin every fix above.

Caveat: the pilot metric and gain strategy are validated against real FM
broadcast and a receive-chain model. The digital-modulation captures, absolute
power calibration, and the entire transmit/protocol path remain software-only
(§12).

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