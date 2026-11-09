# Changelog

User-facing changes to `sdr_dsp`, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) -- while the major version is 0,
minor releases may change the public API.

The dated engineering narrative behind these entries (what was tried, what
was measured, why a design went the way it did) lives in `docs/LOG.md`.

## [Unreleased]

## [0.3.0] - 2026-11-08

First release on PyPI (`pip install sdr-dsp`), and the first one that was
reviewed rather than only tested. The headline is correctness: a review and
mutation pass over the 0.2.0 suite found six runtime bugs under a green suite
(AGC poisoning its output with NaN after silence and collapsing after a level
step; a timing loop that could not track a symbol-rate error and an early-late
detector that was dead; a Costas loop that depended on signal amplitude; a PSD
that returned a silent -200 dB floor on short input; a pipeline that dropped
the resampler's tail), and the tests that had let them through were rewritten
so that each of those bugs now fails a test. A second pass over the whole
library found nine more (ARQ reporting undelivered frames done, `find_frames`
swallowing the frame behind a near-miss, SigMF paths mangled by a dot in the
stem, a streaming PSD that buffered without bound, a clip detector blind to
the +127 rail, among them). Every recovery loop has a measured, documented
pull-in range and lock time.

Because this is the version downstream code will pin against, the public API
was also made consistent before it was frozen, and that is a breaking release
from 0.2.0: six functions take their arguments in a different order, several
parameters were renamed, and a dozen functions now raise where they used to
return something plausible and wrong. Every entry is under **Breaking** below,
with the migration in the same line. Every signature is now annotated, so the
shipped `py.typed` means what it says.

For downstream consumers: `psd(scaling="dbfs")`, `StreamingPSD`, a
machine-readable `capture_health` verdict, SigMF sidecars and a fetch script
for the raw reference captures (which have left git), and a verified wheel +
sdist that contain the package only. The docs were consolidated (one README,
one TODO, one HAL document) and the API handbook is generated and gated in CI.
Development status moves from Alpha to Beta: the software path is complete and
defended; only FM has been validated against real RF, and that is the next
phase.

### Breaking

Signatures and names:
- Every function that takes a signal, a sample rate and a frequency now takes
  them in that order: `(iq, sample_rate, X_hz)`, the order `channelize`,
  `psd`, `capture_health` and the rest already used. Reordered:
  `tune_to_baseband`, `frequency_shift`, `fm_demod`, `fm_modulate`, `add_cfo`,
  `add_phase_noise`. Before, `tune_to_baseband(iq, offset_hz, sample_rate)`
  and `fm_demod(iq, deviation_hz, sample_rate)` had the two floats swapped
  relative to everything else, and a swapped call ran without error. Keyword
  calls are unaffected; positional calls must be reordered.
- `channel_bw` -> `channel_bw_hz` in `channelize`, `capture_health` and
  `estimate_fm_cfo`; `fm_stereo_decode(audio_bw=, pilot_bw=)` ->
  `audio_bw_hz=`, `pilot_bw_hz=`. Every frequency-valued parameter now ends
  in `_hz`.
- `channelize_bank` always returns `(channels, new_rate, center_freqs)`; the
  `return_freqs` flag is gone (it returned a pair by default and a triple on
  request). Callers unpacking two values must unpack three.
- `add_noise(iq, snr_db, seed=None, rng=None)`: `seed` is a positional
  `int | None` (was keyword-only `rng` only); a `str` or a Generator in the
  `seed` slot is a `TypeError`.
- `fsk_demod(threshold_hz=None)` is the canonical "estimate it from the
  signal"; `"auto"` is still accepted as a synonym. Elsewhere `None` means
  auto throughout (`decim=None`, `lock_threshold=None`, `max_gain=None`).
- `nask_slice(levels=[...])` and `fsk_demod_nlevel(thresholds=[...])`: an
  explicit level/threshold list defines the level count; `n_levels` is
  ignored when it is given (it used to have to agree, or raise).
- `sdr_dsp.__all__` lists functions and classes only. `core`, `io`, `sinks`,
  `stream` and `link` are still importable as `sdr_dsp.core` etc., but
  `from sdr_dsp import *` no longer shadows the standard-library `io` module
  (it did). `Pipeline` stays exported at the top level.

Return types:
- The symbol-domain demodulators (`bpsk_demod`, `dbpsk_demod`, `qpsk_demod`,
  `dqpsk_demod`, `psk8_demod`, `qam16_demod`) return a `DemodResult` NamedTuple
  with fields `bits` and `soft` (soft decisions / constellation points). It
  unpacks exactly like the tuple it replaces (`bits, soft = qpsk_demod(x)`)
  and also reads as `result.bits`. The sample-domain demodulators (`fm_demod`,
  `am_demod`, `ook_slice`, `fsk_demod`, ...) return a bare array as before;
  the docstring on `DemodResult` says which family is which.
- `find_frames` rejects a `DemodResult` / tuple with a `TypeError` naming
  `.bits` (it used to coerce the pair into a 2-D array and find nothing),
  and rejects non-1-D or non-0/1 input with a `ValueError`.
- `occupied_bandwidth` is defined as the width between the frequencies at
  which the cumulative power reaches `(1-fraction)/2` and `1-(1-fraction)/2`
  of the total -- the standard 99% OBW. The previous definition grew the
  interval outward from the peak bin, so the same signal measured differently
  depending on where in the span it sat. Values change for any asymmetric
  spectrum; a centred tone or flat band reads the same.
- Dtype policy codified and pinned: IQ out is `complex64` (complex128 in is
  processed as complex64), real-valued results are `float64`. `normalize`
  (preserved complex128) and `matched_filter` (float32 for complex64 input)
  now conform.

Now raise instead of returning something wrong (`ValueError` unless noted):
- `psd()` when `len(iq) < nfft` (returned a flat -200 dB floor).
- `fm_demod` given `deviation_hz` without `sample_rate` or vice versa (fell
  through to rad/sample output ~1e5x too quiet).
- `resample_poly` for a non-integer `up`/`down` (`int(2.7)` made 2.7x into
  2x); `decimate`/`interpolate` for `factor < 1`.
- `deemphasis` on complex input (`TypeError`; the imaginary part was dropped
  with a warning).
- `StreamingFIR` on a complex -> real block dtype change (`TypeError`; call
  `reset()` for a new stream).
- `qpsk_modulate` on an odd bit count (the last bit was silently dropped).
- `slice_to_symbols` / `sample_symbols` with `samples_per_symbol < 1`
  (returned the input unchanged).
- `rrc_taps` with `sps < 1`, `beta` outside [0, 1] or `span < 1` (an
  argument-order slip produced an all-NaN filter silently).
- Every public function validates its scalar parameters with one message
  shape, `"<name> must be <range>, got <value>"`: sample rates, cutoffs,
  bandwidths, `num_taps`, `samples_per_symbol`, `loop_bw`, `damping`,
  `order`, `method`, `window`, `fraction`, `nfft`, `block_size`, byte-like
  payloads and 0/1 bit arrays. Strings that happen to parse as numbers
  (`"10"`) are rejected (`TypeError`), not coerced. 43 of these cases are
  pinned in `tests/test_validation_messages.py`.

Semantics:
- `ARQ` "done" means DELIVERED, not acknowledged: it is emitted as the send
  window's base slides past a frame, i.e. when it and every frame before it
  are acknowledged -- the condition under which the in-order receiver has
  handed it to its application. In windowed mode the receiver ACKs frames it
  can only buffer, and those were being reported done.
- `ARQ` failure is final. When a frame exhausts `max_retries` the engine
  emits `("failed", seq)` for it AND for every other frame in flight or
  queued (`None` for queued messages that never got a seq), sets `.failed`,
  stops transmitting, and `send()` raises `RuntimeError` until `reset()`
  (new) is called on both ends. Previously it slid the window on:
  stop-and-wait then reported later messages "done" that the receiver never
  delivered, and windowed mode delivered nothing at all for the rest of the
  session.
- `find_frames` advances one bit after a match whose CRC fails instead of
  skipping the claimed body (a near-miss sync with a plausible length byte
  swallowed the real frame behind it 2% of the time). A corrupted frame may
  now be reported more than once at nearby offsets, all `crc_ok` False.
- `capture_health`: `ok` is now `verdict == "ok"`; two new checks
  (`clip_fraction`, `dc_ratio`) can fail a capture that passed before.
- `LoopDiagnostics.locked` for `symbol_sync` means "the applied correction
  has settled" (rms step below `lock_threshold`, default sps/16); it does not
  establish signal presence, and the docstring says so.

### Added
- Type annotations on every function in the package, checked in CI with
  `mypy --disallow-untyped-defs`. `py.typed` shipped since 0.2.0 with most
  signatures unannotated; downstream type-checkers now see real types.
  `carrier_recovery`, `symbol_sync` and `Pipeline.run` carry `@overload`s on
  their `diagnostics=` / `profile=` / `sink=` flags, so the return type is
  the array, the pair or the stats tuple rather than a union; and the lazily
  imported `sdr_dsp.Pipeline` / `sdr_dsp.io` / `.sinks` / `.stream` / `.link`
  are declared under `TYPE_CHECKING`, so `from sdr_dsp import Pipeline` is no
  longer `Any` to a type checker. `tests/test_typing_overloads.py` runs mypy
  on a snippet and pins the revealed types.
- `DemodResult` (`sdr_dsp.DemodResult`, `core.demod.result`): the NamedTuple
  the coherent demodulators return (see Breaking).
- `psd(..., scaling="density" | "dbfs")`. Default "density" is unchanged
  (power/Hz). "dbfs" divides by sum(window)^2 so a full-scale complex tone on
  a bin reads 0 dBFS per bin regardless of nfft and sample rate -- the
  spectrum-display scale. Pinned: 0.0 +/- 0.1 dB at two FFT sizes, -1.42 dB
  Hann scalloping for a half-bin tone, -6.02 dB at half amplitude.
- `stream.StreamingPSD(nfft, avg=1, window="hann", scaling="dbfs",
  sample_rate=None, center_freq=0.0)`: buffers blocks of any size and returns
  `psd()` of each `avg * nfft` samples (else None); computes every spectrum a
  block completes (all in `.last_batch`, the newest returned) so a 64k-sample
  block stream does not buffer without bound; `.freqs` is the axis,
  `frames_emitted` counts. `retune(center_freq, drop_buffer=False)` relabels
  the axis for a display retune and keeps the buffered samples;
  `drop_buffer=True` for an RF retune. It calls `core.psd` for the math, so
  its output is bit-equal to `psd()` on the same samples (tested over four
  block-size patterns). `sample_rate` is optional because "dbfs" does not
  need it; "density" does and is refused without it.
- `capture_health` returns a machine-readable `verdict` -- one of
  `HEALTH_VERDICTS = ("clipping", "under_driven", "stuck_dc", "no_carrier",
  "ok")`, the worst failing check in that order -- plus `clip_fraction`
  (samples at the converter's last code on either rail; threshold
  `max_clip_fraction=1e-3`) and `dc_ratio` (|mean| over the rms of the rest;
  threshold `max_dc_ratio=1.0`, with `dc_offset` alongside). The thresholds
  live here so every consumer applies the same ones; `reasons` still lists
  every failure. On the two raw HackRF captures `dc_offset` matches the
  number hackrfpy measured independently at capture time to four digits
  (pinned by a test when the payload is present).
- `ARQ.reset()`, `ARQ.failed`, `ARQ.failed_seq` (see Breaking).
- `deemphasis_coefficients(sample_rate, tau_us)`: the shared pre-warped
  bilinear source for `deemphasis` and `StreamingDeemphasis`.
- `Pipeline.flush()`; `run()` and `stream()` call it when the source is
  exhausted (not when cut short by `max_blocks`).
- `core._validate`: the shared argument-validation helpers behind the
  uniform error messages (private; listed so the pattern is findable).
- SigMF sidecars for the two raw 8 Msps captures
  (`sample_data/hackrf_one/*.sigmf-meta`, with the payload sha256 and the gains
  from the capture-time health report). The sidecars and `.report.json` files
  stay in git; only the 80 MB `.iq` payloads are ignored and distributed as
  release assets, so `FileSource` can read a downloaded copy.
- `tools/fetch_sample_data.py`: downloads the release-asset payloads next to
  their sidecars and verifies them against the sidecar's `sdr_dsp:sha256`
  (refuses a mismatch, installs nothing). `--check` verifies only. Tested
  against a `file://` release; standard library only.
- `.github/workflows/release.yml`: on a `v*` tag, checks the tag against
  `pyproject.toml`, re-runs suite/ruff/mypy/handbook on the tagged commit,
  builds, `twine check`s, verifies the sdist holds the package only, publishes
  to PyPI via trusted publishing (no stored token) and attaches the artifacts
  to the GitHub release. `docs/RELEASING.md` is the checklist around it.
  Packaging verified: sdist (110 KB) and wheel (135 KB) contain `src/`,
  LICENSE, README and `pyproject.toml` only -- no `sample_data/`, `examples/`
  or `tools/`.
- `pyproject.toml`: Development Status Alpha -> Beta; `Changelog` and
  `Documentation` project URLs; Python 3.14 in the CI matrix and classifiers.
- `examples/gain_search_demo.py`: the device-agnostic `GainModel` /
  `search_gain_model` engine run on three simulated radios (HackRF, RTL-SDR
  and a hand-built two-stage model). `docs/EXAMPLES.md` also catalogs the four
  scripts it was missing (`transmit_examples`, `fm_band_explorer`,
  `collect_sample_data`, `sdr_adapter_template`).
- Tests: 89 edge-case tests (`tests/test_core_edges.py`,
  `test_io_link_edges.py`, `test_stream_sources_edges.py`) covering
  empty-input guards, every streaming stage's `reset()`, and branches that had
  no test (LSB modulation, the odd-length Hilbert kernel, the RRC singularity
  tap, SigMF path resolution, gain-search escalation and back-off); line
  coverage 95% -> 99.6%. Tests for behaviour a mutation pass found unguarded:
  the AGC tracker's attack/decay assignment, `estimate_fm_cfo`'s click trim,
  `find_frames`' inclusive `max_sync_errors` boundary, `fsk_demod` auto
  threshold's power weighting, `write_wav` at exactly +/-1.0, `find_bursts`
  `min_len`, 8-PSK Gray labels, `symbol_sync` pull-in over 10 bit patterns,
  the FM discriminator's absolute Hz scale and the resampler's absolute gain
  (so a 2x scale error fails in CI, not only on a checkout with the sample).
- `docs/img/dc_spike_receiver.svg` and `dc_spike_offset_tuning.svg`: the two
  figures `docs/DC_SPIKE.md` referenced but the repo never contained.
- Hardware / bench-report issue form; this changelog.

### Changed
- Docs consolidated: root README is a short pointer; `sdr_dsp/README.md` is
  the canonical install + quickstart (the PyPI page); root `TODO.md` is
  retired into `docs/TODO.md` (open items) and `docs/LOG.md` (history);
  `docs/MODULARITY.md` is merged into `docs/ADDING_AN_SDR.md`;
  `docs/PCAP_CONVERTER_NOTE.md` moves to `docs/notes/`; `docs/README.md`
  leads with user docs; CONTRIBUTING says what is useful (bug reports,
  reproductions, captures, via issues; pull requests are not being taken
  while the API settles) and the PR template says the same; incident-history
  comments in `pyproject.toml` and `tests.yml` shortened with pointers into
  LOG. README named sinks that do not exist (`WavSink`, `IQSink`; the API is
  `write_wav` / `write_iq`) and carried stale example/test counts; REFERENCE
  named the wrong build backend (it is `uv_build`), the wrong handbook path,
  and a module map missing five modules; MODULARITY, ADDING_AN_SDR and
  `docs/TODO.md` described the shipped `GainModel` as future work. Hard-coded
  test counts are gone from the prose: CI reports the number.
- `sdr_dsp/sample_data/hackrf_one/` is ignored; the 150 MB of raw captures
  move to release assets (see `sample_data/README.md`).
- `docs/dev_handbook/generate_api_handbook.py` gained `-o FILE` and
  `--check FILE`; the lint CI job runs `--check` against the checked-in
  handbook and fails when it is stale. The handbook is regenerated (the
  checked-in copy had drifted ~1,100 lines behind the code) and the
  `core.features` section the generator had lost is back.
- The two-tier export split is documented and pinned: `sdr_dsp.X` is the
  signal-processing surface; the radio-characterization layer (impairment
  synthesis, fingerprint features, capture health, gain-search engine and
  models, `get_window`) is `sdr_dsp.core.X` only. `core/__init__.py` says why
  and `test_public_surface.py` fails if a new name lands in a tier by accident.
- `StreamingFIR` carries its delay line across a complex64 -> complex128
  block dtype change instead of silently zeroing it (a filter-length click
  at the switch).
- `symbol_sync` uses the same standard second-order (`loop_bw`, `damping`)
  parametrization as `carrier_recovery`, and each TED's output is divided by
  its measured S-curve slope (`TED_GAIN`) so the error is in samples and one
  `loop_bw` means the same thing for every method. The pull-in table
  measured over 10 bit patterns is in the docstring and pinned: Gardner and
  M&M take +/-0.5% at loop_bw 0.01, 1-1.5% at 0.02, 2-3% at 0.05;
  early-late half that and unstable at 0.05.
- `carrier_recovery` pull-in range (0.02 cycles/sample order 2, 0.01 order 4
  at loop_bw 0.01) and lock times are stated in the docstring and pinned.
- `run_sim` logs tick events and drains both engines' intentions before the
  idle check, so `replay()` reproduces the SENDER (retransmissions included)
  and the "done"s created by a run's last ACK are no longer lost.
- `search_gain_model` steps to the next grid value when snapping lands on
  the current one (the R820T's non-uniform grid stalled the walk at 3.7 dB),
  and a budget-exhausted walk reports the best probe it TOOK -- settings and
  counts from the same probe, ties broken toward more gain when too weak and
  less when too strong -- instead of the last counts against the next,
  untried settings.
- Test suite trimmed by 92 tests with no loss of coverage (per-example
  `ast.parse` and subprocess-import checks that duplicated `pyflakes` and
  `test_examples_run.py`; six tests that imported no example code). A bare
  `pytest` deselects the `hardware` tests via `addopts`, matching CI;
  `pytest -m hardware` still selects them.
- Docstrings: `bpsk_modulate` stated the bit->sign mapping backwards (it is
  bit 1 -> +1, matching `bpsk_demod`); `fingerprint_vector` says its
  `sample_rate` argument is accepted for signature stability and unused;
  stale TODO pointers in `tests/conftest.py` and `measure.py` removed;
  `tree_map.py` no longer claims to belong to another project;
  `GainStage.default_value` and the `GainModel` accessors are documented.
- Package description and README links prepared for PyPI (absolute links to
  the docs; "personal" dropped from the one-line description). The README's
  "hooks that aren't implemented" line is gone (nothing exported is a stub)
  and its quick-start calls the `tune_to_baseband` it imports.
- `zip()` calls carry an explicit `strict=` throughout (bugbear B905).
- `ssb_receiver.py` builds its test signal with `ssb_modulate`;
  `spectrum_analyzer.py` draws on axes from `plot_spectrum`; `cfo_demo.py`
  gains `--fm` for `estimate_fm_cfo` / `correct_fm_cfo`.

### Fixed
- `agc()` tracks the INPUT level and steers the gain toward
  `target / tracked_input_level`. The previous loop smoothed the OUTPUT level,
  which lags the very gain it steers: after a +20 dB step the gain cut at
  `attack` while the level estimate could only fall at `decay`, so the output
  collapsed to ~0 before crawling back over thousands of samples. The step
  response is now monotone and the fast-attack / slow-decay contract is
  pinned by a test for the first time.
- `agc()` no longer poisons its output with NaN after a run of exact digital
  silence. With `max_gain=None` the level estimate decayed to the eps floor,
  the gain overflowed to `inf`, and every later sample came out NaN. The loop
  holds its gain through silence and resumes tracking when signal returns.
- `agc(mode="peak")` was validated and then ignored (both modes ran the rms
  tracker). Peak mode is a real peak detector: instant attack on the held
  input peak, release at `decay`.
- The streaming `AGC` stage carried `|last output sample|` as its level state
  instead of the loop's smoothed level, so a streamed run diverged from the
  whole-record run by ~5% at every block boundary. Streamed and whole-record
  output are bit-identical.
- `symbol_sync` could not track a symbol-rate error at all: its loop gains
  were far too small for the bandwidth they claimed (a 1% sps error was
  free-run, BER 0.37, and the count-only test could not tell). It was also
  amplitude-dependent (a 100x input made the loop skip and emit ~2x too many
  symbols); the TED is normalized and each correction clamped to +/- sps/4.
  `damping` was accepted and never read.
- The early-late TED was dead: its S-curve on random data was ~20x weaker
  than assumed AND inverted, so it could hold nominal timing (the one case
  tested) but never track. It is now the gate projected onto the current
  symbol, Re{conj(s) * (late - early)}, with the right sign. Mueller & Muller
  uses both rails (the real-only form threw away half of QPSK's timing
  information).
- `carrier_recovery` Costas detectors are normalized by the input rms. They
  are linear in amplitude, so the loop gains scaled with the input level: a
  100x stronger input ran 100x hotter and lost lock, a 0.01x input never
  acquired. Decision-directed was unaffected.
- `deemphasis` is discretized with a pre-warped bilinear transform so the
  -3 dB corner lands exactly on 1/(2*pi*tau). The backward-Euler one-pole it
  replaces sat 0.5-0.8 dB low at 48 kHz and made the 75 us and 50 us curves
  hard to tell apart. `StreamingDeemphasis` carries state with lfilter zi/zf.
- `channelize_bank(decim=N//2)` is a true 2x-oversampled polyphase bank: the
  commutator advances N/2 per frame and channel k is rotated by
  exp(-j*2*pi*k*m*D/N). It equals a direct mix-filter-decimate to correlation
  0.9999 and carries the prototype's transition band unaliased. The previous
  implementation interpolated the critically-sampled output, which could not
  do that. `decim = N` output is unchanged.
- `ssb_modulate` doubled the DC bin along with the positive frequencies, so
  the analytic signal of a constant message was twice the message. DC and
  the Nyquist bin are kept at 1x; for zero-mean audio nothing changes.
- `save_iq`, `load_iq`, `FileSource` mangled any path with a dot in its stem:
  `fm_98.1MHz_8Msps.iq` was written and looked up as `fm_98.sigmf-*` (two
  chained `with_suffix` calls). Known capture extensions and short letter-led
  suffixes are stripped; a digit-led dotted run is part of the name.
- `capture_health` clip detection missed the +127 rail (8-bit two's
  complement is asymmetric: +127/128 = 0.992, and the threshold was 0.996),
  so a capture pinned at +127 read as stuck DC. Each rail has its own
  threshold; -127 (one code short) does not count.
- `load_iq(offset_samples=N)` past the end of the file raised a numpy
  "negative dimensions" error without `count`; now empty either way, and a
  negative offset is a `ValueError`. `FileSource.read(n)` ignored `count`.
- `fsk_demod(smooth_samples=k)` with k longer than the signal returned k
  samples (np.convolve "same" is max(M, N)); the window is clamped.
- `am_demod([])`, `ook_slice([])`, `evm_stats([])` and
  `estimate_iq_imbalance([])` return empty / NaN / (0, 0) cleanly instead of
  a raw numpy error or a `RuntimeWarning`.
- Tests that could not fail: the AGC streaming test compared boundary steps
  against a set that contained them, the whole-vs-streamed test allowed 15%
  (hid the state-carry bug), the QPSK Gray test asserted a list typed into
  the test rather than the demod's output, `test_channel.py`'s low-SNR test
  asserted only inside an `if` that never fired. All are exact now.

### Quality gates
- ruff: bugbear (`B`) enabled; `docs/dev_handbook/*.py` linted in CI.
- mypy: `disallow_untyped_defs` / `disallow_incomplete_defs` on for the
  whole package.
- Coverage floor raised from 85% to 95%.
- `test_public_surface.py` requires a docstring on every module, pins the
  export tiers and the dtype policy, and checks that `import *` shadows no
  standard-library name.
- API handbook `--check` gates CI.

## [0.2.0] - 2026-10-24

First version prepared for packaging. Not yet published to PyPI. Highlights
of the tree as of this version, by area; see `docs/MODULATIONS.md` for what
"supported" means per scheme and `docs/sdr_dsp_REFERENCE.md` for the full map.

### Added
- **Receive:** FM demodulation with carrier-offset estimate/correct and a
  pilot-locked stereo decoder (validated on real over-the-air captures);
  AM / SSB / CW; OOK / ASK; 2-FSK and N-level FSK; the PSK family; QAM-16;
  spread-spectrum despreading. Digital schemes are verified against
  synthetic ground truth only.
- **Transmit:** matching modulators with pulse shaping, driven through the
  `TXSink` seam; a loopback sink for closed-loop software tests.
- **Link layer:** framing with CRC, and a windowed ARQ engine (window 1 =
  stop-and-wait) with pluggable drivers, proven in closed-loop simulation.
- **Core DSP:** FIR design/apply, polyphase resampling, channelization,
  mixing, spectral estimates, AGC, timing and carrier sync, calibration,
  detection, measurement, and channel-impairment modelling with feature
  extraction (fingerprint F0-F2/F4/F5).
- **Streaming:** `Pipeline` plus block-wise `StreamingFIR`,
  `StreamingResampler` and related stages for live use.
- **I/O:** SigMF read/write; `FileSource` / `ArraySource`; WAV, IQ, plot and
  TX sinks. Device adapters (HackRF via `hackrfpy`) live in `examples/`,
  not the core.
- **Gain search:** device-agnostic `GainModel` / `GainStage` with a
  measured-pilot quality metric and clip back-off.
- `py.typed` marker (most signatures are still unannotated; see
  `docs/TODO.md`).

### Infrastructure
- CI matrix: Windows / Ubuntu / macOS x Python 3.11-3.13, an oldest-declared
  numpy/scipy leg, a build-the-wheel-and-import-it leg, ruff + mypy, and an
  LF line-ending guard.

[Unreleased]: https://github.com/LC-Linkous/sdr_dsp_python/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/LC-Linkous/sdr_dsp_python/releases/tag/v0.3.0
[0.2.0]: https://github.com/LC-Linkous/sdr_dsp_python/releases/tag/v0.2.0