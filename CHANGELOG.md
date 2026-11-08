# Changelog

User-facing changes to `sdr_dsp`, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) -- while the major version is 0,
minor releases may change the public API.

The dated engineering narrative behind these entries (what was tried, what
was measured, why a design went the way it did) lives in `docs/LOG.md`.

## [Unreleased]

### Fixed (loop characterization pass)
- `carrier_recovery` Costas detectors are normalized by the input rms. They
  are linear in amplitude (sign(I)*Q is |s|*sin(error)), so the loop gains
  scaled with the input level: a 100x stronger input ran 100x hotter and
  lost lock, a 0.01x input never acquired. Decision-directed was unaffected.
  The pull-in range (0.02 cycles/sample order 2, 0.01 order 4 at
  loop_bw 0.01) and lock times are now stated in the docstring and pinned.
- `symbol_sync` could not track a symbol-rate error at all: its loop gains
  were far too small for the bandwidth they claimed (a 1% sps error was
  simply free-run, BER 0.37, and the count-only test could not tell). The
  loop filter now uses the same standard second-order (loop_bw, damping)
  parametrization as `carrier_recovery`, and each TED's output is divided by
  its measured S-curve slope (`TED_GAIN`) so the error is in samples and one
  loop_bw means the same thing for every method.
- The early-late TED was dead: its S-curve on random data was ~20x weaker
  than assumed AND inverted, so it could hold nominal timing (the one case
  tested) but never track. It is now the gate projected onto the current
  symbol, Re{conj(s) * (late - early)}, with the right sign. Mueller & Muller
  now uses both rails (the real-only form threw away half of QPSK's timing
  information).
- The pull-in table measured over 10 bit patterns is in the `symbol_sync`
  docstring and pinned by tests: Gardner and M&M take +/-0.5% at loop_bw
  0.01, 1-1.5% at 0.02, 2-3% at 0.05; early-late half that and unstable at
  0.05. `LoopDiagnostics.locked` for the timing loop now means "the applied
  correction has settled" (rms step below `lock_threshold`, default sps/16)
  -- and the docstring says plainly that it does not establish signal
  presence; a narrow loop on noise settles too.
- `rrc_taps(0.35, 8, 8)` -- beta in the sps slot -- used to become sps=0 and
  return an all-NaN filter silently; arguments are validated.
- `deemphasis` is discretized with a pre-warped bilinear transform so the
  -3 dB corner lands exactly on 1/(2*pi*tau). The backward-Euler one-pole it
  replaces sat 0.5-0.8 dB low at 48 kHz (dt/tau = 0.28 is not small) and
  made the 75 us and 50 us curves hard to tell apart; no test pinned the
  constant until now. `deemphasis_coefficients(sample_rate, tau_us)` is the
  shared source for `deemphasis` and `StreamingDeemphasis` (which now carries
  state with lfilter zi/zf).
- `channelize_bank(decim=N//2)` is a true 2x-oversampled polyphase bank: the
  commutator advances N/2 per frame and channel k is rotated by
  exp(-j*2*pi*k*m*D/N). It equals a direct mix-filter-decimate to correlation
  0.9999 and carries the prototype's transition band unaliased (a tone there
  reads at its true offset; the critically-sampled bank folds it). The
  previous implementation interpolated the critically-sampled output, which
  could not do that. `decim = N` output is unchanged.
- Dtype policy codified and pinned (`core/__init__` docstring,
  `test_public_surface.py::test_dtype_policy`): IQ out is complex64 (complex128
  in is processed as complex64), real-valued results are float64.
  `normalize` (preserved complex128) and `matched_filter` (float32 for
  complex64 input) were the two outliers and now conform.
- `StreamingFIR` carries history across a block dtype change (see below).

### Changed
- Docs consolidated: root README is a short pointer; `sdr_dsp/README.md` is
  the canonical install + quickstart (the PyPI page); root `TODO.md` is
  retired into `docs/TODO.md` (open items) and `docs/LOG.md` (history);
  `docs/MODULARITY.md` is merged into `docs/ADDING_AN_SDR.md`;
  `docs/PCAP_CONVERTER_NOTE.md` moves to `docs/notes/`; `docs/README.md`
  leads with user docs; CONTRIBUTING and the PR template agree (small PRs
  welcome, features via an issue first); incident-history comments in
  `pyproject.toml` and `tests.yml` shortened with pointers into LOG.
- `sdr_dsp/sample_data/hackrf_one/` is ignored; the 150 MB of raw captures
  move to release assets (see `sample_data/README.md`).
- `docs/dev_handbook/generate_api_handbook.py` gained `-o FILE` and
  `--check FILE`; the lint CI job now runs `--check` against the checked-in
  handbook and fails when it is stale. The handbook itself is regenerated
  (8,160 lines; the checked-in copy had drifted ~1,100 lines behind the code)
  and the `core.features` section the generator had lost is back.
- `StreamingFIR` carries its delay line across a block dtype change
  (complex64 -> complex128, say) instead of silently zeroing it, which put a
  filter-length click into the output at the switch; a complex -> real change
  raises `TypeError` rather than discarding the imaginary history (call
  `reset()` for a genuinely new stream).
- The two-tier export split is now documented and pinned: `sdr_dsp.X` is the
  signal-processing surface; the radio-characterization layer (impairment
  synthesis, fingerprint features, capture health, gain-search engine and
  models, `get_window`) is `sdr_dsp.core.X` only. `core/__init__.py` says why
  and `test_public_surface.py` fails if a new name lands in a tier by accident.
- Test suite trimmed by 92 tests with no loss of coverage: the per-example
  `ast.parse` check (pyflakes already parses every file), the per-example
  subprocess import check for the ~40 examples that `test_examples_run.py`
  runs end to end (now only the 8 hardware/interactive examples get it), and
  six tests in `test_examples_demods.py` that imported no example code and
  duplicated core tests line for line.
- Docstrings: `bpsk_modulate` stated the bit->sign mapping backwards (it is
  bit 1 -> +1, matching `bpsk_demod`); `fingerprint_vector` now says its
  `sample_rate` argument is accepted for signature stability and unused;
  `tests/conftest.py` no longer points at a TODO item closed in patch 0014;
  a `measure.py` docstring no longer cites TODO wording that exists in neither
  TODO; `tree_map.py` no longer claims to belong to another project.
- `fm_demod` raises `ValueError` when given `deviation_hz` without
  `sample_rate` or vice versa. It used to fall through to the raw rad/sample
  output silently, leaving the caller with audio ~1e5x too quiet and no hint.
- `resample_poly` raises `ValueError` for a non-integer `up` or `down`
  (`int(2.7)` silently made a requested 2.7x resample a 2x one). Integral
  floats (`2.0`) and numpy integers are still accepted.
- `psd()` raises `ValueError` when `len(iq) < nfft` (see Fixed below).
- A bare `pytest` now deselects the `hardware` tests via `addopts`, matching
  what CI runs; `pytest -m hardware` still selects them.
- Docs: README named sinks that do not exist (`WavSink`, `IQSink`; the API is
  `write_wav` / `write_iq`), counted 43 examples (47) and 641 tests (1,200+).
  REFERENCE named the wrong build backend (hatchling; it is `uv_build`), the
  wrong handbook path, two different stale test counts, and a module map
  missing `gain_search.py`, `stream/stages.py`, `sources/probe.py`,
  `demod/phase.py` and the FM-CFO / capture-health functions. MODULARITY §2,
  ADDING_AN_SDR and `docs/TODO.md` described the `GainModel` generalization as
  future work; it shipped in 0.2.0 and is now described as it is. Hard-coded
  test counts are gone from the prose: CI reports the number.

### Fixed
- `agc()` tracks the INPUT level and steers the gain toward
  `target / tracked_input_level`. The previous loop smoothed the OUTPUT level,
  which lags the very gain it steers: after a +20 dB step the gain cut at
  `attack` while the level estimate could only fall at `decay`, so the error
  stayed on the "cut" side and the output collapsed to ~0 before crawling back
  over thousands of samples. The step response is now monotone (output comes
  down to target and stays) and the fast-attack / slow-decay contract is
  pinned by a test for the first time.
- `ssb_modulate` doubled the DC bin along with the positive frequencies, so
  the analytic signal of a constant message was twice the message. DC and
  the Nyquist bin are now kept at 1x (the `scipy.signal.hilbert` weighting);
  for zero-mean audio nothing changes.
- `am_demod([])`, `ook_slice([])` and `evm_stats([])` return empty / NaN
  results cleanly instead of a raw numpy `ValueError` or a `RuntimeWarning`.
- Tests: four behaviours that a mutation pass showed were unprotected now have
  tests -- the AGC level tracker's attack/decay assignment, `estimate_fm_cfo`'s
  click trim (disabling it changed nothing), `find_frames`' inclusive
  `max_sync_errors` boundary, and `fsk_demod("auto")`'s power weighting (an
  unweighted mean passed, because every fixture padded with exact zeros).
  `test_channel.py`'s low-SNR test asserted only inside an `if` that never
  fired at 0 dB; it now sweeps 12 dB where corrupt frames ARE found and
  requires every `crc_ok` frame to carry the true payload.
- `agc()` no longer poisons its output with NaN after a run of exact digital
  silence. With `max_gain=None` the level estimate decayed to the eps floor,
  the gain overflowed to `inf`, and every later sample -- including signal
  after the gap -- came out NaN. The loop now holds its gain through silence
  and resumes tracking when signal returns.
- The streaming `AGC` stage carried `|last output sample|` as its level state
  instead of the loop's smoothed level, so a streamed run diverged from the
  whole-record run by ~5% at every block boundary. It now carries the exact
  loop variable; streamed and whole-record output are bit-identical.
- `agc(mode="peak")` was validated and then ignored (both modes ran the rms
  tracker). Peak mode is now a real peak detector: instant attack on the held
  input peak, release at `decay`.
- `symbol_sync` was amplitude-dependent: the TED output scales with |x|^2 and
  drove the loop unnormalized, so a strong input (100x) made the loop skip and
  emit ~2x too many symbols. The TED is now normalized by signal power and each
  correction is clamped to +/- sps/4. `damping` was accepted and never read;
  the loop is now a second-order PI loop and `damping` sets its integral gain.
- `psd()` returned a flat -200 dB spectrum when `len(iq) < nfft` instead of
  failing, so `occupied_bandwidth` and `snr_db` reported confident nonsense on
  short inputs (99% of the sample rate; -3 dB). It now raises `ValueError`.
- `Pipeline.run()` and `Pipeline.stream()` never flushed stages with look-ahead
  (`StreamingResampler`), silently dropping the tail of every run. Both now
  call the new `Pipeline.flush()` when the source is exhausted (not when cut
  short by `max_blocks`); a pipeline run yields the same sample count as the
  whole-record function.
- Tests: the AGC streaming test compared boundary steps against a set that
  contained them (could not fail) and the whole-vs-streamed test allowed 15%
  (hid the state-carry bug); both are exact now. The QPSK Gray test asserted a
  list typed into the test rather than the demod's output. New tests pin the
  FM discriminator's absolute Hz scale and the resampler's absolute gain
  without `sample_data`, so a 2x scale error or a missing `up` gain
  compensation fails in CI rather than only on a checkout with the sample.
- Docstrings on `GainStage.default_value` and
  `GainModel.by_name` / `min_settings` / `default_frontend` / `frontend_db`;
  the public-surface test was failing on every CI platform without them.

### Changed
- Package description and README links prepared for PyPI (absolute links to
  the docs; "personal" dropped from the one-line description). The README's
  "hooks that aren't implemented" line is gone (nothing exported is a stub)
  and its quick-start now calls the `tune_to_baseband` it imports.
- Python 3.14 added to the CI matrix and the trove classifiers.
- `zip()` calls carry an explicit `strict=` throughout (bugbear B905):
  `strict=True` wherever the sequences must match, which turns a silent
  truncation into an error.
- `ssb_receiver.py` builds its test signal with `ssb_modulate`;
  `spectrum_analyzer.py` draws on axes from `plot_spectrum`; `cfo_demo.py`
  gains `--fm` for `estimate_fm_cfo` / `correct_fm_cfo`.

### Quality gates
- ruff: bugbear (`B`) enabled; `docs/dev_handbook/*.py` now linted in CI.
- Coverage floor raised from 85% to 95%.
- `test_public_surface.py` also requires a docstring on every module.

### Added
- `examples/gain_search_demo.py`: the device-agnostic `GainModel` /
  `search_gain_model` engine run on three simulated radios (HackRF, RTL-SDR
  and a hand-built two-stage model). `docs/EXAMPLES.md` also catalogs the four
  scripts it was missing (`transmit_examples`, `fm_band_explorer`,
  `collect_sample_data`, `sdr_adapter_template`).
- 89 edge-case tests (`tests/test_core_edges.py`, `test_io_link_edges.py`,
  `test_stream_sources_edges.py`) covering empty-input guards, argument
  validation, every streaming stage's `reset()`, and alternative branches
  that had no test (LSB modulation, the odd-length Hilbert kernel, the RRC
  singularity tap, explicit ASK/FSK decision levels, SigMF path resolution,
  gain-search escalation and back-off). Line coverage 95% -> 99.6%.
- Hardware / bench-report issue form and a pull-request template.
- This changelog.

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

[Unreleased]: https://github.com/LC-Linkous/sdr_dsp_python/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/LC-Linkous/sdr_dsp_python/releases/tag/v0.2.0
