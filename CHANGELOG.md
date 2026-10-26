# Changelog

User-facing changes to `sdr_dsp`, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and versions follow
[Semantic Versioning](https://semver.org/) -- while the major version is 0,
minor releases may change the public API.

The dated engineering narrative behind these entries (what was tried, what
was measured, why a design went the way it did) lives in `docs/LOG.md`.

## [Unreleased]

### Fixed
- Docstrings on `GainStage.default_value` and
  `GainModel.by_name` / `min_settings` / `default_frontend` / `frontend_db`;
  the public-surface test was failing on every CI platform without them.

### Changed
- Package description and README links prepared for PyPI (absolute links to
  the docs; "personal" dropped from the one-line description).
- Python 3.14 added to the CI matrix and the trove classifiers.

### Added
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
