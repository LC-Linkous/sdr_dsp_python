# sdr_dsp — Build Plan

A personal, fully-functional DSP library for software-defined radio, written in
Python. Consumes IQ from `hackrfpy` (its sibling project) or from files, and
turns it into meaning: spectra, filtered channels, demodulated audio, decoded
signals. Built for personal use and as a portfolio piece — **not** published to
PyPI, but built as a real installable library (pyproject + `uv build`) so it has
clean structure, is testable, and is reusable across the author's own projects.

> **Status (current):** The library is functionally complete. Core DSP, the full
> demodulation suite (18 demods across analog/ASK/FSK/PSK/QAM/spread-spectrum),
> the composable recovery layer (carrier + symbol timing), and the streaming
> Pipeline are all built and tested — **149 tests, 61 public functions, 38
> examples**, every demod with both a test and an example. The sections below are
> the original plan and the expansion roadmap (see the bottom); they are kept as
> the design record. What remains is non-code: release polish (README is current;
> screenshots, `uv build` verification) and hardware validation with a live SDR.

---

## 1. Identity and philosophy


- **Name:** `sdr_dsp` — device-agnostic DSP for SDR; HackRF is the current
  source, not the only conceivable one.
- **It is a LIBRARY, not a FRAMEWORK.** Importable functions and classes you
  call and orchestrate yourself. No runtime, no scheduler, no flowgraph engine,
  no GUI. This is the line that keeps it *fully functional* without becoming a
  GNU Radio competitor: GNU Radio is a framework (engine + blocks + scheduler);
  sdr_dsp is the DSP math you'd otherwise wire blocks together to get.
- **Not a GNU Radio competitor — a complement.** GNU Radio remains the tool for
  large real-time flowgraph projects. sdr_dsp is for Python coding/scripting
  where you want direct, classful control over the DSP. Inspiration from GNU
  Radio *features* is welcome; its *scale* is explicitly out of scope.
- **Class-oriented, with functional cores.** Pipelines and sources/sinks are
  classes (the author prefers this); the underlying DSP operations are pure
  functions those classes call.

## 2. The implementation rule (scipy boundary)

The defining engineering decision, stated so it's not re-litigated per function:

> **sdr_dsp implements its own signal-processing operations.** It uses scipy for
> problems that are *solved, hard, and stable* (filter coefficient design), and
> numpy for the FFT. scipy also serves as a **test oracle** — own
> implementations are verified against scipy's known-good results. The radio DSP
> is the library's own; scipy is a design-time tool and a test reference, not a
> runtime crutch.

**Implemented by sdr_dsp (the library's own code):**
- FIR filter *application* (the convolution / overlap method)
- All demodulation — FM (phase-discriminator), AM (envelope), OOK/ASK, FSK
- Frequency translation / mixing (complex-exponential multiply)
- Resampling *operation* (decimation, interpolation, polyphase structure)
- Measurement — power, SNR, occupied bandwidth
- Spectral framing — windowing application, averaging, spectrogram assembly,
  dB scaling

**Borrowed (solved, hard, stable — reinventing is foolish):**
- FFT → `numpy.fft` (standardized, deep to implement well, no educational or
  practical gain in rolling our own)
- Filter *coefficient design* → `scipy.signal` (`firwin`, `butter`, `remez`).
  **scipy designs the taps; sdr_dsp applies them.** This is the clean split.
- `scipy.signal.resample_poly` etc. used as a **verification reference** for the
  library's own resampler (see the benchmark example), not necessarily in the
  hot path.

Rationale the author endorses: filters and FFTs are well-understood and
standardized — no educational value in reimplementing from scratch — so own the
*radio operations* and the *application* logic, borrow the *design* math.

## 3. Architecture — the source/core/sink seam

The one structural principle: **core DSP operates on arrays, never on devices or
files.** Pure functions: `complex64` + sample rate in, processed array out.
Anything touching hardware or disk lives in adapters. This is what makes it
modular and reusable — and lets the author (or anyone) point a different SDR at
it later by writing one adapter.

```
Data flow:   SOURCE  ->  CORE (pure DSP)  ->  SINK
             (where IQ      (the math)         (where it goes:
              comes from)                       wav, plot, file)

Rule: core/ NEVER imports sources/ or sinks/.
```

- **Sources** produce IQ blocks + metadata (sample_rate, center_freq). hackrfpy
  is the *reference* source; a file source is the *workhorse* for development.
- **Core** is pure DSP on arrays. The reusable heart of the library.
- **Sinks** consume results: a WAV writer (audio), a plotter (spectra), a file
  writer (save processed IQ back out).

## 4. File structure

```
sdr_dsp/                              (repo root)
└── sdr_dsp/                          (uv project dir; mirrors hackrfpy nesting)
    ├── pyproject.toml
    ├── README.md
    ├── src/sdr_dsp/
    │   ├── __init__.py
    │   ├── py.typed
    │   ├── core/                     # PURE DSP — no I/O, no device imports
    │   │   ├── __init__.py
    │   │   ├── spectral.py           # psd, spectrogram, windowing, dB scaling
    │   │   ├── filters.py            # design (scipy) + APPLY (ours)
    │   │   ├── resample.py           # decimate/interpolate/resample (ours)
    │   │   ├── mixing.py             # frequency translation (ours)
    │   │   ├── demod.py              # fm, am, ook, fsk (ours)
    │   │   └── measure.py            # power, snr, occupied bandwidth (ours)
    │   ├── sources/                  # adapters: where IQ comes from
    │   │   ├── __init__.py
    │   │   ├── base.py               # IQSource protocol (the seam)
    │   │   ├── file_source.py        # load .iq + SigMF (no hardware)
    │   │   └── hackrf_source.py      # wraps hackrfpy (live capture)
    │   ├── sinks/                    # adapters: where results go
    │   │   ├── __init__.py
    │   │   ├── wav_sink.py           # write audio WAV
    │   │   ├── iq_sink.py            # save processed IQ + SigMF back out
    │   │   └── plot_sink.py          # matplotlib spectra/spectrograms
    │   └── io/
    │       ├── __init__.py
    │       └── sigmf.py              # read/write SigMF (cf32_le for processed)
    ├── examples/                     # end-to-end demos (the 1-5 sequence)
    ├── tests/
    │   ├── conftest.py
    │   ├── helpers/                  # synthetic signal generators (ground truth)
    │   └── ...
    └── sample_data/                  # reuse hackrfpy captures, or fresh ones
```

## 5. Dependencies

```toml
dependencies = [
    "numpy>=1.26",
    "scipy>=1.11",          # filter design + test oracle
    "hackrfpy",             # the IQ source (installed from local wheel/path)
]
[project.optional-dependencies]
plotting = ["matplotlib"]   # matplotlib ONLY — no PyQt5 (Windows wheel issues)
audio = ["sounddevice"]     # OPTIONAL: live audio playback (else write WAV)
[dependency-groups]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]
```

- **hackrfpy** is wired as a **local dependency** (path or the built wheel), not
  from PyPI. During development a path dependency is cleaner so changes flow
  through; the wheel is fine for a pinned snapshot.
- **matplotlib only** for plotting — the PyQt5-on-Windows lesson from hackrfpy
  is already learned.
- **sounddevice** optional: nice for live playback, but the default audio path is
  "write a WAV you can play" so the core demos need no audio hardware/driver.

## 6. File save/load (a first-class requirement)

Saving captures and loading them back is core to the workflow (develop against
files, no board needed). Built on SigMF for interoperability:

- **Load:** `FileSource` reads a `.iq`/`.sigmf-data` + its `.sigmf-meta` sidecar
  into `complex64` with sample_rate + center_freq. Reuses/extends hackrfpy's
  `load_iq` / `read_sigmf_meta`.
- **Save:** `iq_sink` writes processed IQ back out **with updated metadata**.
  Interop note: hackrfpy captures are `ci8` (complex int8, native HackRF);
  sdr_dsp works in `complex64` internally, so *processed* output is written as
  **`cf32_le`** (complex float32, SigMF's canonical RF type) and the
  `core:datatype` field is set accordingly. Round-trip (load ci8 → process →
  save cf32 → reload) must be lossless within float precision.
- This makes every example reproducible from saved data and lets processed
  results (e.g. a filtered channel) be saved as a new recording.

## 7. Build order — foundation first, then examples

DSP reality: the exciting examples sit on shared primitives, so the primitives
come first even though the examples are the visible payoff.

**Phase 0 — scaffolding**
- Repo + uv project + pyproject with hackrfpy wired in. `uv build` works.
- `IQSource` protocol; `FileSource` against existing `sample_data`.
- Test harness + synthetic signal generators (the ground-truth helpers).

**Phase 1 — core primitives (the foundation)**
- `filters.py`: design via scipy (`firwin`/`butter`) + **own FIR apply**;
  verified against `scipy.signal.lfilter`.
- `resample.py`: **own** decimation/interpolation; verified against
  `scipy.signal.resample_poly`.
- `spectral.py`: windowing + PSD (numpy FFT, own scaling/averaging) + dB.
- `mixing.py`: frequency translation.
- Each ships with tests asserting correctness vs. scipy/known signals.

**Phase 2 — Example 1: FM receiver** (the satisfying one)
- FileSource (or live) → filter to one station → `fm_demod` → resample to
  48 kHz → `wav_sink`. **You hear a station.** Exercises every Phase-1 piece.

**Phase 3 — Example 2: spectrum analyzer**
- Proper PSD + spectrogram with correct scaling; ties to hackrfpy's
  `relative_power_db` for a dB reference. The "done right" waterfall.

**Phase 4 — Example 3: resampler benchmark** (the cool verification one)
- Benchmark + correctness comparison of sdr_dsp's own resampler vs.
  `scipy.signal.resample_poly`: speed, and numerical agreement. Demonstrates
  the "implement it, then prove it against the oracle" philosophy as a runnable
  artifact.

**Phase 5 — Example 4: OOK/ASK decoder @ 433 MHz** (hardware-testable)
- Envelope detect → threshold → bit timing. **Testable with a key fob** — real
  signal validation, the hackrfpy hardware discipline applied to DSP.

**Phase 6 — Example 5: channelizer** (capstone)
- Pull one narrow channel from a wide capture: filter + decimate + frequency
  shift. Ties filtering + resampling + mixing together.

**Later / not priority:** an example protocol decoder (build on the OOK
front-end). Fun, explicitly deferred.

## 8. Testing approach (carried from hackrfpy)

- **Pure core = trivially testable.** Synthetic signals with known properties
  (a generated tone, a known FM-modulated carrier, a clean OOK burst) feed the
  functions; assert outputs against ground truth the author controls.
- **scipy as oracle.** Own FIR apply asserted == `lfilter`; own resampler
  asserted ≈ `resample_poly` within tolerance. Implement to understand, verify
  against known-good.
- **Synthetic signal generators** live in `tests/helpers/` and double as the
  data source for examples that shouldn't need hardware.
- **Hardware-marked tests** (`@pytest.mark.hardware`, auto-skip without a board)
  for the live `HackRFSource` and the key-fob OOK decode.
- **FileSource + sample_data** means the bulk of development and CI needs no
  board.

## 9. Scope boundary (the "no" list)

Protects the project from sprawl, exactly as hackrfpy's no-list did.

**In scope:** filtering, spectral analysis, resampling, mixing/translation,
analog + simple digital demod, measurement, file save/load (SigMF), classful
pipelines, demos.

**Out of scope:** a runtime/scheduler/flowgraph engine; real-time streaming
framework with backpressure; a GUI builder; a large block/protocol-decoder zoo;
anything whose honest answer is "use GNU Radio for that." One protocol decoder
*example* is a welcome later addition; a decoder *framework* is not.

## 10. The learning arc (this is for the author, first)

Each phase teaches a real DSP concept by building it:
- FM receiver → filtering + demodulation + resampling, end to end.
- Analyzer → windowing, FFT scaling, dB reference.
- Resampler benchmark → polyphase resampling + how to validate an implementation.
- OOK decoder → envelope detection + timing recovery, on a real signal.
- Channelizer → decimation + frequency translation.

By the end: the author has *built* the operations GNU Radio provides as black
boxes — the point being to understand them, with a genuinely reusable library as
the artifact.

## 11. The bridge from hackrfpy (the handoff)

sdr_dsp consumes exactly what hackrfpy was designed to hand off:
- `load_iq` + SigMF sidecars → `FileSource` (offline).
- `HackRF` / `open_receiver` → `HackRFSource` (live).
- `relative_power_db` → the dB reference for the analyzer.
hackrfpy stops at "honest, gain-aware complex64 over time"; sdr_dsp begins there.
The two projects form a clean two-layer stack: device+IQ, then DSP.
```

---

# Expansion Plan (post-v0.1 — completing the DSP surface)

After the initial library + 20 examples + 67 tests, a critical audit identified
real gaps. This section is the roadmap to fill them. Built in verified batches;
tests written alongside, consolidated at the end of each batch.

## Design constraints (carried from the author's review)

- **No silent assumptions in analysis.** AGC/normalization, carrier-offset
  estimation, and any "intelligent" measure-and-tune logic must be
  USER-CONTROLLED and explicit. The library may MEASURE and offer to correct,
  but never auto-applies corrections that change data interpretation. This is
  load-bearing for honest data analysis.
- **Streaming is in scope and important.** Showing signal at capture, during
  demod, and message recovery LIVE is a core goal — it's what proves the
  project isn't a "party trick." Some latency is acceptable; the streaming
  layer ORCHESTRATES the pure-function core, it does not replace it.
- **A representative demod suite, not exhaustive.** Cover ~90% of real signals
  without becoming a modem/FEC library.

## Demodulation suite (target set)

Common demods ranked by real-world prevalence; the chosen set in **bold**:
- **FM** (broadcast, NOAA, ham) — have it.
- **AM** (aircraft, broadcast, shortwave) — have it.
- **OOK/ASK** (key fobs, cheap ISM sensors) — have it.
- **FSK** (2-level: weather stations, TPMS, IoT, pagers) — BUILD. The big gap.
- **SSB** (USB/LSB: ham voice, marine/aviation HF) — BUILD. The analog gap.
- **BPSK** (satellites, PSK31, data links) — BUILD. Gateway to phase mod.
- GFSK/MSK/GMSK (Bluetooth, GSM, AIS) — basic FSK demod recovers these
  adequately for demos; NOTE rather than separate functions.
- QPSK and higher — DEFER (modem territory).

## Batch 1 — foundational primitives
- `normalize` / AGC — explicit, user-controlled modes (peak, rms, none);
  never silently applied.
- `to_db` / `from_db` — consistent epsilon, one canonical place (replaces the
  scattered 1e-9/1e-12/1e-20 hand-rolled conversions).
- `instantaneous_frequency` / `instantaneous_phase` — refactored OUT of
  fm_demod so FM and FSK both build on the public primitive.
- `correlate` / `convolve` — wrap the np.correlate conjugation footgun once,
  correctly (the bug already hit once in matched_filter).

## Batch 2 — demod suite
- `fsk_demod` (2-level, on instantaneous_frequency)
- `ssb_demod` (USB/LSB, via analytic signal / sideband selection)
- `bpsk_demod` (with documented limits — no carrier recovery yet)

## Batch 3 — detection / measurement
- `find_bursts` (energy detector: threshold envelope -> start/stop indices)
- `estimate_cfo` / carrier-offset MEASUREMENT (measures, does NOT auto-apply,
  per the no-assumptions constraint)

## Batch 4 — streaming / real-time (designed deliberately, not rushed)
- A block-processing pipeline that orchestrates the pure-function core, so a
  live source -> demod -> message can be shown updating together.
- Architecture TBD with the author (Pipeline class vs generator chain vs
  live-display-focused). Must respect the pure core.

## Batch 5 — edge-of-scope demonstrations (the author wants these as demos)
- SSB doubles as the analytic-signal demonstration.
- Carrier recovery as a BPSK example (not a high-priority core addition).
- Advanced multirate as a channelizer extension.

## Outstanding / todo (lower priority)
- More datatype loaders (ci16 etc.) if real captures need them.
- The folded-but-unbuilt niche examples (constellation, two-tone intermod,
  snr_vs_gain) — spin up if a lesson needs them.
- Release polish: README de-draft, screenshots, uv build verification,
  .gitignore/license hygiene.
