# sdr_dsp

A personal, fully-functional DSP library for software-defined radio, written in
Python. It consumes IQ samples — from a file or a live SDR — and turns them into
meaning: filtered channels, spectra, demodulated audio, decoded signals.

It is a **library, not a framework.** You import functions and classes and
orchestrate the pipeline yourself in plain Python (or with the included
`Pipeline` helper). There is no required runtime, scheduler, or GUI. It is not a
GNU Radio competitor — GNU Radio remains the right tool for large real-time
flowgraphs; `sdr_dsp` is for direct, scriptable, inspectable DSP in Python.

## Design philosophy

**The radio DSP is the library's own code.** Filtering (application),
demodulation, resampling, mixing, measurement, and the recovery loops are
implemented here. `scipy.signal` is used only to *design* filter coefficients
(the solved, stable math), `numpy.fft` provides the FFT, and scipy also serves
as a **test oracle** — the library's own implementations are verified against
scipy's known-good results. scipy is a design-time tool, not a runtime crutch.

**Device-agnostic by structure.** The core DSP operates on `complex64` arrays
and knows nothing about any device. IQ arrives through a *source* that satisfies
the `IQSource` protocol — `ArraySource` and `FileSource` ship with the library
(no device dependency); a device source lives in *your* application/example code
(see `examples/hackrf_capture.py` for a HackRF reference). The library provides
the hooks and the protocol; you provide the hardware.

**Honest, user-controlled DSP.** Nothing is silently assumed. Normalization,
carrier-offset correction, and the recovery loops are explicit and
user-driven — the library will *measure* an offset but never auto-applies a
correction that changes how you'd interpret the data. The recovery loops expose
their convergence (per-sample error, lock state, optional CSV) so you can see
whether they worked rather than trust that they did.

## What it does

- **Filtering** — FIR design (via scipy) and application (ours): lowpass,
  bandpass, highpass.
- **Resampling** — our polyphase resampler, decimation, interpolation
  (verified against scipy).
- **Spectral** — PSD (Welch) and spectrogram, our scaling on numpy's FFT.
- **Mixing** — frequency translation, tune-to-baseband, DC removal.
- **Measurement** — power, SNR, occupied bandwidth, burst detection, carrier
  offset estimation.
- **Detection** — matched filter, correlation/convolution (conjugation handled).
- **Recovery** — composable carrier recovery (Costas / decision-directed) and
  symbol timing recovery (Gardner / early-late / Mueller-Müller), with
  diagnostics.
- **Demodulation** — a broad suite (see `MODULATIONS.md`): AM, FM, SSB, DSB-SC,
  CW/Morse, OOK, N-ASK, 2-FSK, N-FSK/CPFSK, BPSK, QPSK, 8-PSK, DBPSK, DQPSK,
  QAM-16, DSSS, FHSS.
- **Streaming** — a `Pipeline` that orchestrates the pure core block-by-block,
  with taps for live display.

## File tree

```
sdr_dsp/
├── pyproject.toml
├── README.md
├── MODULATIONS.md            # the modulation support table (honest status)
├── HARDWARE.md               # HackRF context + cross-SDR comparison
├── src/sdr_dsp/
│   ├── core/                 # PURE DSP — arrays in, arrays out
│   │   ├── filters.py        # scipy designs taps; we apply them
│   │   ├── resample.py       # our polyphase resampler
│   │   ├── spectral.py       # PSD, spectrogram
│   │   ├── mixing.py         # frequency translation, DC removal
│   │   ├── measure.py        # power, SNR, bandwidth, bursts, CFO
│   │   ├── detect.py         # matched filter, correlate/convolve
│   │   ├── sync.py           # carrier + symbol-timing recovery loops
│   │   ├── util.py           # dB conversion, explicit normalization
│   │   └── demod/            # demodulation, by family
│   │       ├── phase.py      # instantaneous phase/frequency
│   │       ├── analog.py     # FM, AM, SSB, DSB-SC, CW, de-emphasis
│   │       ├── ask.py        # OOK, N-ASK
│   │       ├── fsk.py        # 2-FSK, N-FSK/CPFSK
│   │       ├── psk.py        # BPSK, QPSK, 8-PSK, DBPSK, DQPSK
│   │       ├── qam.py        # QAM-16
│   │       ├── spread.py     # DSSS despread, FHSS hop detect
│   │       └── timing.py     # edges, symbol-rate, slicing
│   ├── sources/              # IQSource protocol + ArraySource + FileSource
│   ├── sinks/                # WAV, processed-IQ, plot helpers
│   ├── stream/               # the Pipeline orchestration layer
│   └── io/sigmf.py           # read ci8 captures, write cf32_le output
├── examples/                 # 38 runnable examples (one per demod + more)
├── tests/                    # 149 tests; scipy-oracle + synthetic ground truth
└── sample_data/              # SigMF recordings for the examples
```

## Setup

This project uses [uv](https://docs.astral.sh/uv/). From the project directory:

```bash
uv sync                  # create the venv, install numpy + scipy + dev tools
uv run pytest -q         # run the test suite (expect: 149 passed)
uv build                 # build the wheel + sdist into dist/
```

The library installs on **numpy + scipy alone** — no SDR hardware or device
library required. Everything in `src/sdr_dsp/` works on arrays and files.

### Example dependencies

The example scripts may need more than the library does. The fast path to run
any example:

```bash
uv sync --extra examples     # matplotlib + sounddevice + hackrfpy
```

Or install just what one example needs via the granular extras, all defined in
`pyproject.toml` (none are library dependencies):

| Extra | Installs | For |
|---|---|---|
| `plotting` | matplotlib | analyzer, spectrogram, constellation, filter demos |
| `audio` | sounddevice | live audio playback (`live_fm_listen.py`) |
| `examples-hackrf` | hackrfpy | the live-capture examples |
| `examples` | all of the above | one-shot setup for every example |

Live capture also needs the `hackrf-tools` binaries at the OS level (see the
[hackrfpy](https://pypi.org/project/hackrfpy/) docs); file-based examples need
none of that.

## Quick start: hear a station

```bash
uv run python examples/fm_receiver.py sample_data/fm_2Msps.iq --out station.wav
```

This loads a wideband-FM recording, filters to the station, FM-demodulates,
resamples to 48 kHz, and writes a playable WAV. If the station isn't at the
capture's center, tune with `--tune <offset_hz>`.

## Using the library

```python
from sdr_dsp.sources import FileSource
from sdr_dsp.core import design_lowpass, fir_apply, fm_demod, resample_poly

src = FileSource("sample_data/fm_2Msps.iq")      # reads rate/freq from SigMF
taps = design_lowpass(100_000, src.sample_rate)  # scipy designs the taps
iq = fir_apply(src.iq, taps)                     # our code applies them
audio = fm_demod(iq, deviation_hz=75_000, sample_rate=src.sample_rate)
audio = resample_poly(audio, 48_000 // 16, int(src.sample_rate) // 16)
```

A coherent demod, composing the recovery primitives explicitly:

```python
from sdr_dsp.core import carrier_recovery, symbol_sync, qpsk_demod

corrected = carrier_recovery(iq, method="costas", order=4)   # remove carrier offset
symbols = symbol_sync(corrected, samples_per_symbol=4)       # recover timing
bits, _ = qpsk_demod(symbols)                                # decide bits
```

The same chain, streamed block-by-block with a live tap:

```python
from sdr_dsp import Pipeline

pipe = (Pipeline(source)
        .add(lambda b: fir_apply(b, taps), "filter")
        .add(lambda b: fm_demod(b, 75_000, fs), "demod")
        .tap(lambda b: meter.update(b)))         # live display, flow unchanged
audio = pipe.run()
```

## Examples

38 runnable examples in `examples/`, spanning data inspection, the teaching
fundamentals (IQ basics, windowing, aliasing, filtering), every demodulation in
the suite, the recovery loops, spectral analysis, and live streaming. Most run
on files or synthesize their own signal; the `live_*` examples use a HackRF.
Every demodulation function has both a unit test and a runnable example.

## Documentation

- **`MODULATIONS.md`** — the modulation support table, with honest status
  (Supported / Demonstrable / Visualize-only / Out-of-scope) and the recovery
  layer.
- **`HARDWARE.md`** — the HackRF One context: what its 8-bit ADC means per
  modulation, and a cross-SDR comparison so expectations are calibrated.

## Testing approach

Two disciplines, carried from the sibling `hackrfpy` project:
- **scipy as oracle** — own implementations (FIR application, resampler) are
  asserted equal to scipy's within numerical tolerance.
- **synthetic ground truth** — signals are generated with known properties (a
  tone at a known frequency, an FM message, an OOK/FSK/PSK bit pattern) and the
  DSP is checked against what it should recover.

All 149 tests run without hardware. Hardware validation (live capture across the
bands) is driven manually with a real device.
