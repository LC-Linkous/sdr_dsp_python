# sdr_dsp

A device-agnostic digital signal processing library for software-defined
radio. It operates on complex baseband IQ (`numpy.complex64`) and provides the
building blocks to receive, analyze, demodulate, modulate, frame, and exchange
data over a radio link. It is a **library, not a framework** -- no GUI, no
flowgraph runtime; you call functions and compose them. The DSP core imports no
specific radio; device adapters (HackRF, files) live at the edges.

Developed and tested against a **HackRF One**, but the core processes IQ
regardless of source. See [`docs/HARDWARE.md`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/docs/HARDWARE.md)
for what that hardware can and can't do. For what is validated on real radio
versus in software only, see the Status section of the
[repository README](https://github.com/LC-Linkous/sdr_dsp_python#status).

## Install

The library itself depends on **numpy + scipy only**; it installs and runs
with no SDR hardware or device library. The extras are for the example scripts.
(Until the first PyPI release is cut, install from a clone -- see the uv
section below; the `pip install` lines are what that release will look like.)

```bash
pip install sdr-dsp                      # core
pip install "sdr-dsp[plotting]"          # + matplotlib: analyzer/spectrogram/teaching examples
pip install "sdr-dsp[audio]"             # + sounddevice: live FM playback
pip install "sdr-dsp[examples-hackrf]"   # + hackrfpy: live capture/transmit examples
pip install "sdr-dsp[examples]"          # everything above in one install
```

With [uv](https://docs.astral.sh/uv/), from a clone, run everything from the
`sdr_dsp/` directory (the one with `pyproject.toml`, one level below the repo
root):

```bash
cd sdr_dsp
uv sync                             # library + test suite + dev tools (editable)
uv sync --extra plotting            # same extras as above: plotting, audio,
uv sync --extra examples            #   examples-hackrf, examples
uv run pytest                       # hardware tests are deselected by default
uv run python examples/fm_receiver.py sample_data/fm_2Msps.iq --out station.wav
```

Live capture and transmit additionally need the `hackrf-tools` binaries
installed at the OS level; file-based examples need none of that.

## Quickstart: hear a station

```python
from sdr_dsp import tune_to_baseband, design_lowpass, fir_apply, fm_demod
from sdr_dsp.io import load_iq

iq, meta = load_iq("sample_data/fm_2Msps.iq")
fs = meta["global"]["core:sample_rate"]
# A direct-conversion SDR puts a DC spike at its tuned frequency, so real
# captures are often made with the radio tuned a little to one side and the
# station brought back to 0 Hz in software. This capture is centered, so the
# offset is 0; see docs/DC_SPIKE.md for the recipe.
iq = tune_to_baseband(iq, offset_hz=0.0, sample_rate=fs)
audio = fm_demod(fir_apply(iq, design_lowpass(100e3, fs)),
                 deviation_hz=75e3, sample_rate=fs)
```

`examples/fm_receiver.py` is the same thing with decimation, de-emphasis, a
15 kHz audio lowpass, optional stereo, and a WAV writer around it.

## What's in the box

- `sdr_dsp.core` -- the DSP itself: filter design/apply, polyphase resampling,
  mixing, spectral estimates, measurement, AGC, sync, framing with CRC, the
  demodulators and their matching modulators, the channel simulator, and the
  gain-search engine. Pure arrays in, arrays out.
- `sdr_dsp.stream` -- `Pipeline` plus block-wise `StreamingFIR`,
  `StreamingResampler` and related stages for chunked/live processing.
- `sdr_dsp.sources` -- where IQ comes from: the `IQSource` protocol,
  `ArraySource`, `FileSource` (SigMF), and the file-path capture probe.
- `sdr_dsp.sinks` -- where results go: `write_wav`, `write_iq`, plot sinks,
  and the `TXSink` protocol with a `LoopbackSink` for tests.
- `sdr_dsp.io` -- SigMF read/write (`load_iq`, `read_meta`), ci8 captures in,
  cf32 out, annotations.
- `sdr_dsp.link` -- the framed link protocol: windowed ARQ engine and
  pluggable drivers, proven station-to-station in loopback.

## Documentation

- [`docs/README.md`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/docs/README.md) -- index of everything below.
- [`docs/MODULATIONS.md`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/docs/MODULATIONS.md) -- what's supported and what "supported" means per scheme.
- [`docs/EXAMPLES.md`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/docs/EXAMPLES.md) -- catalog of every runnable script in [`examples/`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/sdr_dsp/examples).
- [`docs/sdr_dsp_REFERENCE.md`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/docs/sdr_dsp_REFERENCE.md) -- the comprehensive technical reference (architecture, module map, usage, limitations).
- [`docs/DC_SPIKE.md`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/docs/DC_SPIKE.md) -- the offset-tuning recipe every direct-conversion capture needs.
- [`docs/ADDING_AN_SDR.md`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/docs/ADDING_AN_SDR.md) -- the `IQSource`/`TXSink` seam and how to port to another radio.
- [`CHANGELOG.md`](https://github.com/LC-Linkous/sdr_dsp_python/blob/main/CHANGELOG.md) -- user-facing changes by version.

---

*The documentation is being AI-summarized to fix spelling and make the
development easier to follow. All mistakes are human, and will likely take a
revision or two to fix experimentally.*
