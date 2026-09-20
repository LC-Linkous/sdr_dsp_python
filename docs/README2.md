# sdr_dsp

A device-agnostic digital signal processing library for software-defined
radio. It operates on complex baseband IQ (`numpy.complex64`) and provides the
building blocks to receive, analyze, demodulate, modulate, frame, and exchange
data over a radio link. It is a **library, not a framework** — no GUI, no
flowgraph runtime; you call functions and compose them. The DSP core imports no
specific radio; device adapters (HackRF, files) live at the edges.

Developed and tested against a **HackRF One**, but the core processes IQ
regardless of source. See `docs/HARDWARE.md` for what that hardware can and
can't do, and why it's a good development platform.

## Status

Pre-release; there is no Version 1 yet, and it is heavily under development.
Some features have hooks that aren't implemented. What is proven where:

- **FM receive** — validated on real, ear-verified over-the-air captures.
- **Other demodulators (AM/SSB/CW, OOK/ASK, FSK, the PSK family, QAM-16)** —
  verified against synthetic ground truth; see `docs/MODULATIONS.md` for the
  honest per-scheme status (Supported / Demonstrable / Visualize-only).
- **Transmit and the ARQ link protocol** — proven in software (closed-loop
  simulation); the real-radio transmit path is not yet validated. See §12 of
  `docs/sdr_dsp_REFERENCE.md`.

## Start here

- `docs/sdr_dsp_REFERENCE.md` — the comprehensive technical reference
  (architecture, module map, usage, extension guide, limitations).
- `docs/EXAMPLES.md` — a catalog of every runnable script in `examples/`.
- `docs/MODULATIONS.md` — what's supported and what "supported" means.
- `docs/HARDWARE.md`, `docs/DC_SPIKE.md` — device behavior and the offset-tuning
  recipe every direct-conversion capture needs.
- `docs/LOG.md` — dated development log of major changes and fixes.

Quick example:

```python
from sdr_dsp import tune_to_baseband, design_lowpass, fir_apply, fm_demod
from sdr_dsp.io import load_iq

iq, meta = load_iq("sample_data/fm_2Msps.iq")
fs = meta["global"]["core:sample_rate"]
# (already at baseband here; tune_to_baseband handles off-center captures)
audio = fm_demod(fir_apply(iq, design_lowpass(100e3, fs)),
                 deviation_hz=75e3, sample_rate=fs)
```

Setup uses [uv](https://docs.astral.sh/uv/): `uv sync` for the library and
tests, `uv sync --extra examples` for everything the examples can use.

---

*The documentation is being AI-summarized to fix spelling and make the
development easier to follow. All mistakes are human, and will likely take a
revision or two to fix experimentally.*