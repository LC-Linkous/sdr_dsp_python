# Examples

A catalog of every script in `examples/`, grouped by what it teaches or does.
Most run on a file or synthesize their own signal (no hardware). The
**Needs** column flags anything beyond the library's numpy + scipy:

- **plot** — matplotlib (`uv sync --extra plotting`)
- **audio** — sounddevice (`uv sync --extra audio`)
- **HW** — a live HackRF + the hackrf-tools binaries (`uv sync --extra examples-hackrf`)
- *(blank)* — runs on the library alone, on a file or a synthesized signal

To run any example: `uv run python examples/<name>.py` (add `--help` for options).
The fast setup for everything: `uv sync --extra examples`.

---

## Getting started / data inspection

| Example | What it does | Needs |
|---|---|---|
| `inspect_capture.py` | "What did I just capture?" Prints datatype, rate, center frequency, duration, power, DC offset, and clipping for any SigMF file. The first tool to reach for when something looks wrong. | |
| `iq_basics.py` | "What *is* IQ data?" Visualizes a signal three ways — the complex plane, magnitude, and phase over time — to build intuition for quadrature samples. | plot |
| `signal_survey.py` | Measures power, occupied bandwidth, and SNR across one or more captures. The triage step: what's here and how strong? | |

## DSP fundamentals (teaching)

| Example | What it does | Needs |
|---|---|---|
| `windowing_demo.py` | Why FFT windows matter: the same tone through rectangular / Hann / Blackman windows, showing spectral leakage shrink. | plot |
| `nyquist_aliasing_demo.py` | See a tone alias to the wrong frequency when undersampled — the sampling theorem made visual. | plot |
| `dc_offset_demo.py` | The center DC spike (LO leakage) every direct-conversion SDR produces, and the one-line fix. | plot |
| `filter_explorer.py` | "What does a filter actually do?" Shows a filter's frequency response next to a real signal's spectrum before and after. | plot |
| `decimation_stages.py` | Why multi-stage decimation beats one big step — same result, less work, shown with timing. | plot |
| `spectrogram_chirp.py` | A frequency sweep drawn as a diagonal line in time-frequency. The satisfying one. | plot |
| `matched_filter_demo.py` | Pull a known pattern out of heavy noise by correlation — the optimal detector, made visible. | plot |
| `resampler_benchmark.py` | The library's resampler vs scipy: speed and numerical agreement. Demonstrates the "implement it, then verify against the oracle" philosophy. | plot |

## Spectral analysis

| Example | What it does | Needs |
|---|---|---|
| `spectrum_analyzer.py` | A proper PSD display with noise-floor estimate and peak markers. | plot |
| `waterfall.py` | Offline spectrogram (time-frequency heatmap) of a recording. | plot |

## Filtering and channelization

| Example | What it does | Needs |
|---|---|---|
| `channelizer.py` | Pull one narrow channel out of a wide capture: tune → filter → decimate, tied together. | plot |

## Analog demodulation

| Example | What it does | Needs |
|---|---|---|
| `fm_receiver.py` | FM broadcast receiver: file → filter → FM demod → resample → WAV. The end-to-end demo. | |
| `am_receiver.py` | AM demodulation (aircraft, broadcast, shortwave): file → audio WAV. | |
| `ssb_receiver.py` | Single-sideband (USB/LSB) voice to audio — ham/marine/aviation HF. Try the wrong sideband to hear why it matters. | |
| `dsb_sc_demo.py` | Double-sideband suppressed-carrier demod; shows the carrier suppressed vs plain AM. | |
| `dsb_sc_receiver.py` | DSB-SC receiver variant (earlier version of the above). | |
| `cw_decoder.py` | Decode Morse code (CW) from a keyed tone — envelope, timing, and a Morse lookup. | |

## Digital demodulation — amplitude & frequency

| Example | What it does | Needs |
|---|---|---|
| `ook_decoder.py` | Decode on-off keying (e.g. a 433 MHz key fob): envelope → threshold → timing → bits. | |
| `nask_decoder.py` | Multi-level amplitude-shift keying (4-ASK): more bits per symbol, tighter spacing. | |
| `ask_decoder.py` | M-ASK decoder (earlier version of the N-ASK example). | |
| `fsk_decoder.py` | Decode FSK — the ISM-band workhorse (weather stations, TPMS, sensors, pagers). Handles 2-FSK and 4-FSK. | |

## Digital demodulation — phase & QAM

| Example | What it does | Needs |
|---|---|---|
| `psk_decoder.py` | Coherent BPSK / QPSK / 8-PSK end to end: carrier recovery → timing recovery → bit decisions. | |
| `differential_psk_demo.py` | DBPSK / DQPSK — phase decoding with **no** carrier recovery; proves a constant phase offset cancels out. | |
| `qam_decoder.py` | QAM-16 (4 bits/symbol), the amplitude+phase modulation. Demonstrable on clean/strong signals. | plot |
| `constellation_recovery.py` | Watch carrier + timing recovery clean a scrambled constellation into readable clusters, with the loop error converging. The visual payoff of the recovery layer. | plot |

## Spread spectrum

| Example | What it does | Needs |
|---|---|---|
| `dsss_demo.py` | Despread a direct-sequence signal with a known code — pulls the signal out from *under* the noise floor (processing gain). | |
| `fhss_visualizer.py` | See a frequency-hopping signal jump channels in a spectrogram, with the detected hop track overlaid. | plot |

## Measurement & detection

| Example | What it does | Needs |
|---|---|---|
| `burst_detector.py` | Find packet bursts in a mostly-silent capture: energy detection with gap-merging and minimum-length filtering. | |
| `cfo_demo.py` | Measure a carrier frequency offset — and only correct it if you ask. Embodies the measure-don't-auto-apply principle. | |

## Streaming

| Example | What it does | Needs |
|---|---|---|
| `live_pipeline_fm.py` | Streaming FM demod with a live level-meter tap — the "watch it happen" demo. Runs on a file or a HackRF. | plot |
| `streaming_options.py` | Three streaming architectures (Pipeline class, generator chain, live-display) on the same task, compared in runnable code. | |

## Live hardware (HackRF)

| Example | What it does | Needs |
|---|---|---|
| `hackrf_capture.py` | The capture helper — implements the `IQSource` protocol from outside the library. The reference for wiring in any SDR. Not run directly; imported by the live examples. | HW |
| `live_433_capture.py` | Capture a 433 MHz burst (e.g. a key fob) and save it for the OOK/FSK decoders to run on offline. | HW |
| `live_fm_listen.py` | Tune an FM station and play it through your speakers in real time. | HW, audio |
| `live_spectrum.py` | Live spectrum display from the HackRF — "what's on this frequency right now?" | HW, plot |

---

## Notes

- **Two pairs overlap by design history:** `nask_decoder.py` / `ask_decoder.py`
  (both M-ASK) and `dsb_sc_demo.py` / `dsb_sc_receiver.py` (both DSB-SC) are
  alternate takes from different build rounds. Either works; the first-named is
  the more recent.
- **Every demodulation function** in the library has both a unit test and at
  least one example here. See `MODULATIONS.md` for the function-by-function
  status table.
- **Capture targets for the live examples:** FM broadcast (88–108 MHz) → FM
  receiver; aircraft band (~120 MHz AM) → AM receiver; ham HF → SSB/CW; 433/915
  MHz ISM → OOK/FSK/burst detection; 2.4 GHz → FHSS visualizer. The `live_*`
  capture scripts save SigMF files so the offline decoders can run on real data
  repeatedly.
