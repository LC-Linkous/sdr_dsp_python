# sample data

Captures for trying the library and downstream processing without owning a
board.

The data in the /hackrf_one directory is collected from a hackRF One SDR for three FM radio stations




`fm_2Msps.iq` is a REAL over-the-air broadcast FM capture: 103.7 MHz,
recorded on a HackRF One at 8 Msps (hackrfpy `tests/fm_reference/`, verified
by ear), imported at 2 Msps by `tools/import_reference_capture.py`. The
import validates before and after (19 kHz stereo pilot +27.3 dB, channel
+20.9 dB above the noise floor) and the sidecar carries full provenance:
source file and sha256, original rate, gains, measured pilot, and the level
rescale applied at requantization.

For a fully-known fixture (exact deviation, pilot level, SNR, CFO), generate
the synthetic companion instead: `tools/make_synthetic_sample.py` writes
`fm_synthetic_2Msps.iq` with every parameter stated in its docstring.

- sample rate: 2 Msps

Each `.iq` is interleaved int8 I/Q (HackRF native) with a `.sigmf-meta` sidecar
describing frequency, rate, and gains. Load with:

```python
from sdr_dsp.io import load_iq, read_meta

iq, meta = load_iq("fm_2Msps.iq")   # complex64, normalized to +/-1
meta = read_meta("fm_2Msps.iq")     # sidecar only, no sample data read
```

For a quick look at what's actually in a capture:

```bash
uv run python examples/inspect_capture.py sample_data/fm_2Msps.iq
```

## Files

- `fm_2Msps.iq` — wideband FM, centered 98.0 MHz. **See the note below.**
- `fm_sweep.csv` — swept power measurements across the FM band.

## Known problem: `fm_2Msps.iq` contains no station

This capture is at the noise floor and needs to be re-recorded. It uses only
about 3 of 127 available ADC counts — every sample in the file is one of
`{-2, -1, 0, 1, 2}` — and its spectrum is flat to within 2 dB across the full
2 MHz, with no station hump anywhere in the span.

Demodulating it produces the FM *noise triangle*: output power rising steeply
with audio frequency (roughly 47 dB from 100 Hz to 22 kHz) and no 19 kHz stereo
pilot. That is the signature of a phase discriminator running on noise with no
carrier present. It sounds like hiss, not program material.

This is a property of the recording, not of the DSP — `examples/fm_receiver.py`
recovers audio correctly from a capture that actually contains a station. The
receiver example now runs a health check and warns before writing a WAV that
will only contain noise.

Until this file is replaced, `examples/fm_receiver.py` will run to completion
and write a valid WAV, but the audio will be hiss. To re-record: connect an
antenna, confirm the tuned frequency has a strong local station, and raise
LNA/VGA gain until the capture uses a healthy fraction of the ADC range
(a peak somewhere in the 60–110 count region is a reasonable target — high
enough to be well clear of the noise floor, low enough to avoid clipping).
