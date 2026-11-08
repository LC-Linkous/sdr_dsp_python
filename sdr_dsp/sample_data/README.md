# sample data

Captures for trying the library and downstream processing without owning a
board.

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

- `fm_2Msps.iq` + `fm_2Msps.sigmf-meta` -- wideband FM, 103.7 MHz, 2 Msps,
  the real capture described above. The examples default to it.
- `fm_sweep.csv` -- swept power measurements across the FM band.
- `hackrf_one/` -- the raw 8 Msps source captures; see below.

## Large raw captures (`hackrf_one/`)

`hackrf_one/` is the home of the raw 8 Msps ci8 captures the sample above was
imported from -- 103.7 MHz and 98.1 MHz, 5 s each, 80 MB each, byte-identical
to hackrfpy's `tests/fm_reference/`. The `.iq` payloads are **not in git**
(`.gitignore` excludes them); they are distributed as GitHub release assets.
What IS in git is everything needed to use a downloaded copy:

| file | what it is |
|---|---|
| `fm_98.1MHz_8Msps.sigmf-meta`, `fm_103.7MHz_8Msps.sigmf-meta` | SigMF sidecars: datatype, rate, centre, gains, sha256 of the payload. `FileSource` and `load_iq` read these. |
| `*.report.json` | the `capture_health` report written at capture time |

To use them: download the `.iq` files from the release assets into this
directory, next to their sidecars, and check them:

```bash
sha256sum fm_98.1MHz_8Msps.iq      # must match sdr_dsp:sha256 in the sidecar
uv run python examples/inspect_capture.py sample_data/hackrf_one/fm_98.1MHz_8Msps.iq
```

Nothing in the examples or the default test run needs them; only
`fm_2Msps.iq` is required.

Download: `<release asset URL to be added>`

Captures are promoted into this directory only through
`tools/promote_to_sample_data.py`, which re-runs `capture_health` after
copying and records the result in `manifest.json` (written on first
promotion; the shipped capture came in through `import_reference_capture.py`,
whose provenance is in the sidecar instead). A file that is here has passed
its own declared expectation. If a capture ever starts sounding like
hiss, run `inspect_capture.py` on it first: a flat spectrum using only a few
ADC counts is a property of the recording, not of the DSP.
