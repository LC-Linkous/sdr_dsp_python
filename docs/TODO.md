# TODO

Open work, roughly in priority order. Items the LOG defers "to TODO" land
here. Strike through or delete when done; the LOG records what happened.
This is the only TODO; the September 2026 session journal that used to live
at the repo root is condensed into `LOG.md` (entry "Sept-Oct 2026 sessions").

## Hardware (the gate everything else waits on)

- [ ] **Run `uv run pytest -m hardware tests/test_hardware_rx_smoke.py -v`
      on the bench.** Written 2026-10-24 without a board; the first run is
      its validation. Set `SDR_DSP_TEST_FM_HZ` to a local station.
- [ ] **Wired one-way TX bench test** of the FSK packet chain
      (`examples/two_station_link.py`). Everything digital, TX and protocol
      is software-only until this runs (REFERENCE §12).
- [ ] **Two-SDR bench test** (needs a SECOND HackRF): one board runs
      `examples/transmit_examples.py`, the other captures with the RX chain,
      and the matching demod decodes it -- closing `demod(modulate(x)) == x`
      over a REAL channel instead of `apply_channel()`. This is the
      real-world validation the digital demods still lack (only FM is
      hardware-verified). Gate it behind the `hardware` marker. It is the
      only step blocked on the second board; everything no-hardware is done.
- [ ] **Windows stdout-pipe corruption at 8 Msps** (`capture_array` /
      `open_receiver`; LOG 2026-09-20). Unresolved upstream in hackrfpy 1.1.0.
      Suspect throughput rather than framing: watch `hackrf_transfer`'s
      dropped-sample stats on stderr, try a large `read_samples`, and A/B
      against the file path again. Until it's settled, every measurement
      probe stays on `sources/probe.py`; `HackRFCapture` (live adapter)
      still streams through the pipe. Also audit the live-listen path
      (`PersistentReceiver`): at 2 Msps it may keep up, but listen for
      glitches during `live_fm_listen.py`.
- [ ] Re-verify `HackRFCapture.blocks()` live once hackrfpy ships the
      `PersistentReceiver.blocks()` odd-byte fix (I/Q swap after an odd
      chunk; found 2026-10-24).
- [ ] CFO-ppm capture experiment: the cheap go/no-go for whether real
      same-model devices separate on the fingerprint features (LOG 07-11).
- [ ] Recapture a native 2 Msps reference and re-import it with
      `tools/import_reference_capture.py`. The shipped sample is decimated
      from 8 Msps -- fine, but a native capture removes one processing step
      from the provenance chain. Candidate: dev_data's `fm_reference_long`
      (106.5 MHz, pilot +28.7 dB) if it is still on disk (not shipped).
- [ ] Field-verify a quiet frequency at this location and re-run
      `tools/collect_dev_data.py --only 5` with it as `--quiet`. The code side
      is done (preflight hunts below 87.5 MHz; a below-band carrier is blocked
      as a quiet pick), but 87.7 MHz turned out to be occupied here and
      in-band 91.5 failed too.
- [ ] Extend real-data testing beyond FM: as digital captures (OOK/FSK/PSK)
      are collected, add e2e tests like `test_real_demod_e2e.py` that decode
      real captures with known ground-truth bits, not just synthesis.

## Library

- [x] Lift the HackRF three-knob gain model out of `core/gain_search.py`
      into a `GainModel` so `search_gain` can drive another radio
      (`search_gain_model`, `hackrf_gain_model`, `rtlsdr_gain_model`; patch
      0022, 0.2.0). Still open from it: a probe layer that is not written
      against hackrfpy's `capture(...)` (`sources/probe.py`).
- [ ] RDS decoder at 57 kHz -- the next no-hardware feature. Builds on the
      stereo decoder's pilot-lock + coherent-detection machinery. The real
      103.7 capture carries actual RDS frames; the synthetic sample's 57 kHz
      carrier is RDS-*like* (random bits, no group framing) and is NOT
      decodable, so extend `tools/make_synthetic_sample.py` with real RDS
      group encoding first (so there is synthetic ground truth), then
      validate against the real capture's frames.
- [ ] RTL-SDR adapter (`examples/rtlsdr_capture.py`): the first REAL
      non-HackRF SDR, to exercise the porting kit (`docs/ADDING_AN_SDR.md`,
      `examples/sdr_adapter_template.py`, `test_modularity.py`). RX-only,
      cu8 -- exactly what the template models. The adapter and its cu8
      normalization are offline-testable now; only live verification waits
      on the dongle (none on hand). Fills the RTL-SDR porting-record row.
- [ ] Give `live_fm_listen.py`'s `FMStream` over to the `Streaming*` stages
      (`stream/stages.py`) now that they exist; it should become a thin
      composition, and the stereo path can then go live too.
- [ ] Meaningful impairment round-trip demo: `apply_device_impairments` ->
      CFO/carrier recovery -> `decide_symbols` ->
      `estimate_phase_noise_variance` against the reference. A naive version
      gave nonsense (CFO rotates the constellation; the phase-noise variance
      needs the reference), so this is a feature, not cleanup.
- [ ] F3 transient features (fingerprint plan) -- deferred to the hardware era.

## Cross-repo (hackrfpy)

- [ ] File the `capture_array` Windows bug upstream with the A/B repro from
      LOG 2026-09-20 (pipe vs file vs reference, same board/antenna/station,
      minutes apart). Also: interrupting `capture_array` mid-stream can
      orphan `hackrf_transfer.exe`, which deadlocks the next capture.
      Suggested upstream fixes: a Windows warning or file-backed fallback,
      and orphan reaping on interrupt.
- [ ] Check how hackrfpy's `pilot_snr_db` is computed: if it discriminates
      without channelizing first, port the fix from
      `sdr_dsp.core.measure.fm_pilot_excess_db` (unchannelized, a real 8 Msps
      capture measured +1.8 dB vs +17 to +27 dB channelized).
- [ ] Consider preflight-style pilot scoring in hackrfpy's
      `scan_then_capture.py` so both repos pick stations the same way.

## Tooling / typing

- [x] mypy strict + `py.typed` honesty: every function in the package is
      annotated and `disallow_untyped_defs` / `disallow_incomplete_defs` are
      on for the whole package in `pyproject.toml` (0.3.0). Loose "array-like"
      inputs are `Any` on purpose; outputs are `np.ndarray`. Remaining
      strictness (`warn_return_any`, `no_implicit_reexport`) is not planned.
- [x] Line-ending normalization: the index was already LF-clean
      (`git ls-files --eol`); CI now guards it. No renormalize commit needed.

## Docs

- [ ] Write up the `--calibrate` dry run
      (`uv run python tools/collect_dev_data.py --station 106.5e6 --calibrate`):
      probe walk + chosen gain + measured pilot, records nothing. Show a real
      transcript -- the front-end sweep at vga=0, amp on/off, the clip backoff
      at vga=20->2, the final "reference: lna=40 vga=2 amp=on ... pilot" line
      -- with a short read of what each stage proves.
- [ ] Describe the collection workflow (preflight -> calibrate -> collect),
      amp semantics in `search_gain`, and what the pilot metric means and its
      ~6 dB threshold, somewhere a user will find it (HARDWARE.md or a new
      note under `notes/`).

## Repo hygiene

- [ ] `sample_data/hackrf_one/*.iq` is ~150 MB of raw IQ committed directly,
      byte-identical (sha256) to `hackrfpy/tests/fm_reference/`. Move to a
      GitHub release asset (or Git LFS) and fetch on demand; each file is
      ~80 MB, near GitHub's 100 MB hard limit, and `.git` is already large.
- [ ] LOG entries dated 2026-09-21..23 describe work that is in commits
      stamped 2026-09-20 (the LOG was written ahead of the commits). Note
      it in the LOG header or accept the drift.

## Known limitations / warnings

Latent problems and unverified assumptions. Not tasks so much as things to
keep in view.

- **Verify state against YOUR tree, never a reported pass count.** A stale
  editable install once read as green, and patches were once assumed applied
  when they had not been shared. Run the suite locally and trust that.
- **Only FM is hardware-validated.** AM/SSB/CW, OOK/ASK, all FSK, the PSK
  family and QAM-16 are verified against synthesis only. MODULATIONS.md's
  status words encode this; the two-SDR bench test is what closes it.
- **The gain-search strategy is validated against a MODEL** plus one real
  station, not across bands/antennas. It assumes SNR improves with amp+LNA
  and not VGA, which is right for the HackRF in principle; step sizes and
  clip behavior on a given board are worth a `--calibrate` dry run before
  trusting a full corpus.
- **`capture_array` is unsafe on Windows at high rates.** All our probes
  route around it via the file path, but any new code that calls it directly
  reintroduces the sample-drop corruption. Prefer `sources/probe.py`.
- **Orphaned `hackrf_transfer.exe`** can deadlock the next capture after an
  interrupted one on Windows. Recovery: `taskkill /IM hackrf_transfer.exe /F`.
- **87.7 MHz is not quiet at the test location.** Any negative-control
  capture needs a VERIFIED quiet frequency; the guard-band assumption is
  region- and location-specific.
- **Synthetic RDS is not real RDS.** `make_synthetic_sample`'s 57 kHz carrier
  is random bits with no group framing. Do not use it as RDS ground truth.
- **Rare-flake vigilance.** The hash()-seeded fixture flake is fixed and CI
  pins `PYTHONHASHSEED`, but any new test using unseeded randomness or the
  builtin `hash()` for seeding can reintroduce it. Seed with
  `default_rng(fixed)` or hashlib.
- **Keep new files LF.** The tree is LF and CI checks the index; one CRLF
  file reintroduces the mixed state.
