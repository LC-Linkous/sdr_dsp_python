# TODO (September 2026 session journal)

> **The live list is `docs/TODO.md`.** This file is the running journal from
> the September 2026 review/repair sessions, kept for the reasoning behind
> the open items. Items marked done below after 2026-10 were closed when
> the tree was prepared for packaging; the hardware items remain open.

Running list from the September 2026 review/repair sessions. Done so far,
for context: capture_health mid-band noise reference, the channelized 19 kHz
pilot check, SNR-first `search_gain` with the RF amp reachable, pilot
validation in all collection tools, the 15 kHz audio lowpass, the preflight
tool + `--station/--quiet` overrides, a real ear-verified 103.7 MHz sample
with provenance, and the pyproject/uv alignment. Suite: 785 passed / 1 skipped
as of patch 0018 (verify against YOUR tree, not a reported number).

Session 2026-09-20 (patches 0015-0018): lint gate re-greened (0015); the
sub-8 Msps warning deduped at the logging layer (0016); FM CFO estimate/correct
landed and validated on the real 103.7 capture (0017, Phase B step 3);
transmit_examples.py + offline round-trip tests through the TXSink seam (0018,
Phase C step 5). Line-ending normalization run locally. New doc: MODULARITY.md.
All of Phase A-C is now done; what remains needs hardware. NOTE: several new
files are untracked and must be `git add`ed -- see section 6.

## 1. Hardware session -- DONE 2026-09-20, follow-ups remain

- [x] Preflight PASS: 106.5 MHz chosen at pilot +27.7 dB (after the
      capture_array pipe-corruption fix, patch 0008).
- [x] Calibrate + full corpus: 17 captures, 57 MB, 15 clean; reference
      lna=40 vga=2 amp=on, pilot +23 to +30 dB.
- [ ] Recapture a native 2 Msps reference and re-import it with
      `tools/import_reference_capture.py` (current sample is decimated from
      8 Msps -- fine, but a native capture removes one processing step from
      the provenance chain). Candidate: dev_data's fm_reference_long
      (106.5, pilot +28.7) is already on disk and importable today.
- [~] Verified quiet frequency: the CODE is done -- preflight now hunts below
      87.5 MHz (old TV channel 6 territory) and a below-band carrier is blocked
      as a quiet pick without being ranked as a station (tests
      test_quiet_can_be_found_below_the_band /
      test_below_band_carrier_not_ranked_as_a_station in the untracked
      test_collection_fixes.py). REMAINING (needs the radio): actually run the
      field verification at this location -- 87.7 is occupied (27/128 counts,
      +8.8) and in-band 91.5 failed too -- pick a verified quiet freq and
      re-run `--only 5` with it as `--quiet`.
- [x] Gain-ladder "weak" tolerance at single-digit counts: DONE in
      `collect_dev_data.judge_capture` -- tolerance is
      `max(2.5*target, target+3 counts)`, so the ratio governs comfortable
      levels and the absolute floor governs the quantization regime. The exact
      failing case (target 3.0, measured 5.0 -> was UNEXPECTEDLY STRONG) is now
      "ok", covered by synthetic no-hardware tests in test_collection_fixes.py.
      (Was listed as open; the tree already has it. VERIFY / commit the tests.)

## 2. DSP features (ground truth now exists for all of these)

- [x] TX modulation examples: `transmit_examples.py` (patch 0018) modulates
      OOK/FSK/BPSK/QPSK as framed packets and keys them through a TXSink,
      proving demod(modulate(x)) == x across the sink seam. Offline by default
      (LoopbackSink, self-checking); `--sink hackrf` drives the guarded
      HackRFSink to the device seam and stops. 24 offline tests
      (test_transmit_examples.py); transmit/legal warning in the file header.
      The actual radiation is the two-SDR bench test below.
- [ ] Two-SDR bench test (needs a SECOND HackRF): one board runs
      transmit_examples.py, the other captures with live_433_capture.py /
      the RX chain, and the captured signal is decoded by the matching demod
      -- closing demod(modulate(x)) == x over a REAL channel instead of
      apply_channel(). This is the real-world validation the digital demods
      still lack (only FM is hardware-verified). Gate it behind the
      `hardware` marker like the other board tests. Prereq: TX examples above
      + all non-hardware pipeline work done.
- [ ] Extend real-data testing beyond FM: as digital captures (OOK/FSK/PSK)
      are collected, add e2e tests like test_real_demod_e2e.py that decode
      real captures with known ground-truth bits, not just synthesis. FM is
      covered (patch 0012); the digital schemes are still synthetic-only.
- [ ] RTL-SDR adapter (`examples/rtlsdr_capture.py`): the first REAL non-HackRF
      SDR, to exercise the modularity kit (docs/ADDING_AN_SDR.md,
      examples/sdr_adapter_template.py, test_modularity.py). RTL-SDR is cheap,
      RX-only, cu8 -- exactly what the template models. The adapter + its cu8
      normalization are offline-testable NOW (copy the template, wire pyrtlsdr,
      unit-test _to_complex64); only live verification waits on the ~$30 dongle
      (no device on hand as of 2026-09-21). Fills the RTL-SDR porting-record
      row from stubbed to code-complete. Prereq for a truly convincing "not
      locked to HackRF" claim: a second real radio, not just the template.

- [x] Stereo decoder: `fm_stereo_decode` (patch 0013), pilot-locked, 71-79 dB
      separation on synthetic, real L-R recovered from the 103.7 capture.
      `write_wav` gained stereo; `fm_receiver.py --stereo` uses it.
- [ ] RDS at 57 kHz: the real capture carries actual RDS frames; note the
      synthetic's 57 kHz carrier is RDS-*like* (random bits, no framing) and
      is NOT decodable -- extend the generator with real RDS group encoding
      when this starts.
- [x] CFO estimate/correction (patch 0017): `estimate_fm_cfo` /
      `correct_fm_cfo` in core.measure, exported top-level. Uses the
      discriminator DC term (channelized first, like fm_pilot_excess_db); NOT
      the pilot -- with a differential discriminator a carrier offset adds DC
      but does not move the 19 kHz pilot, so the pilot carries no CFO info (a
      test pins this). Real 103.7 capture: genuine -315 Hz (~3 ppm) corrected
      to -2 Hz, stable across halves. 10 tests (test_cfo.py).
- [x] `fm_receiver.py`: decimate to 250 kHz after the channel filter before
      demod (patch 0011) -- ~8x fewer samples through the discriminator,
      audio unchanged.

## 3. Code quality

- [~] Typing: mypy default-checks now green and gated in CI (patch 0014; 5
      real type errors fixed, scipy override added). STILL TODO as one unit:
      annotate src/sdr_dsp fully, ship a py.typed marker, then enable the
      strict flags (disallow_untyped_defs etc.). The gate is in place; the
      annotation campaign is the remaining work.
- [x] Unused imports resolved per file (patch 0010): fsk_decoder now uses
      the library fsk_demod for 2-FSK; two_station_link uses run_link;
      nask_decoder + impairment_extraction dead imports dropped;
      drivers.py dead `asdict`/`engines` removed; f-strings fixed;
      windowing_demo uses a new public `get_window` (was the `_` alias).
- [x] `fsk_demod_nlevel(smooth_samples=)` added and fsk_decoder's 4-FSK
      branch rewired to the library (patch 0011).
- [ ] Meaningful impairment round-trip demo: apply_device_impairments ->
      CFO/carrier recovery -> decide_symbols -> estimate_phase_noise_variance
      against the reference. Attempted in the cleanup pass but the naive
      version gave nonsense (CFO rotates the constellation; phase-noise
      variance needs the reference), so it was reverted to an import drop.
      This is a feature, not cleanup.
- [x] Flaky test cured (patch 0012): autouse conftest fixture seeds
      np.random before every test, eliminating the order-dependent
      global-RNG flake as a class. Three consecutive full runs at 734/734.
- [x] Migrated the 3 legacy global-np.random tests to explicit
      default_rng(seed) (patch 0014). AND found/fixed the real flake root
      cause: test_promote_sample_data seeded fixtures from the per-process
      builtin hash() -- now hashlib-seeded and deterministic across
      PYTHONHASHSEED.
- [x] Line endings: normalization run locally (2026-09-20) per
      LINE_ENDING_NORMALIZATION.md -- .gitattributes moved to the git root,
      `git add --renormalize .`, committed alone. Tree is LF now; new patches
      no longer carry CRLF context noise. (Keep NEW files LF to match.)
- [x] Lint gate re-greened (patch 0015): three ruff errors had slipped into
      two untracked diagnostic scripts that never went through the gate --
      `tools/diagnose_rate.py` (unused numpy import; a lambda assignment) and
      `tools/diagnose_path.py` (missing trailing newline). Fixed in-style
      (fix, not blanket per-file-ignore, matching preflight_collection). ruff
      is clean again; CI's ruff step would otherwise have been red on push.

## 4. Cross-repo (hackRF_python)

- [ ] **File a capture_array bug**: on Windows at 8 Msps the stdout-pipe
      streaming path drops samples, corrupting phase while power stats stay
      healthy. Complete repro: same board/antenna/station/gain, minutes
      apart -- capture_array: channel +7.2 dB / pilot +1.4 dB; capture to
      file: +17.5 / +19.4; verified reference file: +20.6 / +26.5. Also:
      interrupting capture_array mid-stream can orphan hackrf_transfer.exe,
      which deadlocks the next capture (fixed locally by
      `taskkill /IM hackrf_transfer.exe /F`). Suggested fixes upstream: a
      Windows warning or file-backed fallback in capture_array, and
      orphan reaping on interrupt.
- [ ] Audit the live-listen path: `open_receiver`/PersistentReceiver uses
      the same pipe streaming. At 2 Msps (4 MB/s) it may keep up, but listen
      for glitches during `live_fm_listen.py`; if audible, the persistent
      receiver needs the same treatment as the probes.
- [ ] Check how hackrfpy's `pilot_snr_db` is computed: if its check
      discriminates without channelizing first, port the fix from
      `sdr_dsp.core.measure.fm_pilot_excess_db` (unchannelized, a real
      8 Msps capture measured +1.8 dB vs +17 to +27 channelized).
- [ ] Consider preflight-style pilot scoring in hackrfpy's
      `scan_then_capture.py` so both repos pick stations the same way.

## 5. Docs & infra

- [ ] README + examples for the `--calibrate` dry run, e.g.
      `uv run python tools/collect_dev_data.py --station 106.5e6 --calibrate`
      -- it's unique to this workflow (probe walk + chosen gain + measured
      pilot, records nothing) and is the right habit between preflight and
      a full collection. Show a real transcript: the front-end sweep at
      vga=0, amp on/off comparison, the clip backoff at vga=20->2, and the
      final "reference: lna=40 vga=2 amp=on ... pilot +25.6 dB" line, with
      a short read of what each stage proves.
- [ ] Update `docs/dev_handbook` for the new workflow: preflight ->
      calibrate -> collect; amp semantics in `search_gain`; what the pilot
      metric means and its ~6 dB threshold.
- [x] Quieter runs (patch 0016): the sub-8 Msps warning is now deduped at the
      LOGGING layer. Root cause: `dedup_warnings` used redirect_stderr, but the
      warning comes via logging.getLogger("hackrfpy").warning(); once anything
      installs a StreamHandler bound to the real stderr (a plain basicConfig
      does it), redirect_stderr never sees the record -- so it printed ~16x per
      station. A logging.Filter on the hackrfpy logger drops repeat WARNING+
      records before any handler; INFO progress is left alone. Regression tests
      cover the bound-handler path the old code missed.
- [x] CI workflow (patch 0014): pytest matrix {win,ubuntu,mac}x{3.11-3.13}
      + ruff + mypy, hardware tests deselected, PYTHONHASHSEED pinned.
      VERIFY on push: .github/ was deleted in a working tree this session --
      confirm it is committed (see section 6) so CI actually runs.
- [x] MODULARITY.md (docs/): the HAL story -- core / IQSource+TXSink Protocols
      / device adapter / hardware -- with a diagram, the sample-format
      normalization table (ci8/cu8/cf32 -> complex64), and the honest note that
      the gain-search/probe tooling is HackRF-shaped and not yet portable.
- [x] Add `[project.urls]` to pyproject when the repo is pushed. (Done; Homepage,
      Repository and Issues are set.)

## 6. Open issues & warnings

Latent problems, unverified assumptions, and things that could bite. Not
tasks so much as things to keep in view.

- **Verify patch state against YOUR tree, never a reported pass count.** This
  session twice reasoned from the wrong tree: once a stale editable install
  read as green, once patches 13-14 were assumed applied when they had not
  been shared. "744 passes" is only meaningful on the machine it ran on. When
  in doubt, run the suite locally and trust that.
- **Only FM is hardware-validated.** AM/SSB/CW, OOK/ASK, all FSK, the PSK
  family, QAM-16 are verified against SYNTHESIS only. The status words in
  MODULATIONS.md ("Supported" vs "Demonstrable") encode this, but it is easy
  to forget that no digital demod has ever touched a real off-air signal.
  The two-SDR bench test (section 2) is what closes this.
- **The gain-search amp/level strategy is validated against a MODEL** (the
  RadioModel in the tests) plus one real station, not across bands/antennas.
  It assumes SNR improves with amp+LNA and not VGA, which is right for the
  HackRF in principle; step sizes and clip behavior on a given board are
  worth a `--calibrate` dry run before trusting a full corpus.
- **capture_array is unsafe on Windows at high rates** (section 4). All OUR
  probes route around it via the file path, but any NEW code that calls
  capture_array directly reintroduces the sample-drop corruption. Prefer
  sources/probe.py. Upstream bug not yet filed.
- **Orphaned hackrf_transfer.exe** can deadlock the next capture after an
  interrupted one on Windows. Recovery: `taskkill /IM hackrf_transfer.exe /F`.
  Watch for it during interactive sessions.
- **87.7 MHz is not quiet at the test location** -- it has a carrier
  (correctly flagged by validation). Any negative-control capture needs a
  VERIFIED quiet frequency; the default guard-band assumption is
  region- and location-specific.
- **Synthetic RDS is not real RDS.** make_synthetic_sample's 57 kHz carrier
  is random bits with no group framing -- decodable-looking, not decodable.
  Do NOT use it as RDS ground truth; the real capture's frames are the only
  valid reference (section 2 RDS item).
- **py.typed IS shipped (since patches 0019-0020) but overpromises**: most
  signatures are unannotated, so downstream mypy sees `Any` for much of the
  API. Strict flags come on per module as each is annotated (docs/TODO.md).
- **Rare-flake vigilance.** The known flake (hash()-seeded fixtures) is
  fixed and CI pins PYTHONHASHSEED, but any NEW test using unseeded
  randomness or the builtin hash() for seeding can reintroduce order/run
  dependence. Always seed with default_rng(fixed) or hashlib.
- ~~**Untracked files must be `git add`ed before they push.**~~ (Resolved: all
  listed files, including this TODO.md and .github/, are tracked on main.)
  Original note: The 785-passing
  state depends on files git does not yet track: TODO.md (kept untracked on
  purpose), the three test files that were never added (test_collection_fixes,
  test_real_demod_e2e, test_stereo_decode), and this session's new
  test_cfo.py, test_transmit_examples.py, examples/transmit_examples.py, and
  docs/MODULARITY.md. A clone/CI without them is NOT the tree tested here.
- ~~**Confirm .github/ is committed.**~~ (Confirmed; tests.yml runs on main.)
  Original note: It was deleted in a working tree this
  session (unstaged); CI (patch 0014) only runs if the workflow is actually in
  the pushed tree. Check `git status` for a stray .github deletion before push.
- **Line endings normalized 2026-09-20** -- tree is LF. Keep NEW files LF; a
  CRLF file reintroduces the mixed state the normalization just cleared.

## 7. Build plan

Ordered so everything that needs no hardware is done and CI-green before the
second HackRF is required. Each step is a single reviewable patch (or, for
the normalization, a local git op) unless noted.

**Phase A -- finish the pipeline hygiene (no hardware) -- DONE**
1. [x] Line-ending normalization run locally (LINE_ENDING_NORMALIZATION.md):
   .gitattributes at the git root, `git add --renormalize .`, committed alone.
2. [x] Quieter preflight runs: sub-8 Msps warning deduped (patch 0016).
   Also this phase: lint gate re-greened (patch 0015).

**Phase B -- DSP features with existing ground truth (no hardware)**
3. [x] CFO estimate/correction (patch 0017): discriminator-DC method,
   validated on the real 103.7 capture (-315 Hz -> -2 Hz).
4. [ ] RDS decoder at 57 kHz (section 2) -- THE NEXT NO-HARDWARE FEATURE.
   Builds on the stereo decoder's pilot-lock + coherent-detection machinery.
   Needs the synthetic generator extended with real RDS group framing first
   (so there IS synthetic ground truth), then validate against the real
   capture's actual frames. Bigger than CFO; the only no-hardware item left.

**Phase C -- TX generation path (no hardware to build/test, hardware to fire)**
5. [x] transmit_examples.py + offline tests (patch 0018): all four schemes
   round-trip through the TXSink seam; demod(modulate(x)) == x proven offline.
   This was the last big no-hardware chunk and it unblocks Phase E.

**Phase D -- collection follow-ups (needs the ONE HackRF already on hand)**
6. [ ] Re-import a native 2 Msps reference (section 1) -- one command against
   fm_reference_long already on disk (needs the dev_data capture present; it is
   NOT shipped in this tree).
7. [~] Gain-ladder tolerance re-verdict: DONE in code+tests (section 1). The
   only part still needing the radio is finding a VERIFIED quiet frequency at
   this location and re-running `--only 5` with it.

**Phase E -- two-SDR validation (needs the SECOND HackRF)**
8. [ ] Two-SDR bench test (section 2): TX on one board, RX+decode on the other,
   closing the loop over a real channel. Gated behind the `hardware` marker.
   This is the payoff -- the first real-world validation of the digital
   demods -- and everything above is arranged so this is the ONLY step
   blocked on the second board.

**Ongoing / opportunistic**
- Cross-repo hackrfpy items (section 4): file the capture_array bug; port the
  channelized pilot check. Do when touching that repo.
- Docs (section 5): --calibrate writeup, dev_handbook, [project.urls]. Fold
  in as features land; reference docs stay frozen until the code stabilizes
  (LOG.md + this TODO are the only live docs meanwhile).
- Typing campaign + py.typed (section 3): a low-risk grind for any time, but
  do it AFTER the feature work so annotations aren't rewritten as APIs move.