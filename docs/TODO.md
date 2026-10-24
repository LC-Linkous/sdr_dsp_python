# TODO

Open work, roughly in priority order. Items the LOG defers "to TODO" land
here. Strike through or delete when done; the LOG records what happened.

## Hardware (the gate everything else waits on)

- [ ] **Wired one-way TX bench test** of the FSK packet chain
      (`examples/two_station_link.py`). Everything digital, TX and protocol
      is software-only until this runs (REFERENCE §12).
- [ ] **Windows stdout-pipe corruption at 8 Msps** (`capture_array` /
      `open_receiver`; LOG 2026-09-20). Unresolved upstream in hackrfpy 1.1.0.
      Suspect throughput rather than framing: watch `hackrf_transfer`'s
      dropped-sample stats on stderr, try a large `read_samples`, and A/B
      against the file path again. Until it's settled, every measurement
      probe stays on `sources/probe.py`; `HackRFCapture` (live adapter)
      still streams through the pipe.
- [ ] Re-verify `HackRFCapture.blocks()` live once hackrfpy ships the
      `PersistentReceiver.blocks()` odd-byte fix (I/Q swap after an odd
      chunk; found 2026-10-24).
- [ ] CFO-ppm capture experiment: the cheap go/no-go for whether real
      same-model devices separate on the fingerprint features (LOG 07-11).

## Library

- [ ] Lift the HackRF three-knob gain model out of `core/gain_search.py`
      into a `GainModel` protocol so `search_gain` can drive another radio
      (MODULARITY §2).
- [ ] Give `live_fm_listen.py`'s `FMStream` over to the `Streaming*` stages
      (`stream/stages.py`) now that they exist; it should become a thin
      composition, and the stereo path can then go live too.
- [ ] F3 transient features (fingerprint plan) -- deferred to the hardware era.

## Tooling / typing

- [ ] mypy strict + `py.typed` honesty: the marker ships but most signatures
      are unannotated (~350 errors under `disallow_untyped_defs`). Enable the
      strict flags per module as each is annotated; until then the marker
      overpromises, exactly the hackrfpy pre-1.1.0 situation.
- [ ] Line-ending normalization: `.gitattributes` is committed, the
      `git add --renormalize .` commit has not been made. Do it alone.

## Repo hygiene

- [ ] `sample_data/hackrf_one/*.iq` is 160 MB of raw IQ committed directly,
      byte-identical (sha256) to `hackrfpy/tests/fm_reference/`. Move to Git
      LFS or a release asset and fetch on demand; each file is 80 MB, near
      GitHub's 100 MB hard limit, and `.git` is already 121 MB.
- [ ] LOG entries dated 2026-09-21..23 describe work that is in commits
      stamped 2026-09-20 (the LOG was written ahead of the commits). Note
      it in the LOG header or accept the drift.
