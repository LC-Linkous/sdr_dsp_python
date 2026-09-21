# Adding a new SDR (and keeping the library un-locked)

This is the working process for pointing `sdr_dsp` at a radio other than the
HackRF One, and the record of which radios have been checked. The design goal
it serves: **the DSP core is device-agnostic, and no single SDR is special.**
The concept is in [MODULARITY.md](MODULARITY.md); this file is the *procedure*.

Three artifacts make the claim real and keep it honest:

- `examples/sdr_adapter_template.py` — a fill-in-the-blanks adapter to copy.
- `tests/test_modularity.py` — a guard that fails the build if the core ever
  imports a device library, and that drives the pipeline from a non-HackRF
  source.
- this document — the step-by-step process and the porting record below.

## The rule that must never break

Nothing under `src/sdr_dsp/` may import a device SDK (hackrfpy, pyrtlsdr,
SoapySDR, uhd, ...). The core operates on `complex64` arrays and nothing else.
Adapters that talk to hardware live in `examples/` (or your own code), outside
the library. `tests/test_modularity.py::test_core_imports_no_device_library`
enforces this automatically — if it is green, the library is not locked to any
radio, HackRF included.

## The process

1. **Copy the template.** `cp examples/sdr_adapter_template.py
   examples/<device>_capture.py` (and `_sink.py` if the radio transmits).
   Rename `TemplateSDRSource` to e.g. `RtlSdrSource`.

2. **Fill in the four marked spots** (`>>> FILL IN` in the template):
   1. open/tune/start the device (your vendor library call);
   2. read native samples in a `blocks()` loop;
   3. **normalize the native sample format to `complex64`** — the step most
      often gotten wrong (see the table below);
   4. transmit only: convert `complex64` back to the device TX format.
   Delete the `_FakeSDR` stand-in once the real device is wired.

3. **Set `sample_rate` and `center_freq` from what the device actually tuned**,
   not from a constant. Downstream measurements (`estimate_fm_cfo`, the pilot
   check, channelizers) trust these.

4. **Register the vendor library** in `tests/test_modularity.py::DEVICE_LIBS`.
   This does two things: it lets the guard confirm the core still doesn't
   import it, and it documents that the SDK is an *adapter-only* dependency.

5. **Verify, in this order:**
   1. *File first.* Point a `FileSource` at a saved capture and run your
      pipeline. This debugs the DSP with zero hardware variables.
   2. *Modularity guard.* `pytest tests/test_modularity.py` — the core stays
      device-free and a non-HackRF source drives the pipeline.
   3. *Format check.* Confirm your `_to_complex64` maps device midpoint → ~0
      and full scale → ~±1 (there's a cu8 example test in test_modularity.py
      to copy for your format).
   4. *Live.* Run the real adapter; compare a live capture against the file
      result. Gate any test that needs the board behind the `hardware` marker.

6. **Record it** in the porting log below, and open a PR.

## Sample-format normalization (the usual gotcha)

Do the scaling ONCE, inside the adapter's `blocks()`/`_to_complex64`. The core
only ever sees `complex64` in roughly [-1, 1].

| Radio (vendor lib) | Native format | Normalize to complex64 |
|--------------------|---------------|------------------------|
| HackRF (hackrfpy) | `ci8` (int8) | `(i + 1j*q) / 128.0` |
| RTL-SDR (pyrtlsdr) | `cu8` (uint8, mid 127.5) | `((i - 127.5) + 1j*(q - 127.5)) / 127.5` |
| USRP (uhd) | `cf32` / `sc16` | cast to complex64 / scale int16 by 32768 |
| SoapySDR devices | `CF32` or `CS16` | as above, per the stream format requested |

`src/sdr_dsp/io/sigmf.py` (`load_iq`) is the shipped worked example for `ci8`;
`FileSource` reuses it. Return `np.ascontiguousarray(..., dtype=np.complex64)`
— some vendor buffers are views or non-contiguous.

## What ports for free vs. what needs work

- **Receive + demod: free.** Write an `IQSource`; the entire DSP (filters,
  demod, CFO, channelize, measure, framing) runs unchanged.
- **Transmit: free at the seam.** Write a `TXSink`; the modulate/frame stack
  and `transmit_examples.py` drive it. Keep it guarded (`armed=False`) like
  `examples/hackrf_sink.py`.
- **Auto-gain search: NOT free.** `sources/probe.py`, `core/gain_search.py`
  (`search_gain`), and `tools/preflight_collection.py` assume the HackRF's
  `lna`/`vga`/`amp` gain model. A new radio needs its own gain model before
  auto-gain works; until then, set gains manually in the adapter. Lifting a
  `GainModel` protocol out of `search_gain` is the clean generalization when a
  second radio actually needs it (tracked in TODO §2/§6).

## Porting record

One row per radio, updated as each is added. "Modularity guard" = the file-first
+ `test_modularity.py` checks pass; "Live verified" = tested against the board.

| SDR | Vendor lib | Native fmt | RX | TX | Adapter file | Modularity guard | Live verified | Notes |
|-----|-----------|-----------|----|----|--------------|------------------|---------------|-------|
| HackRF One | hackrfpy | ci8 | ✅ | ⚠️ guarded | `examples/hackrf_capture.py`, `hackrf_sink.py` | ✅ | ✅ (RX; FM by ear) | reference adapter; TX device call left for bench |
| _(template)_ | _(fake cu8)_ | cu8 | ✅ | ⚠️ guarded | `examples/sdr_adapter_template.py` | ✅ | n/a (no hardware) | self-demo proves the seam; copy this |
| _RTL-SDR_ | _pyrtlsdr_ | _cu8_ | _—_ | _n/a (RX-only)_ | _—_ | _—_ | _—_ | _example next target; copy the template_ |

Copy the last row and fill it in for your radio. Keep this table honest — a ✅
in "Modularity guard" should mean the tests actually pass on your tree.
