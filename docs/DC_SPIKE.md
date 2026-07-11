# The DC Spike — What It Is and How to Work Around It

The tall, always-present peak at the exact center of a capture's spectrum is not
a signal on the air — it's the radio seeing itself. This note explains where it
comes from, why it matters for this library's demods and measurements, and the
recommended workaround (offset tuning, which the library already supports via
`tune_to_baseband`). It's a companion to `HARDWARE.md`.

---

## 1. Where the spike comes from

The HackRF One is a **direct-conversion (zero-IF) receiver**. To tune to
433 MHz, it generates a local oscillator (LO) at exactly 433 MHz and mixes the
antenna signal against it, so the target lands at 0 Hz in the complex baseband.

The problem is that no mixer is perfectly isolated. Some LO energy leaks
backward toward the RF port, reflects, and re-enters the mixer's own input — and
a signal mixed with *itself* produces a constant: cos(ωt)·cos(ωt) contains a DC
term. Small analog DC offsets in the baseband amplifiers and ADC add to it. The
result is a strong component at exactly 0 Hz baseband — which is exactly the
**center of the FFT**, because "0 Hz baseband" is, by construction, "the
frequency you tuned to."

![Zero-IF receiver: LO leakage self-mixes to a DC spike at the center of the baseband spectrum](img/dc_spike_receiver.svg)

The spike is therefore a property of the *receiver architecture*, not of the
spectrum. It moves when you retune (it's always at your center), it doesn't
respond to the antenna, and its level varies with gain settings and temperature.

## 2. Why it matters here

**It sits exactly where a dead-on-tuned signal is.** Tune directly to a
transmission and the spike is *inside* the channel, superimposed on the signal.

**OOK/ASK is the worst case.** `ook_envelope` is `|iq|`, and a DC offset raises
the "off" level of every envelope sample — shrinking the on/off contrast that
`ook_slice`'s threshold depends on. The midpoint default assumes the "off" level
is near zero; a strong spike breaks that quietly.

**FSK/PSK are more tolerant, not immune.** A constant complex offset is not a
constant frequency, so `instantaneous_frequency` doesn't simply shift — but the
offset distorts the phase trajectory, and the distortion grows as the signal
gets weaker relative to the spike.

**Measurements are polluted.** `power_dbfs` of a dead-center capture includes
the spike's energy, and that energy is not signal. Any survey or calibration
work over a spike-contaminated bin is biased.

## 3. The recommended fix: offset tuning

Don't capture with the signal at the hardware's 0 Hz. Tune the hardware 100–200
kHz *away* from the target, then bring the signal to baseband **in software**
with `tune_to_baseband` (in `core.mixing`). The spike stays at the *hardware's*
0 Hz — which after the software mix is 100–200 kHz away from your signal, where
a filter removes it or you simply ignore it.

![Three panels: dead-on tuning puts the spike in the signal; offset capture separates them; tune_to_baseband recenters the signal with the spike moved aside](img/dc_spike_offset_tuning.svg)

Concretely, for a 433.92 MHz target:

```python
from sdr_dsp import tune_to_baseband

fs = 2e6
hw_center = 433.77e6            # tune 150 kHz LOW of the target
# ... capture iq at hw_center ...
iq = tune_to_baseband(iq, offset_hz=150e3, sample_rate=fs)
# signal now at 0 Hz; the DC spike is at -150 kHz — filter or ignore
```

This costs nothing (one complex multiply the library already implements) and
should be the default recipe for any capture where the target frequency is
known. It is why capture tools in the wild almost always record off-center.

## 4. About `remove_dc` (and when not to trust it)

`remove_dc` subtracts the complex mean of the record — the right operation when
the record is mostly noise plus a DC offset, because then the mean *is* the
offset. But it carries the same silent precondition as other whole-record
statistics: **it assumes the signal is a small fraction of the record.** If a
strong burst dominates the capture (a typical bench capture: trigger, grab the
packet), the burst's own mean contaminates the estimate, and subtracting it
bends the signal.

If you must use DC subtraction on a burst-dominated record, estimate the DC
from a signal-free segment (before the burst) rather than the whole record. Or
— better — use offset tuning and never have the spike near your signal in the
first place.

## 5. Per-device behavior (this is adapter-layer knowledge)

How much DC contamination to expect is a property of each radio's architecture,
not of the DSP:

| Radio | Architecture | DC spike |
|---|---|---|
| HackRF One | zero-IF (direct conversion) | Prominent, always at center. Offset-tune by default. |
| RTL-SDR (R820T / R860 tuner) | low-IF | Small or absent — the tuner converts to a small intermediate frequency and the offset is handled digitally. |
| RTL-SDR (older E4000 tuner) | zero-IF | Prominent, like the HackRF. |
| USRP | varies; DC offset correction available | Typically managed by the driver; verify per device. |

The library's device-agnostic stance holds: the DSP core neither knows nor
cares which architecture produced the samples. The knowledge of *whether to
offset-tune* belongs where the device does — in the capture/adapter layer and
in per-device notes like this one — and the correction (`tune_to_baseband`,
filtering, `remove_dc` with its caveat) is an explicit, opt-in operation like
everything else here. No stage silently "fixes" DC, because whether the DC is a
receiver artifact or part of the signal is context only the user has.

See `HARDWARE.md` for the broader per-device expectations table and
`MODULATIONS.md` for what each demod tolerates.
