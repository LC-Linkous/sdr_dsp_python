"""FM carrier-frequency-offset estimation and correction (Phase B, step 3).

Two grounds of truth:

* Synthetic -- an FM signal with a KNOWN injected CFO. The discriminator-DC
  estimator must recover it to within a few Hz, with the right sign, and
  correcting must drive the residual to zero.
* Real -- the shipped 103.7 MHz capture carries a genuine, unknown crystal
  offset (measured ~-315 Hz, i.e. ~3 ppm on a HackRF). We can't assert its
  exact value, but it must be finite, stable across the record (a real offset,
  not noise), and correction must collapse it toward zero.

The pilot-invariance test pins the physical reason the estimator uses the
discriminator DC term and NOT the 19 kHz pilot: with a differential
discriminator a carrier offset adds a DC term but leaves the pilot where it is.
"""

from pathlib import Path

import numpy as np
import pytest

from sdr_dsp import (add_cfo, correct_fm_cfo, estimate_fm_cfo, fm_demod,
                     fm_modulate)

SAMPLE = (Path(__file__).resolve().parent.parent / "sample_data"
          / "fm_2Msps.iq")
FS = 2_000_000.0


def _fm_signal(n=None, fs=FS, deviation_hz=60_000.0, seed=0):
    """A wideband FM signal from a few audio tones (zero-mean message)."""
    if n is None:
        n = int(fs * 0.3)
    t = np.arange(n) / fs
    msg = (0.5 * np.sin(2 * np.pi * 440 * t)
           + 0.3 * np.sin(2 * np.pi * 1200 * t)
           + 0.2 * np.sin(2 * np.pi * 30 * t))
    return fm_modulate(msg, deviation_hz=deviation_hz, sample_rate=fs)


# ---------------------------------------------------------------------------
# synthetic: known injected offset
# ---------------------------------------------------------------------------
def test_zero_cfo_reads_near_zero():
    iq = _fm_signal()
    assert abs(estimate_fm_cfo(iq, FS)) < 5.0


@pytest.mark.parametrize("true_cfo", [+1500.0, -3200.0, +12000.0, -500.0])
def test_recovers_injected_cfo_with_sign(true_cfo):
    iq = add_cfo(_fm_signal(), true_cfo, FS)
    est = estimate_fm_cfo(iq, FS)
    assert est is not None
    assert np.sign(est) == np.sign(true_cfo)
    assert abs(est - true_cfo) < 10.0, f"est {est} vs {true_cfo}"


def test_correction_drives_residual_to_zero():
    iq = add_cfo(_fm_signal(), -3200.0, FS)
    corrected, applied = correct_fm_cfo(iq, FS)
    assert abs(applied - (-3200.0)) < 10.0
    assert abs(estimate_fm_cfo(corrected, FS)) < 5.0


def test_correct_uses_explicit_cfo_when_given():
    """Passing cfo_hz skips estimation and applies exactly that shift.

    Feed a zero-CFO signal but claim a +2000 Hz offset. If the explicit value
    is honoured (not re-estimated to ~0), the signal is shifted DOWN by 2000
    and now reads -2000 -- which both proves estimation was skipped and pins
    the correction direction (correct = shift by -cfo).
    """
    iq = _fm_signal()
    corrected, applied = correct_fm_cfo(iq, FS, cfo_hz=2000.0)
    assert applied == 2000.0
    assert abs(estimate_fm_cfo(corrected, FS) - (-2000.0)) < 10.0


def test_guards_return_none_and_noop():
    iq = _fm_signal()
    assert estimate_fm_cfo(iq[:100], FS) is None          # too short
    assert estimate_fm_cfo(iq, 100_000.0) is None         # rate too low
    # correct_fm_cfo must be a safe no-op when the offset can't be estimated
    out, applied = correct_fm_cfo(iq[:100], FS)
    assert applied == 0.0 and len(out) == 100


# ---------------------------------------------------------------------------
# the design decision, pinned: CFO moves the discriminator DC, not the pilot
# ---------------------------------------------------------------------------
def test_pilot_position_is_cfo_invariant_but_dc_shifts():
    """Why the estimator uses the DC term and not the 19 kHz pilot.

    Build FM whose message contains a 19 kHz pilot tone. A carrier offset must
    (a) shift the demodulated DC by exactly the offset, and (b) leave the pilot
    sitting at 19 kHz. If someone ever 'improves' the estimator to track the
    pilot, (b) is why it would measure nothing.
    """
    fs = FS
    n = int(fs * 0.3)
    t = np.arange(n) / fs
    msg = 0.3 * np.sin(2 * np.pi * 19_000.0 * t) + 0.4 * np.sin(2 * np.pi * 800 * t)
    iq = fm_modulate(msg, deviation_hz=60_000.0, sample_rate=fs)
    cfo = 4000.0
    dirty = add_cfo(iq, cfo, fs)

    clean_d = fm_demod(iq, deviation_hz=60_000.0, sample_rate=fs)
    dirty_d = fm_demod(dirty, deviation_hz=60_000.0, sample_rate=fs)
    # (a) DC shifts by cfo/deviation (demod is scaled by deviation here)
    dc_shift_hz = (np.mean(dirty_d) - np.mean(clean_d)) * 60_000.0
    assert abs(dc_shift_hz - cfo) < 50.0

    # (b) the pilot peak stays at 19 kHz in both
    def pilot_peak_hz(d):
        w = np.hanning(len(d))
        spec = np.abs(np.fft.rfft(d * w))
        f = np.fft.rfftfreq(len(d), 1.0 / fs)
        band = (f > 17_000) & (f < 21_000)
        return float(f[band][np.argmax(spec[band])])
    assert abs(pilot_peak_hz(clean_d) - 19_000.0) < 50.0
    assert abs(pilot_peak_hz(dirty_d) - 19_000.0) < 50.0


# ---------------------------------------------------------------------------
# real capture: a genuine, unknown offset
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SAMPLE.exists(), reason="sample_data not present")
def test_real_capture_has_stable_finite_cfo_and_corrects():
    from sdr_dsp.io.sigmf import load_iq
    iq, _ = load_iq(str(SAMPLE))
    cfo = estimate_fm_cfo(iq, FS)
    assert cfo is not None
    # a plausible crystal offset for a HackRF at ~104 MHz: tens of Hz to a few kHz
    assert 10.0 < abs(cfo) < 5000.0, f"implausible CFO {cfo} Hz"
    # correction collapses it toward zero
    corrected, applied = correct_fm_cfo(iq, FS, cfo_hz=cfo)
    resid = estimate_fm_cfo(corrected, FS)
    assert abs(resid) < 0.2 * abs(cfo) and abs(resid) < 50.0
    # it's a real offset, not noise: the two halves agree
    h = iq.size // 2
    c1, c2 = estimate_fm_cfo(iq[:h], FS), estimate_fm_cfo(iq[h:], FS)
    assert abs(c1 - c2) < 150.0, f"halves disagree: {c1} vs {c2}"
