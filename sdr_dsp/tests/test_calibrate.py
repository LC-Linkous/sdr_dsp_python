"""Tests for opt-in absolute power calibration (dBFS -> dBm).

Verifies the offset math, the compute-from-reference helper, the conditions
stamping, the frequency-drift warning (on by default, silenceable), and
save/load fidelity. The honest default (power_dbfs) is untouched and tested
elsewhere.
"""

import os
import tempfile
import warnings

import numpy as np
import pytest

from sdr_dsp.core import power_dbfs, power_dbm, Calibration, compute_cal_offset


def _tone(amp=0.1, n=10000):
    return (amp * np.exp(2j * np.pi * 0.1 * np.arange(n))).astype(np.complex64)


def test_power_dbm_is_dbfs_plus_offset():
    iq = _tone()
    assert power_dbm(iq, 0.0) == pytest.approx(power_dbfs(iq))
    assert power_dbm(iq, 12.5) == pytest.approx(power_dbfs(iq) + 12.5)


def test_compute_offset_roundtrips_known_power():
    ref = _tone()
    cal = compute_cal_offset(ref, known_dbm=-30.0)
    # the reference must read back as exactly its known power
    assert cal.power_dbm(ref) == pytest.approx(-30.0)


def test_compute_stamps_conditions():
    ref = _tone()
    cal = compute_cal_offset(ref, known_dbm=-30.0, frequency_hz=433.92e6,
                             conditions={"lna": 16, "vga": 20},
                             notes="bench cal")
    assert cal.frequency_hz == 433.92e6
    assert cal.conditions["lna"] == 16
    assert cal.notes == "bench cal"
    assert cal.measured_at  # auto-stamped timestamp


def test_drift_warning_fires_far_away():
    cal = compute_cal_offset(_tone(), known_dbm=-30.0, frequency_hz=433.92e6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal.power_dbm(_tone(), at_frequency_hz=483.92e6)  # 50 MHz away
    assert len(w) == 1
    assert "calibration" in str(w[0].message).lower()


def test_drift_warning_silent_near():
    cal = compute_cal_offset(_tone(), known_dbm=-30.0, frequency_hz=433.92e6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal.power_dbm(_tone(), at_frequency_hz=434.0e6)  # 80 kHz away
    assert len(w) == 0


def test_drift_warning_can_be_silenced():
    cal = compute_cal_offset(_tone(), known_dbm=-30.0, frequency_hz=433.92e6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal.power_dbm(_tone(), at_frequency_hz=483.92e6, warn=False)
    assert len(w) == 0


def test_no_warning_without_frequencies():
    # if either frequency is unknown, there's nothing to compare -> no warning
    cal = compute_cal_offset(_tone(), known_dbm=-30.0)  # no frequency_hz
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal.power_dbm(_tone(), at_frequency_hz=900e6)
    assert len(w) == 0


def test_drift_threshold_configurable():
    cal = compute_cal_offset(_tone(), known_dbm=-30.0, frequency_hz=100e6,
                             drift_warn_hz=1e6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal.power_dbm(_tone(), at_frequency_hz=102e6)  # 2 MHz away, thr 1 MHz
    assert len(w) == 1


def test_save_load_roundtrip():
    d = tempfile.mkdtemp()
    cal = compute_cal_offset(_tone(), known_dbm=-25.0, frequency_hz=915e6,
                             conditions={"sdr": "USRP", "gain_db": 40},
                             notes="lab")
    fp = os.path.join(d, "x.cal.json")
    cal.save(fp)
    loaded = Calibration.load(fp)
    assert loaded.offset_db == cal.offset_db
    assert loaded.frequency_hz == cal.frequency_hz
    assert loaded.conditions == cal.conditions
    assert loaded.notes == cal.notes


def test_load_tolerates_partial_file():
    d = tempfile.mkdtemp()
    fp = os.path.join(d, "partial.cal.json")
    # a minimal hand-written file with only the offset
    import json
    with open(fp, "w") as f:
        json.dump({"offset_db": -10.0}, f)
    cal = Calibration.load(fp)
    assert cal.offset_db == -10.0
    assert cal.frequency_hz is None


def test_calibration_object_reusable():
    cal = Calibration(offset_db=-10.0, frequency_hz=433e6)
    a, b = _tone(amp=0.1), _tone(amp=0.2)
    # same calibration applied to two captures
    assert cal.power_dbm(a) == pytest.approx(power_dbfs(a) - 10.0)
    assert cal.power_dbm(b) == pytest.approx(power_dbfs(b) - 10.0)
