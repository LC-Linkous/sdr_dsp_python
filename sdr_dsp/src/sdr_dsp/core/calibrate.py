"""Opt-in absolute power calibration: turn dBFS into dBm.

The library's honest default is `power_dbfs` -- power relative to full scale,
which needs no external reference and is always correct. Absolute power (dBm)
requires a calibration constant that depends on the *entire* receive chain
(antenna, gains, frequency, the SDR's ADC reference). The library cannot know
that constant, so it never guesses: nothing here runs unless you explicitly
bring a calibration.

    dBm = dBFS + offset

You obtain the offset by measuring a signal of KNOWN power: feed a calibrated
source (e.g. a signal generator at -30 dBm) into your exact setup, and
`compute_cal_offset` derives the offset and stamps it with the conditions it was
measured under. That stamp matters -- the offset is only valid for the gain and
frequency it was measured at. Change the LNA gain or tune far away and it's
wrong, which is why a Calibration carries its own scope and warns when you apply
it far from where it was made.

This is honest about its limits: a calibration is only as good as your reference
source, and only valid near the conditions it was measured at. If you don't have
a calibrated reference, stay with `power_dbfs`.

Nothing in this module is device-specific. The `conditions` dict is free-form,
so a HackRF (lna/vga/amp), a USRP, or any other SDR records whatever it likes;
`frequency_hz` is promoted to a first-class field only because the drift warning
needs it.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

import numpy as np

from .measure import power_dbfs

# default: warn when applying a calibration more than this far from where it was
# measured (the dBFS->dBm offset drifts with frequency).
DEFAULT_DRIFT_WARN_HZ = 5e6


def power_dbm(iq, cal_offset_db):
    """Absolute power in dBm = power_dbfs(iq) + cal_offset_db. OUR code.

    The one-off, stateless form: you supply the calibration offset directly.
    For repeated work, or to keep the offset with the conditions it's valid for,
    use a `Calibration` object instead.

    The result is only meaningful if cal_offset_db came from a real measurement
    of a known reference at the SAME gain/frequency as `iq`. Garbage offset in,
    confidently-wrong dBm out -- there is no way for this function to check that,
    so the responsibility is yours.
    """
    return power_dbfs(iq) + float(cal_offset_db)


@dataclass
class Calibration:
    """A dBFS->dBm offset plus the conditions it was measured under.

    The offset alone is a footgun -- it's only valid for the receive-chain setup
    it was measured at. So a Calibration carries that context: `frequency_hz`
    (first-class, for the drift warning) and a free-form `conditions` dict for
    everything else (gains, antenna, SDR model, temperature -- whatever you want
    to record). `notes` and `measured_at` are for your own bookkeeping.

    Apply it with `.power_dbm(iq)`. If you apply it far from `frequency_hz`, it
    warns (the offset drifts with frequency); pass `warn=False` to silence, or
    widen/disable the threshold with `drift_warn_hz`.
    """
    offset_db: float
    frequency_hz: float | None = None
    conditions: dict = field(default_factory=dict)
    notes: str = ""
    measured_at: str = ""
    drift_warn_hz: float = DEFAULT_DRIFT_WARN_HZ

    def __post_init__(self):
        self.offset_db = float(self.offset_db)
        if self.frequency_hz is not None:
            self.frequency_hz = float(self.frequency_hz)
        if not self.measured_at:
            self.measured_at = datetime.now(timezone.utc).isoformat()

    def power_dbm(self, iq, at_frequency_hz=None, warn=True):
        """Absolute power of `iq` in dBm, using this calibration's offset.

        at_frequency_hz: the frequency `iq` was captured at, if known. When both
            this and the calibration's frequency_hz are set, applying the
            calibration more than drift_warn_hz away raises a warning (the offset
            is only valid near where it was measured).
        warn: set False to silence the drift warning for this call.
        """
        if (warn and at_frequency_hz is not None
                and self.frequency_hz is not None
                and self.drift_warn_hz is not None and self.drift_warn_hz > 0):
            delta = abs(float(at_frequency_hz) - self.frequency_hz)
            if delta > self.drift_warn_hz:
                warnings.warn(
                    f"applying a calibration measured at "
                    f"{self.frequency_hz/1e6:g} MHz to a signal at "
                    f"{float(at_frequency_hz)/1e6:g} MHz "
                    f"({delta/1e6:g} MHz away, > {self.drift_warn_hz/1e6:g} MHz "
                    f"threshold); the dBm reading may be off. Re-calibrate at "
                    f"this frequency, or pass warn=False to silence.",
                    stacklevel=2,
                )
        return power_dbfs(iq) + self.offset_db

    # -- persistence --------------------------------------------------------
    def save(self, path):
        """Write the calibration to a human-readable JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return str(path)

    @classmethod
    def load(cls, path):
        """Load a calibration saved by `save`."""
        with open(path, "r") as f:
            d = json.load(f)
        # tolerate older/partial files: only pass known fields
        known = {"offset_db", "frequency_hz", "conditions", "notes",
                 "measured_at", "drift_warn_hz"}
        return cls(**{k: v for k, v in d.items() if k in known})

    def __repr__(self):
        f = f"{self.frequency_hz/1e6:g}MHz" if self.frequency_hz else "no-freq"
        return (f"Calibration(offset={self.offset_db:+.1f}dB @ {f}"
                + (f", {len(self.conditions)} conditions)" if self.conditions
                   else ")"))


def compute_cal_offset(iq, known_dbm, frequency_hz=None, conditions=None,
                       notes="", drift_warn_hz=DEFAULT_DRIFT_WARN_HZ):
    """Derive a calibration from a measurement of a KNOWN-power reference.

    Feed in `iq` captured from a calibrated source whose true power is
    `known_dbm` (e.g. a signal generator set to -30 dBm), and this returns a
    ready-to-use Calibration stamped with the conditions you pass:

        offset = known_dbm - power_dbfs(reference_iq)

    Record the conditions honestly -- the offset is only valid at the gain and
    frequency this reference was captured at. Example:

        cal = compute_cal_offset(ref_iq, known_dbm=-30.0,
                                 frequency_hz=433.92e6,
                                 conditions={"lna": 16, "vga": 20, "amp": False})
        cal.save("hackrf_433.cal.json")
        ...
        cal = Calibration.load("hackrf_433.cal.json")
        dbm = cal.power_dbm(capture, at_frequency_hz=433.92e6)

    Returns a Calibration. The measurement should be on a steady reference tone;
    a noisy or fluctuating source gives a noisy offset.
    """
    measured_dbfs = power_dbfs(iq)
    offset = float(known_dbm) - measured_dbfs
    return Calibration(
        offset_db=offset,
        frequency_hz=frequency_hz,
        conditions=dict(conditions or {}),
        notes=notes,
        drift_warn_hz=drift_warn_hz,
    )
