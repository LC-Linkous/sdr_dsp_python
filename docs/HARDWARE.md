# Hardware Notes

This library is device-agnostic — it processes complex IQ regardless of source.
But it was developed and tested against a specific radio, the **HackRF One**, and
honesty about that hardware's strengths and limits matters for setting correct
expectations about what the demods can do.

## The HackRF One

| Spec | Value | What it means |
|---|---|---|
| Frequency range | 1 MHz – 6 GHz | Enormous coverage — HF through the common microwave bands. |
| Sample rate | up to 20 Msps | Wide instantaneous bandwidth; you can watch a lot of spectrum at once. |
| ADC resolution | **8-bit** | The key limitation. ~48 dB of theoretical dynamic range. |
| Front-end filtering | minimal | No sharp pre-selection; strong out-of-band signals can intrude. |
| Duplex | half | Receive or transmit, not both at once. |

The HackRF is a **wide, flexible, affordable** front end. Its tradeoff is the
8-bit ADC and limited front-end filtering: less dynamic range and weaker
adjacent-signal rejection than higher-bit radios. That is exactly why it's a good
*development* platform — it covers the whole spectrum and forces honest DSP — but
it's not the best *reception* platform for demanding modulations.

## What the 8-bit ADC means for demodulation

Dynamic range is how well the radio separates a weak signal sitting near a strong
one. With 8 bits, a strong signal uses most of the range and a weak nearby signal
gets quantized coarsely. Consequences for the demod suite:

- **Analog (AM/FM/SSB/CW), OOK, FSK** — fine. These don't need much dynamic
  range; the HackRF handles them well.
- **PSK (BPSK/QPSK/differential)** — fine to good. Phase is robust to amplitude
  quantization.
- **8-PSK** — demonstrable on strong, clean signals. The 45° point spacing is
  more sensitive to noise and any residual carrier error.
- **QAM-16** — demonstrable, not robust. QAM packs information into *amplitude*
  too, which is exactly what 8-bit quantization hits hardest. On a strong clean
  capture it works; over the air at low SNR, expect errors. We provide it to show
  the principle, not to claim a QAM modem.

## Cross-SDR context (so expectations are calibrated)

| Radio | ADC | Rough role |
|---|---|---|
| RTL-SDR | 8-bit | Cheap, narrow (~2.4 MHz), great starter. Similar dynamic-range class. |
| HackRF One | 8-bit | Wide tuning + bandwidth; development workhorse (this library). |
| Airspy | 12-bit | Better dynamic range, narrower tuning. |
| SDRplay | 14-bit | Strong dynamic range and front-end filtering; better for weak-signal/QAM work. |
| USRP (Ettus) | 12–16-bit | Lab-grade; what you'd reach for to do robust high-order demod. |

The point is not that the HackRF is weak — it's that a 12–16 bit radio with
front-end filtering will demodulate QAM and high-order PSK *more robustly*. On the
HackRF, those are "we can show it works on a good capture," and this library is
honest about that line rather than implying every modulation decodes flawlessly.

## Why develop on it anyway

We're pushing a $500 radio toward advanced skill-building deliberately. The
HackRF's coverage means one device exercises the entire demod suite across real
bands — broadcast FM, aircraft AM, ISM-band FSK/OOK, 2.4 GHz spread spectrum —
and its limitations *force* the DSP to be honest. A library that works on
HackRF captures, with clear-eyed caveats where the hardware strains, is more
trustworthy than one tuned to a lab radio and quietly assumed to generalize.

See `MODULATIONS.md` for the per-modulation status table.
