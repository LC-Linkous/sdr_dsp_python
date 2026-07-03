# Modulation Support

This library demodulates the modulations you actually encounter on accessible
bands with a software-defined radio: analog voice, ISM-band digital schemes, and
the approachable digital phase/frequency/amplitude families. **It is not a
modem.** Channel coding (FEC), equalization, and high-order QAM beyond the
demonstrations below are out of scope by design.

The status column is honest about what "supported" means:

- **Supported** — works on real captures (given a reasonable signal). Tested on
  synthetic ground truth; the algorithm is sound and not fragile.
- **Demonstrable** — works on a clean/strong capture or a known parameter (e.g.
  a known spreading code). The principle is real; robust low-SNR reception is
  not claimed.
- **Visualize-only** — we can show it (spectrogram) and detect structure, but
  not decode the data.
- **Out of scope** — not provided, with the reason.

## Status table

| Modulation | Family | Function(s) | Status | Notes |
|---|---|---|---|---|
| AM | analog amplitude | `am_demod` | Supported | Envelope, optional DC block. Aircraft, broadcast, shortwave. |
| FM | analog frequency | `fm_demod` | Supported | Phase discriminator. Broadcast, NOAA, ham. |
| SSB (USB/LSB) | analog | `ssb_demod` | Supported | Sideband selected in frequency domain. Ham voice, marine/aviation HF. |
| DSB-SC | analog | `dsb_sc_demod` | Supported | Both sidebands, carrier suppressed. The AM/SSB midpoint. |
| CW / Morse | on-off (audio) | `cw_decode` | Supported | Tone + envelope + timing + lookup. Hand-keyed timing is loose; may need tuning. |
| OOK / ASK | digital amplitude | `ook_envelope`, `ook_slice` | Supported | Key fobs, cheap ISM sensors, garage doors. |
| M-ASK (4-ASK…) | digital amplitude | `nask_slice` | Supported | N amplitude levels; pass explicit levels for real signals. |
| 2-FSK | digital frequency | `fsk_demod` | Supported | Weather stations, TPMS, IoT, POCSAG pagers. |
| 4-FSK / CPFSK | digital frequency | `fsk_demod_nlevel` | Supported | DMR, P25, some pagers. CPFSK recovers the same way. |
| GFSK | digital frequency | `fsk_demod` | Supported | Bluetooth, nRF24. Gaussian shaping is transmit-side; FSK demod recovers it. |
| MSK / GMSK | digital frequency | `fsk_demod` | Demonstrable | GSM, AIS. Basic FSK recovery works; full demod of specific protocols out of scope. |
| BPSK | digital phase | `bpsk_demod` | Supported | Needs rough carrier alignment (use `carrier_recovery`). PSK31, data links. |
| QPSK | digital phase | `qpsk_demod` | Supported | Coherent; recover carrier + timing first. Gray-coded. Satellites, data links. |
| 8-PSK | digital phase | `psk8_demod` | Demonstrable | Coherent; 45° spacing demands good recovery + SNR. Harder on an 8-bit SDR. |
| DBPSK / DQPSK | digital phase | `dbpsk_demod`, `dqpsk_demod` | Supported | Differential — **no carrier recovery needed**. Robust, block-friendly. |
| QAM-16 | amplitude+phase | `qam16_demod` | Demonstrable | Needs carrier + timing + amplitude scaling. Works on clean/strong captures; not a robust modem (no equalizer). |
| DSSS | spread spectrum | `dsss_despread` | Demonstrable | Despreads with a **known** code. Blind code recovery is out of scope. |
| FHSS | spread spectrum | `fhss_detect_hops` | Visualize-only | See and track hops via spectrogram. Blind decode needs the hop sequence — out of scope. |
| QAM-64/256 | amplitude+phase | — | Out of scope | Needs equalization + the full modem stack. |
| Higher PSK (16-PSK+) | digital phase | — | Out of scope | SNR demands exceed practical use here. |
| FEC / channel coding | — | — | Out of scope | This is a DSP/demod library, not a codec. |

## The recovery layer

Coherent demods (BPSK, QPSK, 8-PSK, QAM-16) need the carrier and symbol timing
recovered first. Those are **separate, composable primitives** in `core.sync`:

- `carrier_recovery(iq, method="costas"|"decision_directed", order=2|4)`
- `symbol_sync(iq, sps, method="gardner"|"early_late"|"mueller_muller")`

They are kept separate (not hidden inside the demods) so you compose the
recovery you want, can test it in isolation, and can inspect convergence — every
loop optionally returns per-sample diagnostics (error, estimate, lock trace) and
can log them to CSV. The library never auto-recovers; nothing is assumed.

The differential demods (DBPSK/DQPSK) deliberately need **no** carrier recovery
— they encode bits in phase *changes*, so a constant offset cancels. That makes
them the most robust digital choice when you don't want a carrier loop.
