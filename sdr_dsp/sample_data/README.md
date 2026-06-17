# hackrfpy sample data

Real recordings from a HackRF One, for trying the library and downstream processing without owning a board.

- collected: 2026-06-15T13:48:13
- device firmware: 2024.02.1 (API:1.08)
- tools: git-b1dbb47
- sample rate: 2 Msps, 0.5s per IQ capture

Each `.iq` is interleaved int8 I/Q (HackRF native) with a `.sigmf-meta` sidecar describing frequency, rate, and gains. Load with:

```python
from hackrfpy import load_iq, read_sigmf_meta
iq = load_iq('fm_2Msps.iq')
meta = read_sigmf_meta('fm_2Msps.iq')
```

## Files

- `fm_2Msps.iq`
- `fm_sweep.csv`
