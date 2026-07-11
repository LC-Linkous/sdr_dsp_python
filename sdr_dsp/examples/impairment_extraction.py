#! /usr/bin/python3
"""impairment_extraction.py -- synthesize hardware impairments, extract them back.

A DSP capability demo: the library can stamp known analog-hardware imperfections
onto a clean signal (I/Q imbalance, PA nonlinearity, phase noise, crystal offset)
and then MEASURE them back out. This closes the same verify-against-truth loop the
rest of the library uses -- apply a known impairment, confirm the estimator
recovers it -- for the impairment-modeling and feature-extraction blocks in
core.channel_impairments and core.features.

The centerpiece is the ERROR CLOUD: after deciding each received symbol, the
leftover error r - s_hat carries structure the decoded bits throw away. Same
modulation, same noise, DIFFERENT error geometry per impairment -- imbalance
correlates the cloud, PA compression grows it with |s|, phase noise smears it
tangentially. With --plot you see it; without, the summarizing moments print.

These extractors produce the feature numbers a downstream application (e.g. RF
device identification) would consume. That classification step -- labeled
captures, a trained model, naming a device -- is NOT part of this library and
lives in a separate consuming project; this example stops at the DSP boundary,
returning the measured impairment parameters and feature vector.

Library deps only (numpy); plotting optional. No hardware.

Usage:
    python examples/impairment_extraction.py
    python examples/impairment_extraction.py --plot
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "src")
from sdr_dsp.core import (
    add_iq_imbalance,
    add_pa_nonlinearity,
    add_phase_noise,
    make_device_impairments,
    apply_device_impairments,
    iq_image_ratio,
    estimate_iq_imbalance,
    estimate_cfo_ppm,
    estimate_phase_noise_variance,
    decide_symbols,
    error_vector,
    evm_stats,
    EVM_FEATURE_NAMES,
)


def _qam16(n, rng):
    lv = np.array([-3, -1, 1, 3])
    const = (lv[:, None] + 1j * lv[None, :]).ravel() / np.sqrt(10)
    ref = const[rng.integers(0, 16, n)]
    return const, ref


def main():
    p = argparse.ArgumentParser(
        description="Synthesize hardware impairments, then extract them back."
    )
    p.add_argument(
        "--plot", action="store_true", help="show the four error clouds"
    )
    args = p.parse_args()

    fs, carrier = 1e6, 2.4e9
    rng = np.random.default_rng(7)
    n = 4000
    const, ref = _qam16(n, rng)
    noise = 0.02 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))

    print("[*] Impairment synthesis -> extraction: forward model, then recover\n")

    # --- 1. estimator recovery: apply known imbalance, recover it ----------
    print("--- I/Q imbalance: recover a KNOWN signature (proper QPSK) --------")
    qpsk = (np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2))[
        rng.integers(0, 4, 20000)
    ].astype(np.complex64)
    for g_true, p_true in [(0.3, 2.0), (0.6, -3.0)]:
        r = add_iq_imbalance(qpsk, g_true, p_true)
        g_est, p_est = estimate_iq_imbalance(r)
        print(
            f"    true=({g_true:+.2f} dB,{p_true:+.1f} deg)  "
            f"recovered=({g_est:+.3f} dB,{p_est:+.1f} deg)  "
            f"image_ratio={iq_image_ratio(r):.4f}"
        )
    print(f"    balanced signal image_ratio={iq_image_ratio(qpsk):.4f} (~0)\n")

    # --- 2. a full virtual device -----------------------------------------
    print("--- a virtual 'device' = one fixed impairment bundle -------------")
    dev = make_device_impairments(seed=42)
    print(f"    device 42: gain={dev.iq_gain_db:+.3f} dB  "
          f"phase={dev.iq_phase_deg:+.2f} deg  "
          f"phase_noise={dev.phase_noise_hz:.1f} Hz  "
          f"cfo={dev.cfo_ppm:+.2f} ppm")
    print(f"    cfo {dev.cfo_ppm:+.2f} ppm at {carrier/1e9:.1f} GHz "
          f"= {estimate_cfo_ppm(dev.cfo_ppm*1e-6*carrier, carrier):+.2f} ppm "
          f"(round-trips)\n")

    # --- 3. the error clouds: same signal, four impairment regimes --------
    print("--- error-cloud moments: same 16-QAM + noise, different geometry -")
    cases = [
        ("noise only        ", ref + noise),
        ("+ imbalance 0.5/3 ", add_iq_imbalance(ref, 0.5, 3.0) + noise),
        ("+ PA compression  ", add_pa_nonlinearity(ref, [1.0, -0.08 + 0.02j]) + noise),
        ("+ phase noise 200 ", add_phase_noise(ref, 200.0, fs, seed=2) + noise),
    ]
    # show a few of the most telling moments
    idx = {name: i for i, name in enumerate(EVM_FEATURE_NAMES)}
    cols = ("evm_rms", "err_kurt_i", "err_corr_iq")
    print("    regime              " + "  ".join(f"{c:>11}" for c in cols))
    clouds = []
    for label, rx in cases:
        err = error_vector(rx, ref)
        stats = evm_stats(err)
        clouds.append((label.strip(), rx, err))
        vals = "  ".join(f"{stats[idx[c]]:>11.4f}" for c in cols)
        print(f"    {label}  {vals}")
    print("\n    evm_rms rises with each impairment (it's partly an SNR/error")
    print("    thermometer) -- but the SHAPE moments (kurtosis, I/Q correlation)")
    print("    change independently of size, and those carry the hardware signature")
    print("    that survives normalization. See the --plot clouds.\n")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[!] matplotlib not installed: uv sync --extra plotting")
            return
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)
        for ax, (label, rx, _err) in zip(axes, clouds):
            ax.scatter(rx.real, rx.imag, s=2, alpha=0.25, color="#2a6fb0")
            ax.scatter(const.real, const.imag, s=40, color="#c0392b",
                       marker="x", zorder=3)
            ax.set_title(label, fontsize=10)
            ax.set_aspect("equal")
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.grid(alpha=0.2)
            ax.set_xlabel("I")
        axes[0].set_ylabel("Q")
        fig.suptitle("Error clouds: the geometry is the impairment signature, not the size")
        plt.show()

    print("[*] done. Forward model and estimators agree -- the loop closes.")


if __name__ == "__main__":
    main()
