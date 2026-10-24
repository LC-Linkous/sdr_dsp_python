"""plot_sink: the matplotlib helpers build real figures (headless).

Previously 0% covered. Runs under the Agg backend so nothing opens a window;
``show=False`` keeps ``plt.show()`` out of the way entirely, and one test
checks that ``show=True`` under Agg is harmless too (it is a no-op there).
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

from sdr_dsp.sinks import plot_spectrum, plot_spectrogram  # noqa: E402

FS = 1_000_000


@pytest.fixture
def tone_iq():
    t = np.arange(50_000) / FS
    return np.exp(1j * 2 * np.pi * 123_000 * t).astype(np.complex64)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_plot_spectrum_returns_a_populated_axes(tone_iq):
    fig, ax = plot_spectrum(tone_iq, FS, show=False)
    assert fig is not None and ax.lines, "no PSD line drawn"
    x, y = ax.lines[0].get_data()
    assert len(x) == len(y) > 0
    # peak of the PSD sits at the tone (baseband -> Hz axis)
    assert abs(x[np.argmax(y)] - 123_000) < FS / 2048 * 2
    assert "Hz" in ax.get_xlabel()


def test_plot_spectrum_center_freq_uses_mhz_axis(tone_iq):
    _, ax = plot_spectrum(tone_iq, FS, center_freq=100e6, title="t", show=False)
    x, y = ax.lines[0].get_data()
    assert "MHz" in ax.get_xlabel()
    assert abs(x[np.argmax(y)] - (100e6 + 123_000) / 1e6) < 0.01
    assert ax.get_title() == "t"


def test_plot_spectrogram_draws_an_image(tone_iq):
    fig, ax = plot_spectrogram(tone_iq, FS, nfft=256, overlap=0.5, show=False)
    assert ax.images, "no spectrogram image drawn"
    img = ax.images[0].get_array()
    assert img.ndim == 2 and img.size > 0
    assert np.all(np.isfinite(img))


def test_show_true_is_harmless_headless(tone_iq):
    # Agg has no window; plt.show() returns immediately (possibly with a
    # UserWarning). The helper must not raise.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, _ = plot_spectrum(tone_iq, FS, show=True)
    assert fig is not None


def test_import_error_message_names_the_extra(monkeypatch):
    import builtins
    from sdr_dsp.sinks import plot_sink
    real_import = builtins.__import__

    def no_mpl(name, *a, **k):
        if name.startswith("matplotlib"):
            raise ImportError("simulated missing matplotlib")
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_mpl)
    with pytest.raises(ImportError, match="pip install matplotlib"):
        plot_sink._plt()
