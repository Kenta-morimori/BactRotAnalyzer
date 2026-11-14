"""Synthetic checks for the poster spectrum-envelope fit and inverse FFT."""

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from utils.functions.frequency_analysis import fft
from utils.functions.poster_fluctuation import synthetic_long_axis
from utils.functions.poster_one_over_f import (
    amplitude_envelope,
    equal_frequency_bins,
    fit_amplitude,
    log_frequency_bins,
    random_phase_inverse_fft,
)


def script_functions():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/10_poster_one_over_f.py"), run_name="poster_fit_test"
    )


def test_amplitude_fit_recovers_known_scale_and_rejects_unusable_spectrum():
    frequency = 0.2 * 2 ** np.arange(9)
    amplitude = amplitude_envelope(frequency, 0.75, exponent=1.3, offset_hz=0.2)
    fit = fit_amplitude(frequency, amplitude)
    assert fit.scale == pytest.approx(0.75)
    assert fit.exponent == pytest.approx(1.3)
    assert fit.offset_hz == pytest.approx(0.2)
    assert fit.point_count == 9
    assert fit.bin_count == 9
    assert fit.rmse_amplitude == pytest.approx(0, abs=1e-14)
    assert fit.r_squared_amplitude == pytest.approx(1)
    assert fit.log_bin_rmse == pytest.approx(0, abs=1e-14)
    with_zero_first_amplitude = amplitude.copy()
    with_zero_first_amplitude[0] = 0
    assert fit_amplitude(frequency, with_zero_first_amplitude).offset_hz == pytest.approx(0.2)
    with pytest.raises(ValueError, match="At least three"):
        fit_amplitude(frequency[:2], amplitude[:2])


def test_amplitude_fit_minimizes_log_residuals_of_log_band_medians():
    frequency = 0.2 * 2 ** np.arange(9)
    observed = amplitude_envelope(frequency, 0.75, exponent=1.3, offset_hz=0.2) * np.exp(
        np.array([0.2, -0.05, 0.03, 0.01, -0.002, 0.08, -0.04, 0.12, -0.1])
    )
    fit = fit_amplitude(frequency, observed)
    binned_frequency, binned_amplitude, counts = log_frequency_bins(frequency, observed)
    x = np.log(binned_frequency + fit.offset_hz)
    residual = np.log(binned_amplitude) - np.log(
        amplitude_envelope(binned_frequency, fit.scale, exponent=fit.exponent, offset_hz=fit.offset_hz)
    )
    assert np.all(counts == 1)
    assert np.mean(residual) == pytest.approx(0, abs=1e-14)
    assert np.dot(x - np.mean(x), residual) == pytest.approx(0, abs=1e-14)


def test_fit_sensitivity_keeps_requested_exponent_and_uses_more_nonempty_bands():
    frequency = np.arange(1, 501, dtype=float) * 0.2
    observed = amplitude_envelope(frequency, 0.75, exponent=1.4, offset_hz=0.2)
    fixed = [fit_amplitude(frequency, observed, max_bins=count, fixed_exponent=1.0) for count in (9, 18, 36)]
    free = [fit_amplitude(frequency, observed, max_bins=count) for count in (9, 18, 36)]
    assert [fit.bin_count for fit in fixed] == sorted(fit.bin_count for fit in fixed)
    assert len({fit.bin_count for fit in fixed}) == 3
    assert all(fit.exponent == 1.0 for fit in fixed)
    assert all(fit.exponent == pytest.approx(1.4, abs=0.02) for fit in free)
    assert all(fit.log_bin_rmse < fixed[index].log_bin_rmse for index, fit in enumerate(free))
    with pytest.raises(ValueError, match="fixed exponent"):
        fit_amplitude(frequency, observed, fixed_exponent=-1)


def test_equal_hz_reference_points_cover_fft_and_keep_exponent_free():
    frequency = np.arange(1, 501, dtype=float) * 0.2
    observed = amplitude_envelope(frequency, 0.75, exponent=1.4, offset_hz=0.2)
    band_counts = []
    for width_hz in (2.0, 1.0, 0.5):
        centers, medians, counts = equal_frequency_bins(frequency, observed, width_hz)
        fit = fit_amplitude(frequency, observed, equal_width_hz=width_hz)
        assert counts.sum() == len(frequency)
        assert len(centers) == len(medians) == fit.bin_count
        assert fit.exponent == pytest.approx(1.4, abs=0.02)
        band_counts.append(fit.bin_count)
    assert band_counts == sorted(band_counts)
    assert len(set(band_counts)) == 3
    with pytest.raises(ValueError, match="frequency-band width"):
        equal_frequency_bins(frequency, observed, 0)


@pytest.mark.parametrize("sample_count,rate_hz", [(1100, 200), (1375, 250)])
def test_random_phase_inverse_fft_reproduces_saved_amplitude_even_and_odd(sample_count, rate_hz):
    offset_hz = rate_hz / sample_count
    frequency, target, phase, signal = random_phase_inverse_fft(sample_count, 1 / rate_hz, 0.6, 1.3, offset_hz, 84)
    frequency_again, target_again, phase_again, signal_again = random_phase_inverse_fft(
        sample_count, 1 / rate_hz, 0.6, 1.3, offset_hz, 84
    )
    assert np.array_equal(signal, signal_again)
    assert np.array_equal(phase, phase_again)
    assert np.array_equal(frequency, frequency_again)
    assert np.array_equal(target, target_again)
    assert not np.array_equal(signal, random_phase_inverse_fft(sample_count, 1 / rate_hz, 0.6, 1.3, offset_hz, 85)[-1])
    actual_frequency, actual_amplitude = fft(signal, 1 / rate_hz)
    assert np.allclose(actual_frequency, frequency)
    assert np.allclose(actual_amplitude, target, rtol=1e-12, atol=1e-12)
    assert np.mean(signal) == pytest.approx(0, abs=1e-14)
    assert np.isrealobj(signal)
    assert np.all((-np.pi <= phase) & (phase < np.pi))


def test_five_case_output_uses_saved_0_5s_fft_and_15_panels(tmp_path, monkeypatch):
    script = script_functions()
    analysis_script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/09_poster_fluctuation.py"), run_name="poster_input_test"
    )
    source = synthetic_long_axis(
        "constant_angle_step_plus_noise",
        100,
        2,
        84,
        mean_angular_speed_rad_s=30.0,
        angular_speed_sd_rad_s=5.0,
    )
    frame, spectra, info = analysis_script["analyze_frame"](source)
    folder = tmp_path / "inputs/tables"
    folder.mkdir(parents=True)
    cases = []
    for index in range(5):
        stem = f"case_{index}"
        frame.to_csv(folder / f"{stem}_time_series.csv", index=False)
        spectra["long_axis"].to_csv(folder / f"{stem}_long_axis_fft.csv", index=False)
        (folder / f"{stem}_metadata.json").write_text(
            json.dumps({"kind": "synthetic", "rate_hz": 100, "long_axis_fft": info["long_axis_fft"]}),
            encoding="utf-8",
        )
        cases.append(script["CaseSpec"](f"case_{index}", f"Case {index}", "inputs/tables", stem, "synthetic"))
    captured = []
    original_savefig = script["plt"].Figure.savefig

    def capture(figure, *args, **kwargs):
        captured.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(script["plt"].Figure, "savefig", capture)
    output = tmp_path / "output"
    script["generate"](tmp_path, output, 84, tuple(cases))
    assert (output / "five_case_one_over_f_0.5s.png").is_file()
    assert len(captured) == 1 and len(captured[0].axes) == 15
    assert all(captured[0].axes[5 + column].get_yscale() == "log" for column in range(5))
    assert all(not axis.collections and axis.get_legend() is None for axis in captured[0].axes)
    assert all(
        axis.xaxis.label.get_fontsize() >= 16 and axis.yaxis.label.get_fontsize() >= 16 for axis in captured[0].axes
    )
    assert all(axis.xaxis.get_ticklabels()[0].get_fontsize() >= 14 for axis in captured[0].axes)
    plotted_fit = captured[0].axes[5].lines[1].get_ydata()
    saved_fit = pd.read_csv(output / "tables/case_0_fit.csv")["fitted_fft_amplitude"].to_numpy()
    assert np.allclose(plotted_fit, saved_fit, rtol=1e-12, atol=1e-15)
    index = pd.read_csv(output / "index.csv")
    assert len(index) == 5
    assert index.phase_seed.tolist() == list(range(84, 89))
    assert all((output / path).is_file() for path in index.fit_file)
    assert all((output / path).is_file() for path in index.fit_bins_file)
    assert all((output / path).is_file() for path in index.inverse_fft_file)
    fit_bins = pd.read_csv(output / "tables/case_0_fit_bins.csv")
    assert len(fit_bins) == index.loc[0, "bin_count"]
    assert fit_bins.fft_point_count.sum() == index.loc[0, "point_count"]
    sensitivity_input = script["load_case"](tmp_path, cases[0])
    expected_primary = fit_amplitude(
        sensitivity_input["frequency_hz"], sensitivity_input["amplitude"], equal_width_hz=1.0
    )
    assert np.allclose(index.exponent, expected_primary.exponent)
    assert index.bin_count.eq(expected_primary.bin_count).all()
    expected_frequency, expected_amplitude, expected_counts = equal_frequency_bins(
        sensitivity_input["frequency_hz"], sensitivity_input["amplitude"], 1.0
    )
    assert np.allclose(fit_bins.frequency_hz, expected_frequency)
    assert np.allclose(fit_bins.median_fft_amplitude, expected_amplitude)
    assert np.array_equal(fit_bins.fft_point_count, expected_counts)
    script["draw_fit_sensitivity"](sensitivity_input, output / "fit_sensitivity_case_0_0.5s.png")
    assert (output / "fit_sensitivity_case_0_0.5s.png").is_file()
    assert len(captured[-1].axes) == 6
    assert all(axis.get_yscale() == "log" for axis in captured[-1].axes)
    assert all(len(axis.collections[0].get_offsets()) > 0 for axis in captured[-1].axes)
    script["draw_equal_hz_sensitivity"](sensitivity_input, output / "fit_equal_hz_case_0_0.5s.png")
    assert (output / "fit_equal_hz_case_0_0.5s.png").is_file()
    assert len(captured[-1].axes) == 6
    assert all(axis.get_yscale() == "log" for axis in captured[-1].axes)
    assert [len(captured[-1].axes[column].collections[0].get_offsets()) for column in range(3)] == [
        fit_amplitude(sensitivity_input["frequency_hz"], sensitivity_input["amplitude"], equal_width_hz=width).bin_count
        for width in (2.0, 1.0, 0.5)
    ]
    assert np.isfinite(index.scale).all()
    assert index.exponent.gt(0).all()
    assert np.allclose(index.offset_hz, 100 / index.sample_count)
    saved = pd.read_csv(folder / "case_0_long_axis_fft.csv")
    saved.loc[saved.window_s == 0.5, "amplitude"] *= 1.1
    saved.to_csv(folder / "case_0_long_axis_fft.csv", index=False)
    with pytest.raises(ValueError, match="saved FFT does not match"):
        script["load_case"](tmp_path, cases[0])
