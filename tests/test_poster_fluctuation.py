"""Synthetic contracts for the poster measurement path."""

import json
import runpy
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd

from utils.functions.poster_fluctuation import (
    analyze,
    axis_angular_velocity,
    band_amplitude_ratios,
    duplicate_source_map,
    major_axis_deg,
    synthetic_long_axis,
)


def test_long_axis_angular_velocity_recovers_after_missing_frame_without_crossing_it():
    time = np.arange(6) / 200.0
    axis = np.array([10.0, 20.0, np.nan, 40.0, 50.0, 60.0])
    angle, omega = axis_angular_velocity(time, axis)
    assert np.isnan(omega[0])
    assert np.isfinite(omega[1])
    assert np.isnan(omega[2:4]).all()
    assert np.isfinite(omega[4:]).all()
    assert angle[3] == 0.0
    assert np.allclose(omega[4:], -np.deg2rad(10) * 200)


def test_long_axis_sign_on_known_ccw_motion():
    frame = synthetic_long_axis("constant_angle_step_plus_noise", 200, 6, 81, angle_noise_sd_rad=0)
    time = frame.time_sec.to_numpy(float)
    _, omega_axis = axis_angular_velocity(time, frame.long_axis_deg.to_numpy(float))
    assert np.allclose(omega_axis[1:], 2 * np.pi * 8, atol=1e-10)
    assert not any(column.endswith("_um") or column.startswith("orbit_") for column in frame)


def test_fluctuation_preserves_rows_and_missing_window():
    time = np.arange(100) / 100.0
    omega = 5 + np.sin(2 * np.pi * 2 * time)
    omega[50] = np.nan
    result = analyze(time, omega, windows=(0.1,))
    assert len(result.frame) == len(time)
    assert np.isnan(result.frame.loc[50, "sd_per_mean_0.1s"])
    assert np.isnan(result.frame.loc[48:52, "sd_per_mean_0.1s"]).all()
    assert np.isfinite(result.frame.loc[30, "sd_per_mean_0.1s"])
    assert result.metadata[0]["sample_count"] < 100


def test_fluctuation_window_slides_one_frame_at_both_acquisition_rates():
    for rate in (200, 250):
        time = np.arange(6 * rate) / rate
        omega = 10 + np.sin(2 * np.pi * 2 * time)
        frame = analyze(time, omega, windows=(0.1,)).frame
        interior_second = (time >= 2) & (time < 3)
        assert frame.loc[interior_second, "sd_per_mean_0.1s"].notna().sum() == rate


def test_fixed_band_ratio_uses_mean_amplitudes_and_excludes_band_edges():
    spectrum = pd.DataFrame(
        {
            "window_s": [0.1] * 8 + [0.2] * 3,
            "frequency_hz": [0, 0.25, 0.5, 1, 79.9, 80, 90, 100, 0.5, 80, 90],
            "amplitude": [100, 2, 4, 900, 800, 1, 3, 1000, 2, 0, 0],
        }
    )
    ratio = band_amplitude_ratios(spectrum, windows=(0.1, 0.2))
    assert ratio.loc[0, "low_bin_count"] == 2
    assert ratio.loc[0, "high_bin_count"] == 2
    assert ratio.loc[0, "low_mean_amplitude"] == 3
    assert ratio.loc[0, "high_mean_amplitude"] == 2
    assert ratio.loc[0, "low_minus_high_amplitude"] == 1
    assert ratio.loc[0, "low_to_high_amplitude_ratio"] == 1.5
    assert np.isnan(ratio.loc[1, "low_to_high_amplitude_ratio"])


def test_fixed_band_ratio_assigns_nominal_fft_boundaries_consistently():
    spectrum = pd.DataFrame(
        {
            "window_s": [1.0] * 4,
            "frequency_hz": [0.5, 1 - 1e-12, 80 - 1e-12, 100 - 1e-12],
            "amplitude": [2, 100, 4, 100],
        }
    )
    ratio = band_amplitude_ratios(spectrum, windows=(1.0,)).iloc[0]
    assert ratio.low_bin_count == 1
    assert ratio.high_bin_count == 1
    assert ratio.low_to_high_amplitude_ratio == 0.5


def test_fixed_band_ratio_preserves_windows_when_fft_has_no_valid_segment():
    empty = pd.DataFrame(columns=["window_s", "frequency_hz", "amplitude"])
    ratios = band_amplitude_ratios(empty, windows=(0.1, 0.2))
    assert ratios.window_s.tolist() == [0.1, 0.2]
    assert ratios.low_bin_count.tolist() == [0, 0]
    assert ratios.high_bin_count.tolist() == [0, 0]
    assert ratios.low_to_high_amplitude_ratio.isna().all()


def test_random_angle_noise_is_independent_without_rotation():
    frame = synthetic_long_axis("random_angle_noise", 200, 6, 81)
    assert len(frame) == 1200
    assert frame.baseline_angle_rad.eq(0).all()
    assert abs(frame.angle_noise_rad.mean()) < 0.001
    assert 0.009 < frame.angle_noise_rad.std() < 0.011
    assert abs(frame.angle_noise_rad.autocorr()) < 0.1
    assert not any(column.endswith("_um") for column in frame)


def test_constant_angle_step_with_noise_uses_angle_difference_and_flat_window_scale():
    for rate in (200, 250):
        steady = synthetic_long_axis("constant_angle_step_plus_noise", rate, 6, 81, angle_noise_sd_rad=0)
        assert np.allclose(np.diff(steady.baseline_angle_rad), 2 * np.pi * 8 / rate)
        noisy = synthetic_long_axis("constant_angle_step_plus_noise", rate, 6, 81)
        time = noisy.time_sec.to_numpy(float)
        _, omega = axis_angular_velocity(time, noisy.long_axis_deg.to_numpy(float))
        expected = 2 * np.pi * 8 + rate * np.diff(noisy.angle_noise_rad.to_numpy(float))
        assert np.allclose(omega[1:], expected, atol=1e-10)
        result = analyze(time, omega)
        medians = [result.frame[f"sd_per_mean_{width:g}s"].median() for width in (0.1, 0.2, 0.5, 1, 1.5, 2)]
        assert max(medians) / min(medians) < 1.2


def test_major_axis_mapping_from_opencv_ellipse():
    # With the long axis as the second returned dimension, OpenCV's angle
    # describes the perpendicular short axis.
    assert major_axis_deg(((0, 0), (30, 120), 90)) == 0
    assert major_axis_deg(((0, 0), (120, 30), 45)) == 45
    assert np.isnan(major_axis_deg(None))


def test_duplicate_source_map_marks_byte_identical_cells(tmp_path):
    a = tmp_path / "a.avi"
    b = tmp_path / "b.avi"
    c = tmp_path / "c.avi"
    a.write_bytes(b"synthetic source A")
    b.write_bytes(b"synthetic source B")
    c.write_bytes(b"synthetic source A")
    result = duplicate_source_map({"day1": [a, b], "day2": [c]})
    assert result == {("day1", 1): None, ("day1", 2): None, ("day2", 1): "day1/No.01"}


def test_long_axis_angle_plot_breaks_at_timestamp_gap():
    time = np.array([0.0, 0.01, 0.02, 0.20, 0.21, 0.22])
    axis = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    angle, omega = axis_angular_velocity(time, axis)
    assert np.isnan(angle[3])
    assert np.isnan(omega[3:5]).all()
    assert angle[4] == 0.0
    assert np.isfinite(omega[5])


def test_one_poster_png_contains_absolute_velocity_and_log_fft(tmp_path, monkeypatch):
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/09_poster_fluctuation.py"), run_name="poster_test"
    )
    source = synthetic_long_axis("constant_angle_step_plus_noise", 200, 6, 81)
    (tmp_path / "tables").mkdir()
    captured = []
    original_savefig = script["plt"].Figure.savefig

    def capture(figure, *args, **kwargs):
        captured.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(script["plt"].Figure, "savefig", capture)
    script["save_case"](source, tmp_path, "synthetic_No.01", "synthetic test", {"kind": "synthetic", "rate_hz": 200})
    assert [path.name for path in tmp_path.glob("*.png")] == ["synthetic_No.01.png"]
    assert (tmp_path / "tables/synthetic_No.01_long_axis_fft.csv").is_file()
    saved = script["pd"].read_csv(tmp_path / "tables/synthetic_No.01_time_series.csv")
    assert np.allclose(saved["long_axis_abs_omega_rad_s"], np.abs(saved["long_axis_omega_rad_s"]), equal_nan=True)
    assert saved["long_axis_phase_rad"].dropna().between(-np.pi, np.pi, inclusive="left").all()
    assert not any(column.endswith("_um") or column.startswith("orbit_") for column in saved)
    assert len(captured[0].axes) == 9
    assert len(captured[0].axes[1].lines) == 1
    assert len(captured[0].axes[2].lines) == 1
    assert captured[0].axes[5].get_yscale() == "log"
    assert captured[0].axes[5].get_xlim()[1] == 100
    assert captured[0].axes[5].get_xticks()[-1] == 100
    assert captured[0].axes[5].xaxis.label.get_fontsize() >= 14
    assert captured[0].axes[5].get_xticklabels()[0].get_fontsize() >= 12
    ratios = script["pd"].read_csv(tmp_path / "tables/synthetic_No.01_band_amplitude_ratio.csv")
    assert len(ratios) == 6
    assert np.allclose(ratios.low_minus_high_amplitude, ratios.low_mean_amplitude - ratios.high_mean_amplitude)
    assert captured[0].axes[6].get_yscale() == "log"
    assert captured[0].axes[7].get_yscale() == "linear"
    assert captured[0].axes[8].get_yscale() == "linear"
    assert len(captured[0].axes[6].lines[0].get_xdata()) == 6
    assert "stops retained" not in captured[0].axes[6].get_title()
    assert np.isnan(captured[0].axes[0].lines[0].get_ydata()).sum() > saved.long_axis_phase_rad.isna().sum()
    for width in (0.1, 0.2, 0.5, 1.0, 1.5, 2.0):
        assert f"long_axis_sd_per_mean_{width:g}s" in saved
        assert f"long_axis_sd_per_mean_{width:g}s_z" in saved


def test_main_routes_measured_plot_to_day_and_preserves_manual_selection(tmp_path, monkeypatch):
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/09_poster_fluctuation.py"), run_name="poster_test"
    )
    measured = tmp_path / "1_rotational_measurements"
    measured.mkdir()
    (measured / "selection.csv").write_text("manual decision\n", encoding="utf-8")
    source = synthetic_long_axis("constant_angle_step_plus_noise", 200, 6, 81)
    source["x_um"] = 0.5 * np.cos(source.baseline_angle_rad)
    source["y_um"] = -0.5 * np.sin(source.baseline_angle_rad)
    source = source[["time_sec", "x_um", "y_um", "long_axis_deg", "axis_ratio"]]
    captured = []
    original_savefig = script["plt"].Figure.savefig

    def capture(figure, *args, **kwargs):
        captured.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(script["plt"].Figure, "savefig", capture)
    script["main"].__globals__["read_avi"] = lambda _path, _settings: source
    script["main"].__globals__["duplicate_source_map"] = lambda _files: {("20230607", 1): None}
    script["main"](
        Namespace(mode="measured", days=["20230607"]),
        {"20230607": [tmp_path / "No.01.avi"]},
        {"20230607": {"rate_hz": 200, "duration_s": 6}},
        tmp_path,
    )
    assert (measured / "20230607/No.01.png").is_file()
    assert (measured / "20230607/tables/No.01_time_series.csv").is_file()
    assert (measured / "20230607/tables/No.01_band_amplitude_ratio.csv").is_file()
    assert not (measured / "20230607/tables/No.01_orbit_fft.csv").exists()
    measured_series = script["pd"].read_csv(measured / "20230607/tables/No.01_time_series.csv")
    assert not any(column.startswith("orbit_") for column in measured_series.columns)
    measured_metadata = json.loads((measured / "20230607/tables/No.01_metadata.json").read_text())
    assert not any(key.startswith("orbit_") for key in measured_metadata)
    assert (measured / "selection.csv").read_text(encoding="utf-8") == "manual decision\n"
    index = script["pd"].read_csv(measured / "index.csv")
    assert index.loc[0, "plot_file"] == "20230607/No.01.png"
    assert index.loc[0, "band_ratio_file"] == "20230607/tables/No.01_band_amplitude_ratio.csv"
    assert len(captured[0].axes) == 12
    assert captured[0].axes[9].get_yscale() == "log"
    assert len(captured[0].axes[9].lines[0].get_xdata()) == 6
    assert captured[0].axes[10].get_yscale() == "linear"
    assert captured[0].axes[11].get_yscale() == "linear"
    ratios = script["pd"].read_csv(measured / "20230607/tables/No.01_band_amplitude_ratio.csv")
    assert np.allclose(ratios.low_minus_high_amplitude, ratios.low_mean_amplitude - ratios.high_mean_amplitude)
    assert not any(column.startswith("orbit_") for column in index.columns)
    assert not (tmp_path / "index.csv").exists()


def test_candidate_band_comparison_keeps_four_cells_separate(tmp_path, monkeypatch):
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/09_poster_fluctuation.py"), run_name="poster_test"
    )
    rows = []
    for day, sample_no in script["POSTER_CANDIDATES"]:
        for width in (0.1, 0.2, 0.5, 1.0, 1.5, 2.0):
            rows.append(
                {
                    "day": day,
                    "sample_no": sample_no,
                    "window_s": width,
                    "low_to_high_amplitude_ratio": 10 * width,
                    "low_mean_amplitude": width,
                    "low_minus_high_amplitude": width / 2,
                }
            )
    captured = []
    original_savefig = script["plt"].Figure.savefig

    def capture(figure, *args, **kwargs):
        captured.append(figure)
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(script["plt"].Figure, "savefig", capture)
    target = tmp_path / "candidate_band_amplitude_ratio.png"
    script["candidate_band_comparison"](pd.DataFrame(rows), target)
    assert target.is_file()
    assert len(captured[0].axes) == 3
    assert all(len(ax.lines) == 4 for ax in captured[0].axes)
    assert captured[0].axes[0].get_yscale() == "log"
    assert all(ax.get_yscale() == "linear" for ax in captured[0].axes[1:])
    assert "stops included" in captured[0].axes[0].get_legend().get_texts()[2].get_text()


def test_main_routes_synthetic_plots_and_index(tmp_path, monkeypatch):
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/09_poster_fluctuation.py"), run_name="poster_test"
    )
    limits = []
    original_savefig = script["plt"].Figure.savefig

    def capture(figure, *args, **kwargs):
        if len(figure.axes) == 9:
            limits.append(figure.axes[5].get_xlim()[1])
            assert figure.axes[5].get_xticks()[-1] == limits[-1]
            assert figure.axes[6].get_yscale() == "log"
            assert all(ax.get_yscale() == "linear" for ax in figure.axes[7:])
            if "random_angle_noise" in figure._suptitle.get_text():
                assert figure.axes[0].get_ylim()[1] < 0.1
        return original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(script["plt"].Figure, "savefig", capture)
    simulation = tmp_path / "2_simulation"
    (simulation / "tables").mkdir(parents=True)
    (simulation / "simulation_steady_rotation_plus_noise_200hz_seed81.png").write_bytes(b"obsolete")
    (simulation / "tables/simulation_random_only_200hz_seed81_metadata.json").write_text("obsolete")
    script["main"](
        Namespace(mode="synthetic", synthetic_rates=[200, 250], synthetic_duration=3, seed=81, null_replicates=1),
        {},
        {},
        tmp_path,
    )
    assert (simulation / "simulation_constant_angle_step_plus_noise_200hz_seed81.png").is_file()
    assert (simulation / "simulation_random_angle_noise_200hz_seed81.png").is_file()
    assert (simulation / "simulation_constant_angle_step_plus_noise_250hz_seed81.png").is_file()
    assert (simulation / "simulation_random_angle_noise_250hz_seed81.png").is_file()
    assert (simulation / "simulation_random_angle_noise_ensemble.png").is_file()
    assert (simulation / "tables/simulation_random_angle_noise_ensemble.csv").is_file()
    assert not list(simulation.glob("simulation_steady_rotation_plus_noise_*.png"))
    assert not list((simulation / "tables").glob("simulation_random_only_*"))
    index = script["pd"].read_csv(simulation / "index.csv")
    assert set(index.plot_file) == {
        "simulation_constant_angle_step_plus_noise_200hz_seed81.png",
        "simulation_random_angle_noise_200hz_seed81.png",
        "simulation_constant_angle_step_plus_noise_250hz_seed81.png",
        "simulation_random_angle_noise_250hz_seed81.png",
    }
    assert len(index.band_ratio_file) == 4
    assert all((simulation / path).is_file() for path in index.band_ratio_file)
    assert not any(column.startswith("orbit_") or column == "coordinate_noise_sd_um" for column in index)
    nulls = script["pd"].read_csv(simulation / "tables/simulation_random_angle_noise_ensemble.csv")
    assert set(nulls.nyquist_hz) == {100, 125}
    assert "peak_to_median_full_band" in nulls and "peak_to_median_0_20hz" not in nulls
    assert limits == [100, 100, 125, 125]
    assert not (tmp_path / "1_rotational_measurements").exists()


def test_wrapped_phase_preserves_observations_and_breaks_only_at_wrap():
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/09_poster_fluctuation.py"), run_name="poster_test"
    )
    time = np.arange(5, dtype=float)
    phase = np.array([np.pi - 0.1, -np.pi + 0.1, -np.pi + 0.2, np.nan, 0.0])
    plotted_time, plotted_phase = script["wrapped_phase_plot"](time, phase)
    assert np.allclose(plotted_time[np.isfinite(plotted_time)], time)
    assert np.allclose(plotted_phase[np.isfinite(plotted_phase)], phase[np.isfinite(phase)])
    assert np.isnan(plotted_phase[1])


def test_random_only_peak_contrast_uses_full_positive_band_below_nyquist():
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/09_poster_fluctuation.py"), run_name="poster_test"
    )
    spectrum = pd.DataFrame({"frequency_hz": [0, 1, 30, 90, 100], "amplitude": [100, 1, 8, 2, 100]})
    summary = script["null_spectrum_summary"](spectrum, 200)
    assert summary["nyquist_hz"] == 100
    assert summary["peak_frequency_hz"] == 30
    assert summary["peak_amplitude"] == 8
    assert summary["peak_to_median_full_band"] == 4
