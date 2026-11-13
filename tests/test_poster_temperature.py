"""Synthetic checks for the local temperature-control measurement path."""

import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from utils.functions.poster_temperature import (
    TemperatureSources,
    analyze_cell,
    cell_coordinates,
    cell_summary,
    orbit_kinematics,
)


def _source(time, centered_x, centered_y, center_x=None, center_y=None, before_x=None):
    n = len(time)
    center_x = np.full(n, 5.0) if center_x is None else np.asarray(center_x, dtype=float)
    center_y = np.full(n, 7.0) if center_y is None else np.asarray(center_y, dtype=float)
    before_x = center_x if before_x is None else np.asarray(before_x, dtype=float)
    return TemperatureSources(
        temperature_c=23,
        frame_rate_hz=200,
        sample_count=1,
        time=pd.DataFrame({"No.1": time}),
        centered_centroid=pd.DataFrame({"x_1": centered_x, "y_1": centered_y}),
        corrected_center=pd.DataFrame({"No.1_x": center_x, "No.1_y": center_y}),
        original_center=pd.DataFrame({"No.1_x": before_x, "No.1_y": center_y}),
    )


def test_reconstructs_absolute_centroid_and_tracks_center_adjustment():
    source = _source([0, 0.005, 0.01], [1, 0, -1], [0, -1, 0], before_x=[5, 4, 5])
    frame = cell_coordinates(source, 1)
    assert frame.source_frame_1based.tolist() == [1, 2, 3]
    assert frame.centroid_x_um.tolist() == [6, 5, 4]
    assert frame.centroid_y_um.tolist() == [7, 6, 7]
    assert frame.radius_um.tolist() == [1, 1, 1]
    assert frame.source_center_adjusted.tolist() == [False, True, False]


def test_orbit_velocity_is_ccw_positive_and_never_crosses_timestamp_gap():
    time = np.array([0, 0.005, 0.01, 0.1, 0.105, 0.11])
    angle = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    source = _source(time, np.cos(angle), -np.sin(angle))
    frame = orbit_kinematics(cell_coordinates(source, 1))
    assert np.isnan(frame.orbit_omega_rad_s.iloc[0])
    assert np.allclose(frame.orbit_omega_rad_s.iloc[[1, 2, 4, 5]], 40)
    assert np.isnan(frame.orbit_omega_rad_s.iloc[3])
    assert np.allclose(frame.orbit_abs_omega_rad_s.iloc[[1, 2, 4, 5]], 40)
    assert frame.orbit_angle_rad.iloc[5] == np.arctan2(-(-np.sin(1.0)), np.cos(1.0))


def test_missing_coordinate_blocks_two_adjacent_velocity_rows():
    time = np.arange(6) / 200
    x = np.ones(6)
    x[2] = np.nan
    frame = orbit_kinematics(cell_coordinates(_source(time, x, np.zeros(6)), 1))
    assert np.isnan(frame.orbit_omega_rad_s.iloc[[0, 2, 3]]).all()
    assert np.isfinite(frame.orbit_omega_rad_s.iloc[[1, 4, 5]]).all()


def test_full_cell_analysis_preserves_frames_and_sliding_window():
    rate = 200
    time = np.arange(6 * rate) / rate
    angle = 2 * np.pi * 7 * time + 0.003 * np.sin(2 * np.pi * 3 * time)
    source = _source(time, np.cos(angle), -np.sin(angle))
    frame, spectrum, ratios, metadata = analyze_cell(cell_coordinates(source, 1))
    assert len(frame) == len(time)
    assert frame.loc[(time >= 2) & (time < 3), "sd_per_mean_0.1s"].notna().sum() == rate
    assert len(metadata) == 6
    assert ratios.window_s.tolist() == [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]
    assert spectrum.frequency_hz.max() < rate / 2
    summary = cell_summary(frame, ratios, 23, 1)
    assert abs(summary["mean_rotation_radius_um"] - 1) < 1e-10
    assert abs(summary["mean_absolute_angular_velocity_rad_s"] - 2 * np.pi * 7) < 0.5


def test_cell_and_comparison_plots_accept_synthetic_data(tmp_path):
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/10_poster_temperature_control.py"),
        run_name="temperature_test",
    )
    plot_t, plot_y = script["_with_gap_breaks"](np.array([0, 0.005, 0.01, 0.1, 0.105]), np.arange(5, dtype=float))
    assert len(plot_t) == 6
    assert np.isnan(plot_t[3]) and np.isnan(plot_y[3])
    assert plot_y[4] == 3
    time = np.arange(6 * 200) / 200
    angle = 2 * np.pi * 7 * time + 0.003 * np.sin(2 * np.pi * 3 * time)
    frame, spectra, ratios, _ = analyze_cell(cell_coordinates(_source(time, np.cos(angle), -np.sin(angle)), 1))
    summary = cell_summary(frame, ratios, 23, 1)
    script["cell_figure"](frame, spectra, ratios, summary, tmp_path / "cell.png", 200)
    assert (tmp_path / "cell.png").stat().st_size > 1000
    rows = []
    groups = []
    for temp in (10, 23, 30):
        row = {**summary, "temperature_c": temp}
        rows.append(row)
        groups.append(ratios.assign(temperature_c=temp, sample_no=1))
    script["comparison_figures"](pd.DataFrame(rows), pd.concat(groups), tmp_path)
    assert len(list(tmp_path.glob("compare_*.png"))) == 4
