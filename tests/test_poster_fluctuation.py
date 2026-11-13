"""Synthetic contracts for the poster measurement path."""

import runpy
from pathlib import Path

import numpy as np

from utils.functions.poster_fluctuation import (
    analyze,
    axis_angular_velocity,
    duplicate_source_map,
    major_axis_deg,
    orbit_angular_velocity,
    synthetic_trajectory,
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


def test_long_axis_and_orbit_sign_on_known_ccw_motion():
    frame = synthetic_trajectory("periodic_plus_noise", 200, 6, 81, noise_rad_s=0, coordinate_noise_sd_um=0)
    time = frame.time_sec.to_numpy(float)
    axis = (-np.rad2deg(frame.true_angle_rad.to_numpy(float))) % 180
    _, omega_axis = axis_angular_velocity(time, axis)
    _, omega_orbit, center = orbit_angular_velocity(time, frame.x_um.to_numpy(float), frame.y_um.to_numpy(float))
    truth = frame.true_angular_velocity_rad_s.to_numpy(float)
    assert np.allclose(omega_axis[1:], truth[1:], atol=1e-10)
    assert np.allclose(omega_orbit[1:], truth[1:], atol=1e-10)
    assert np.allclose(center, [0, 0], atol=1e-10)


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


def test_random_only_control_has_no_injected_periodic_component():
    frame = synthetic_trajectory("random_only", 200, 6, 81, base_hz=0)
    assert len(frame) == 1200
    assert frame.true_angular_velocity_rad_s.std() > 0
    assert abs(frame.true_angular_velocity_rad_s.mean()) < 0.5
    assert np.isfinite(frame[["x_um", "y_um"]]).all().all()


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


def test_one_flat_poster_png_contains_all_window_series(tmp_path):
    script = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "scripts/09_poster_fluctuation.py"), run_name="poster_test"
    )
    source = synthetic_trajectory("periodic_plus_noise", 200, 6, 81)
    source["long_axis_deg"] = (-np.rad2deg(source["true_angle_rad"])) % 180.0
    source["axis_ratio"] = np.nan
    script["save_case"](source, tmp_path, "synthetic_No.01", "synthetic test", {"kind": "synthetic"})
    assert [path.name for path in tmp_path.glob("*.png")] == ["synthetic_No.01.png"]
    assert (tmp_path / "tables/synthetic_No.01_long_axis_fft.csv").is_file()
    saved = script["pd"].read_csv(tmp_path / "tables/synthetic_No.01_time_series.csv")
    for width in (0.1, 0.2, 0.5, 1.0, 1.5, 2.0):
        assert f"long_axis_sd_per_mean_{width:g}s" in saved
        assert f"long_axis_sd_per_mean_{width:g}s_z" in saved
