import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg", force=True)

from utils import param
from utils.functions import repellent_response
from utils.functions.validated_rotation_center import (
    assess_center_window,
    estimate_center_series,
)


def _orbit(seed: int, *, short_arc: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phase = np.linspace(-0.3, 0.3, 360) if short_arc else np.linspace(0, 12 * np.pi, 360, endpoint=False)
    return (
        np.array([1.3, 1.45])
        + np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
        + rng.normal(0, 0.02, (360, 2))
    )


def test_full_orbit_center_is_accepted_and_translation_equivariant():
    points = _orbit(4000)
    fit = assess_center_window(points, 200)
    shifted = assess_center_window(points + [2, -1], 200)
    assert fit["gate_pass"] and shifted["gate_pass"]
    assert np.linalg.norm(np.array([fit["center_x_um"], fit["center_y_um"]]) - [1.3, 1.45]) < 0.03
    assert np.allclose(
        [shifted["center_x_um"], shifted["center_y_um"]],
        [fit["center_x_um"] + 2, fit["center_y_um"] - 1],
        atol=0.003,
    )


def test_short_arc_and_missing_window_cannot_supply_angular_velocity_center():
    short_arc = assess_center_window(_orbit(4000, short_arc=True), 200)
    missing_points = _orbit(4000)
    missing_points[100] = np.nan
    missing = assess_center_window(missing_points, 200)
    for result in (short_arc, missing):
        assert not result["gate_pass"]
        assert np.isnan(result["center_x_um"])
        assert np.isnan(result["center_y_um"])
    assert missing["gate_failure"] == "missing"


def test_center_series_preserves_frames_and_invalidates_stop_and_unsupported_ends():
    n = 1200
    time = np.arange(n) / 200
    phase = 2 * np.pi * np.arange(n) / 60
    rng = np.random.default_rng(5)
    points = [1.3, 1.45] + np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
    points += rng.normal(0, 0.02, points.shape)
    cx, cy, assessments = repellent_response.estimate_rotation_center_validated(
        [time.tolist()], [points[:, 0].tolist()], [points[:, 1].tolist()], stop_indices=[900]
    )
    assert len(cx[0]) == len(cy[0]) == n
    assert np.isfinite(cx[0][500]) and np.isfinite(cy[0][500])
    assert np.isnan(cx[0][0]) and np.isnan(cx[0][899])
    assert np.isnan(cx[0][900:]).all() and np.isnan(cy[0][900:]).all()
    assert any(row["gate_pass"] for row in assessments)
    assert all(row["anchor_frame_0based"] < 900 for row in assessments)


def test_existing_angular_velocity_csv_keeps_invalid_frames(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    n = 1200
    time = np.arange(n) / 200
    phase = 2 * np.pi * np.arange(n) / 60
    rng = np.random.default_rng(6)
    points = [1.3, 1.45] + np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
    points += rng.normal(0, 0.02, points.shape)
    cx, cy, assessments = repellent_response.estimate_rotation_center_validated(
        [time.tolist()], [points[:, 0].tolist()], [points[:, 1].tolist()], stop_indices=[900]
    )
    repellent_response.save_segment_center_coordinate("sample", "00_all_rotational_analysis", cx, cy)
    repellent_response.save_center_assessments("sample", assessments)
    corrected = points - np.column_stack((cx[0], cy[0]))
    result = repellent_response.run_segment_rotational_analysis(
        "sample",
        "00_all_rotational_analysis",
        [time.tolist()],
        [corrected[:, 0].tolist()],
        [corrected[:, 1].tolist()],
        write_time_list=False,
    )
    assert np.nanmedian(result["signed_angular_velocity_list"][0]) > 0  # image-space clockwise
    repellent_response.save_segment_angular_velocity_outputs(
        "sample",
        "00_all_rotational_analysis",
        result["time_list"],
        result["angle_list"],
        result["angular_velocity_list"],
        original_sample_indices=result["valid_indices"],
        write_time_list=False,
    )
    base = tmp_path / "sample/repellent_response/00_all_rotational_analysis"
    center_csv = pd.read_csv(base / "center_coordinate/center_coordinate.csv")
    av_csv = pd.read_csv(base / "angular_velocity/angular-velocity_time-series.csv")
    quality_csv = pd.read_csv(base / "center_coordinate/center_assessment.csv")
    assert center_csv.columns.tolist() == ["No.1_x", "No.1_y"]
    assert av_csv.columns.tolist() == ["No.1"]
    assert len(center_csv) == n and len(av_csv) == n - 1
    assert np.isfinite(av_csv["No.1"].iloc[500])
    assert np.isnan(av_csv["No.1"].iloc[0]) and np.isnan(av_csv["No.1"].iloc[900:]).all()
    assert {"sample_no", "anchor_frame_0based", "gate_pass", "gate_failure"} <= set(quality_csv.columns)


def test_switching_count_does_not_bridge_an_invalid_center_gap():
    time = [(np.arange(6) * 0.1).tolist()]
    velocity = [[-2.0, -1.0, np.nan, 2.0, 1.0]]
    _, count = repellent_response.calculate_angular_velocity_switching_count(time, velocity)
    assert count[0][0] == 0.0
    _, all_invalid = repellent_response.calculate_angular_velocity_switching_count(time, [[np.nan] * 5])
    assert np.isnan(all_invalid[0]).all()


def test_fully_rejected_sample_remains_in_angular_velocity_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    n = 360
    time = np.arange(n) / 200
    rng = np.random.default_rng(8)
    points = [1.3, 1.45] + rng.normal(0, 0.008, (n, 2))
    cx, cy, _ = repellent_response.estimate_rotation_center_validated(
        [time.tolist()], [points[:, 0].tolist()], [points[:, 1].tolist()]
    )
    assert np.isnan(cx[0]).all() and np.isnan(cy[0]).all()
    result = repellent_response.run_segment_rotational_analysis(
        "sample", "00_all_rotational_analysis", [time.tolist()], [cx[0]], [cy[0]], write_time_list=False
    )
    assert result["valid_indices"] == [0]
    assert len(result["angular_velocity_list"][0]) == n - 1
    assert np.isnan(result["angular_velocity_list"][0]).all()


def test_empty_assessment_has_stable_csv_header(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    repellent_response.save_center_assessments("sample", [])
    path = tmp_path / "sample/repellent_response/00_all_rotational_analysis/center_coordinate/center_assessment.csv"
    assert pd.read_csv(path).columns.tolist() == [
        "sample_no",
        "anchor_frame_0based",
        "anchor_time_sec",
        "gate_pass",
        "gate_failure",
    ]


def test_repeated_short_arc_does_not_become_a_false_center():
    n = 1200
    time = np.arange(n) / 200
    frames = np.arange(n)
    phase = np.deg2rad(60) * (2 / np.pi) * np.arcsin(np.sin(np.pi * (frames - 30) / 60))
    rng = np.random.default_rng(7100)
    points = [1.3, 1.45] + np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
    points += rng.normal(0, 0.02, points.shape)
    centers, records = estimate_center_series(time, points)
    assert not np.isfinite(centers).any()
    assert any("angular_coverage" in row.get("gate_failure", "") for row in records)


def test_full_turn_reversals_remain_usable_despite_cross_spectrum_cancellation():
    n = 1200
    time = np.arange(n) / 200
    frames = np.arange(n)
    phase = np.pi * (2 / np.pi) * np.arcsin(np.sin(np.pi * (frames - 30) / 60))
    rng = np.random.default_rng(7100)
    points = [1.3, 1.45] + np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
    points += rng.normal(0, 0.02, points.shape)
    centers, _ = estimate_center_series(time, points)
    valid = np.isfinite(centers).all(axis=1)
    assert valid[200:1000].mean() > 0.9
    assert np.quantile(np.linalg.norm(centers[valid] - [1.3, 1.45], axis=1) / 0.3, 0.95) < 0.1


def test_missing_frames_do_not_poison_distant_valid_windows():
    n = 1200
    time = np.arange(n) / 200
    phase = 2 * np.pi * np.arange(n) / 60
    points = [1.3, 1.45] + np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
    points[520:560] = np.nan
    centers, _ = estimate_center_series(time, points)
    assert np.isnan(centers[520:560]).all()
    assert np.isfinite(centers[250:400]).any()
    assert np.isfinite(centers[800:950]).any()


def test_small_radius_with_large_relative_noise_is_not_reported_as_a_precise_center():
    n = 1200
    time = np.arange(n) / 200
    phase = 2 * np.pi * np.arange(n) / 60
    size = np.ones(n)
    size[n // 2 :] = 0.2
    rng = np.random.default_rng(8105)
    points = [1.3, 1.45] + size[:, None] * np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
    points += rng.normal(0, 0.02, points.shape)
    centers, records = estimate_center_series(time, points)
    assert np.isnan(centers[800:1100]).all()
    assert any("orbit_residual" in row.get("gate_failure", "") for row in records)
