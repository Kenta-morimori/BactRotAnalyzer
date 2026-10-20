"""Production integration: preserve abstention, frame identity, and point times."""

import importlib.util
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

from tests.synthetic.audit import AuditCase, generate
from tests.synthetic.sign import StudyCase
from utils import param
from utils.functions import repellent_response, repellent_sign_reference


def test_production_adapter_preserves_frozen_signs_and_missing_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))

    def no_baseline(*args, **kwargs):
        raise AssertionError("V2 must not recompute the old method")

    monkeypatch.setattr(repellent_response, "estimate_rotation_center_validated", no_baseline)
    data = generate(AuditCase("stop", StudyCase("stop", 0.35, perturbation="dwell_drift_resume")), 10200)
    cx, cy, result = repellent_sign_reference.analyze(
        "study", [data.time], [data.observed[:, 0]], [data.observed[:, 1]], source_starts=[201]
    )
    frame = pd.read_csv(tmp_path / "study/repellent_response/rotation_center/no01/timeseries.csv")
    frozen = repellent_sign_reference.estimate_sign_reference_v2(data.time, data.observed)
    assert np.allclose(frame.rotation_sign, frozen.rotation_sign, equal_nan=True)
    assert np.allclose(np.sign(result["signed_angular_velocity_list"][0]), frame.rotation_sign, equal_nan=True)
    assert len(result["time_list"][0]) == len(result["angular_velocity_list"][0]) == len(frame)
    assert np.array_equal(frame.source_frame_1based, frame.frame_0based + 201)
    stopped = frame.time_sec.between(2.55, 3.45)
    assert frame.loc[stopped, "angular_velocity_rad_s"].isna().all()
    assert frame.loc[stopped, "reference_x_um"].isna().all()
    anchors = frame.frame_0based.to_numpy(int)
    assert np.allclose(np.array(cx[0])[anchors], frame.reference_x_um, equal_nan=True)
    assert np.isnan(np.array(cx[0])[np.setdiff1d(np.arange(len(data.time)), anchors)]).all()


def test_image_space_clockwise_sign_and_cw_rate():
    time = np.arange(1200) / 200
    phase = 2 * np.pi * 20 * time
    for y_direction, expected_sign, expected_cw_rate in ((1, 1, 1.0), (-1, -1, 0.0)):
        points = np.column_stack((1.3 + 0.3 * np.cos(phase), 1.45 + y_direction * 0.18 * np.sin(phase)))
        _, frame, _ = repellent_sign_reference.analyze_sample(time, points, stop_measurement_enabled=False)
        reported = frame.rotation_sign.notna()
        assert reported.sum() > 100
        assert frame.loc[reported, "rotation_sign"].eq(expected_sign).all()
        assert frame.loc[reported, "angular_velocity_rad_s"].mul(expected_sign).gt(0).all()
        _, cw_rate = repellent_response.calculate_angular_velocity_cw_rate(
            [frame.time_sec.tolist()], [frame.angular_velocity_rad_s.tolist()]
        )
        assert np.allclose(np.asarray(cw_rate[0])[np.isfinite(cw_rate[0])], expected_cw_rate)


def test_point_sampled_pre_post_split_uses_stimulus_time_and_keeps_nan():
    args: dict[str, Any] = dict(
        time_list=[[10, 10.025, 10.05, 10.075]],
        angle_list=[[0, np.nan, 1, 2]],
        angular_velocity_list=[[3, np.nan, -4, -5]],
        valid_indices=[0],
        rise_indices=[2000],
        rise_time_list=[10.05],
    )
    pre = repellent_response.split_rotational_series_by_rise(**args, mode="pre")
    post = repellent_response.split_rotational_series_by_rise(**args, mode="post")
    assert pre[0] == [[10, 10.025]] and np.isnan(pre[2][0][1])
    assert post[0] == [[10.05, 10.075]] and post[2] == [[-4, -5]]


def test_switching_does_not_bridge_timestamp_gap():
    _, counts = repellent_response.calculate_angular_velocity_switching_count(
        [[0, 0.025, 0.05, 0.6, 0.625]], [[1, 1, 1, -1, -1]]
    )
    assert counts[0][0] == 0


def test_default_entry_point_selects_v2():
    path = Path(__file__).parents[1] / "scripts/04_repellent_response_analysis.py"
    spec = importlib.util.spec_from_file_location("repellent_entry", path)
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    import inspect

    assert inspect.signature(entry.main).parameters["center_estimator"].default is None
    assert param.center_estimator == "v2"
    assert param.stop_measurement_enabled is True
    assert entry.resolve_center_options(None, None) == ("v2", True)
    assert entry.resolve_center_options("validated", False) == ("validated", False)
    assert entry.resolve_center_options("legacy", True) == ("legacy", True)


def test_stop_measurements_never_fill_primary_and_can_be_disabled():
    from tests.synthetic.stops import Case
    from tests.synthetic.stops import generate as stopped_data

    data = stopped_data(Case("quiet"), 10400)
    centers, enabled, episodes = repellent_sign_reference.analyze_sample(data["time"], data["observed"])
    _, disabled, no_episodes = repellent_sign_reference.analyze_sample(
        data["time"], data["observed"], stop_measurement_enabled=False
    )
    measured = enabled.stop_angular_velocity_rad_s.notna()
    assert measured.any() and not episodes.empty
    assert enabled.loc[measured, "rotation_sign"].isna().all()
    assert enabled.loc[measured, "angular_velocity_rad_s"].isna().all()
    assert np.isnan(centers[enabled.loc[measured, "frame_0based"].to_numpy(int)]).all()
    assert np.allclose(enabled.angular_velocity_rad_s, disabled.angular_velocity_rad_s, equal_nan=True)
    assert disabled.stop_angular_velocity_rad_s.isna().all() and no_episodes.empty
    assert disabled.measurement_reason.eq("disabled").all()


def test_short_record_has_stable_empty_output_schema():
    _, frame, episodes = repellent_sign_reference.analyze_sample(np.arange(3) * 0.005, np.zeros((3, 2)))
    assert frame.empty and episodes.empty
    assert {"segment_id", "measurement_reason", "stop_angular_velocity_rad_s"} <= set(frame.columns)
    assert "start_s" in episodes


def test_param_changes_and_explicit_cli_settings(monkeypatch):
    path = Path(__file__).parents[1] / "scripts/04_repellent_response_analysis.py"
    spec = importlib.util.spec_from_file_location("repellent_options", path)
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(param, "center_estimator", "legacy")
    monkeypatch.setattr(param, "stop_measurement_enabled", False)
    assert entry.resolve_center_options(None, None) == ("legacy", False)
    assert entry.resolve_center_options("v2", True) == ("v2", True)
    parser = entry.build_parser()
    args = parser.parse_args(["--center-estimator", "validated", "--no-stop-measurement"])
    assert entry.resolve_center_options(args.center_estimator, args.stop_measurement) == ("validated", False)


def test_empty_sample_saves_missing_coverage_and_diagnostic_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    repellent_sign_reference.analyze("short", [np.arange(3) * 0.005], [[0, 0, 0]], [[0, 0, 0]])
    folder = tmp_path / "short/repellent_response/rotation_center"
    summary = pd.read_csv(folder / "summary.csv")
    assert summary.evaluation_points.iloc[0] == 0
    assert np.isnan(summary.primary_coverage.iloc[0])
    assert (folder / "no01/overview.png").is_file()
    assert (folder / "no01/diagnostics.png").is_file()
    assert "measurement_reason" in pd.read_csv(folder / "no01/timeseries.csv")
