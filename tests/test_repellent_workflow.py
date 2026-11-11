"""End-to-end inference contracts on synthetic original observations."""

import numpy as np
import pandas as pd
import pytest

from utils.functions import repellent_workflow as workflow
from utils.functions.orbit_center.provisional_center import configuration


def observations():
    t = np.arange(161) * 0.005
    center = np.c_[2 + 0.2 * t, 3 - 0.1 * t]
    xy = center + np.c_[np.cos(2 * np.pi * 24 * t), -0.5 * np.sin(2 * np.pi * 24 * t)]
    frame = pd.DataFrame(
        dict(source_frame_1based=np.arange(161) + 1, time_sec=t, x_um=xy[:, 0], y_um=xy[:, 1], detected=True)
    )
    return frame, center


def test_raw_input_to_center_and_ccw_velocity_preserves_original_rows():
    raw, truth = observations()
    saved = raw.copy(deep=True)
    frame, support, method = workflow.analyze_sample(
        raw,
        sample_no=7,
        source_id="synthetic",
        centroid_method="synthetic",
        frame_period_s=0.005,
        timestamp_quantum_s=1e-6,
        identity="known synthetic target",
        parameters=configuration()["parameters"],
    )
    pd.testing.assert_frame_equal(raw, saved)
    assert len(frame) == len(raw) and frame.sample_no.eq(7).all()
    np.testing.assert_array_equal(frame.source_frame_1based, raw.source_frame_1based)
    np.testing.assert_array_equal(frame.time_sec, raw.time_sec)
    np.testing.assert_allclose(frame[["center_x_um", "center_y_um"]], truth, atol=1e-7)
    assert np.isnan(frame.angular_velocity_rad_s.iloc[0])
    assert frame.angular_velocity_rad_s.iloc[1:].gt(0).all()
    assert len(support) == len(raw)
    assert method["identity"] == "known synthetic target"


def test_target_change_missing_and_reacquisition_break_measurements():
    raw, _ = observations()
    raw["target_id"] = ["a"] * 80 + ["b"] * 81
    raw["reacquired"] = False
    raw.loc[100, "reacquired"] = True
    raw.loc[120, ["x_um", "y_um"]] = np.nan
    frame, _, _ = workflow.analyze_sample(
        raw,
        sample_no=1,
        source_id="synthetic",
        centroid_method="synthetic",
        frame_period_s=0.005,
        timestamp_quantum_s=1e-6,
        identity="known",
        parameters=configuration()["parameters"],
    )
    for col in ["angular_velocity_rad_s", "uncompleted_angular_velocity_rad_s"]:
        assert frame[col].iloc[[80, 100, 101, 120, 121]].isna().all()
    assert frame[["center_x_um", "center_y_um"]].iloc[[100, 120]].isna().all().all()


def test_unknown_identity_and_duplicate_frames_are_rejected():
    raw, _ = observations()
    kwargs = dict(
        sample_no=1, source_id="s", centroid_method="synthetic", frame_period_s=0.005, timestamp_quantum_s=1e-6
    )
    with pytest.raises(ValueError, match="identity"):
        workflow.analyze_sample(raw, identity="", **kwargs)
    raw.loc[1, "source_frame_1based"] = 1
    with pytest.raises(ValueError, match="source frames"):
        workflow.analyze_sample(raw, identity="known", **kwargs)


def test_background_absent_leaves_pre_post_metrics_unavailable():
    raw, _ = observations()
    frame, _, _ = workflow.analyze_sample(
        raw,
        sample_no=1,
        source_id="s",
        centroid_method="synthetic",
        frame_period_s=0.005,
        timestamp_quantum_s=1e-6,
        identity="known",
    )
    metrics, features = workflow.measurement_metrics(frame, {})
    assert all(value is None for value in metrics.values()) and features is None


def test_normal_day_route_keeps_configured_source_scope_and_background_boundary(tmp_path, monkeypatch):
    from utils import param
    from utils.functions import raw_centroid, repellent_response

    raw, _ = observations()
    monkeypatch.setattr(param, "input_dir_bef", str(tmp_path))
    monkeypatch.setattr(param, "get_analysis_frame_ranges_config", lambda day: [(21, 141)])
    monkeypatch.setattr(repellent_response, "ensure_time_list", lambda day: [raw.time_sec.to_list()])
    monkeypatch.setattr(raw_centroid, "load_all", lambda day, clocks: [raw])
    monkeypatch.setattr(repellent_response, "get_background_intensity_time_series", lambda *a, **k: [np.ones(len(raw))])
    monkeypatch.setattr(repellent_response, "detect_rise_points", lambda **k: [dict(rise_index=60)])
    monkeypatch.setattr(repellent_response, "add_rise_time_to_results", lambda *a: None)
    monkeypatch.setattr(repellent_response, "get_tiff_sample_names", lambda day: ["synthetic"])
    monkeypatch.setattr(
        workflow, "acquisition_policy", lambda path: dict(frame_period_s=0.005, timestamp_quantum_s=1e-6)
    )
    folder = tmp_path / "day/tiff_data/synthetic"
    folder.mkdir(parents=True)
    (folder / "first.tif").write_bytes(b"synthetic fixture; metadata reader tested separately")
    bundle, features = workflow.recalculate_day("day")
    frame = bundle["observations"]
    assert frame.source_frame_1based.tolist() == list(range(21, 142))
    np.testing.assert_array_equal(frame.time_sec, raw.time_sec.iloc[20:141])
    assert bundle["settings"]["rise_source_frames"] == {"1": 81}
    assert set(bundle["settings"]["runtime_versions"]) == {"numpy", "pandas", "scipy"}
    assert all(row.startswith("utils/") for row in bundle["settings"]["code_sha256"])
    assert features is not None


def test_cli_rejects_output_containing_a_measurement_input(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    from tests.test_repellent_result_export import observations as result_observations
    from tests.test_repellent_result_export import settings
    from utils import param

    path = Path(__file__).parents[1] / "scripts/04_repellent_response_analysis.py"
    spec = importlib.util.spec_from_file_location("input_separation_entry", path)
    assert spec is not None and spec.loader is not None
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    source = tmp_path / "day/repellent_response/01_brightness_change/background.csv"
    source.parent.mkdir(parents=True)
    source.write_text("original measured input")
    options = settings()
    options["input_hashes"] = {str(source): "synthetic"}
    monkeypatch.setattr(
        workflow,
        "recalculate_manifest",
        lambda path: (dict(observations=result_observations(), settings=options), None),
    )
    with pytest.raises(ValueError, match="output contains a measurement input"):
        entry.main("day", 0.5, 3, 3, 1, 0.2, 1, None, measurement_input="synthetic.json")
    assert source.read_text() == "original measured input"
    assert not (tmp_path / "day/repellent_response/output_manifest.csv").exists()
