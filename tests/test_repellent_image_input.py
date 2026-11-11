"""Synthetic images exercise settings, cache isolation and canonical analysis."""

import importlib.util
import json
from argparse import Namespace
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from utils.functions import repellent_input as inputs
from utils.functions.repellent_workflow import recalculate_manifest
from utils.run_context import input_inventory
from utils.run_inputs import manifest_inputs, repellent_inputs, tiff_inputs


def recording(tmp_path, *, frames=80, gap=False):
    root = tmp_path / "data/day/tiff_data/recording"
    root.mkdir(parents=True)
    first = datetime(2026, 1, 1)
    for index in range(frames):
        frame = index + int(gap and index >= 40)
        plane = np.full((64, 64), 500, np.uint16)
        angle = 2 * np.pi * 12 * index / 200
        if index != 25:
            cv2.circle(plane, (32 + round(5 * np.cos(angle)), 32 - round(5 * np.sin(angle))), 3, 10000, -1)
        stamp = (first + timedelta(seconds=frame / 200)).strftime("%m/%d/%Y %H:%M:%S.%f")
        if index == 30:
            stamp = "unavailable"
        Image.fromarray(plane).save(
            root / f"recording_{frame:05}.tif",
            tiffinfo={306: stamp, 40092: "a\tb\tINTERNAL FRAME INTERVAL\t0.005".encode("utf-16le")},
        )
    return root


def settings_file(tmp_path, **kwargs):
    recording(tmp_path, **kwargs)
    settings = inputs.draft_settings("day", "tiff", tmp_path / "data", px2um=[0.2, 0.2])
    settings["order_confirmed"] = True
    settings["samples"][0].update(
        sample_no=7,
        target_confirmed=True,
        threshold=2000,
        roi=[15, 15, 34, 34],
        seeds={"1": [37, 32], "32": [35, 36], "42": [28, 29]},
    )
    # Seeds after missing timestamps and frame gaps are explicitly specified.
    paths = sorted((tmp_path / "data/day/tiff_data/recording").glob("*.tif"))
    for no in list(settings["samples"][0]["seeds"]):
        matching = [p for p in paths if int(p.stem.rsplit("_", 1)[1]) + 1 == int(no)]
        if matching:
            with Image.open(matching[0]) as image:
                mask = np.asarray(image) > 2000
                y, x = np.nonzero(mask)
                settings["samples"][0]["seeds"][no] = [float(x.mean()), float(y.mean())]
    path = tmp_path / "settings.json"
    inputs._write(path, settings)
    return path, settings


def prepare(path, **kwargs):
    return inputs.prepare_measurements(path, path.parent / "outputs", source="tiff", day="day", **kwargs)


def test_draft_never_borrows_avi_calibration_or_selects_an_unconfirmed_target(tmp_path):
    recording(tmp_path)
    config = tmp_path / "data/day/config.ini"
    config.write_text(
        "[Settings]\npx2um_x=0.02\npx2um_y=0.02\n[Tiff_info]\ntiff_data=recording\n"
        "[RepellentResponse]\nanalysis_frame_ranges=5-60\n"
    )
    with pytest.raises(ValueError, match="calibration"):
        inputs.draft_settings("day", "tiff", tmp_path / "data")
    spec = inputs.draft_settings("day", "tiff", tmp_path / "data", px2um=[0.2, 0.2])
    assert spec["order_confirmed"] and spec["samples"][0]["analysis_frame_range"] == [5, 60]
    assert spec["samples"][0]["px2um_x"] == 0.2 and spec["samples"][0]["threshold"] is None
    path = tmp_path / "settings.json"
    inputs._write(path, spec)
    with pytest.raises(ValueError, match="confirm target"):
        prepare(path)
    assert not (tmp_path / "outputs").exists()


def test_tiff_bridge_keeps_original_frames_missing_time_reason_and_reacquisition(tmp_path):
    path, _ = settings_file(tmp_path, gap=True)
    manifest = prepare(path)
    entry = json.loads(manifest.read_text())["samples"][0]
    raw = pd.read_csv(entry["centroids"])
    assert len(raw) == 80 and 41 not in raw.source_frame_1based.to_list()
    assert raw[["x_um", "y_um"]].iloc[25].isna().all()
    assert raw[["time_sec", "x_um", "y_um"]].iloc[30].isna().all()
    assert raw.reason.iloc[30] == "invalid_time"
    np.testing.assert_allclose(raw.x_um, raw.x_px * 0.2, equal_nan=True)
    bundle, features = recalculate_manifest(manifest)
    observed = bundle["observations"]
    assert observed.sample_no.eq(7).all() and observed.source_id.eq("recording").all()
    np.testing.assert_array_equal(observed.source_frame_1based, raw.source_frame_1based)
    np.testing.assert_allclose(observed.time_sec, raw.time_sec, equal_nan=True)
    assert observed.angular_velocity_rad_s.iloc[[25, 26, 27, 30, 31, 40]].isna().all()
    assert bundle["fluctuation"] is None and bundle["candidates"] is None and features is None
    assert bundle["settings"]["image_input"]["background_policy"] == "external_only"
    assert not list((tmp_path / "outputs").rglob("sample7"))


def test_cache_reuse_setting_changes_input_changes_and_tampering_are_distinct(tmp_path, monkeypatch):
    path, settings = settings_file(tmp_path)
    first = prepare(path)
    csv = Path(json.loads(first.read_text())["samples"][0]["centroids"])
    before = csv.read_bytes(), csv.stat().st_mtime_ns
    original = inputs._extract
    monkeypatch.setattr(inputs, "_extract", lambda *args: pytest.fail("unchanged measurement should reuse cache"))
    assert prepare(path) == first
    assert before == (csv.read_bytes(), csv.stat().st_mtime_ns)
    monkeypatch.setattr(inputs, "_extract", original)
    settings["samples"][0]["threshold"] = 2500
    inputs._write(path, settings)
    changed = prepare(path)
    assert changed != first and csv.read_bytes() == before[0]
    with pytest.raises(ValueError, match="regenerate"):
        recalculate_manifest(first)
    other_csv = Path(json.loads(changed.read_text())["samples"][0]["centroids"])
    other_csv.write_text("corrupt")
    prepare(path)
    assert len(pd.read_csv(other_csv)) == 80
    image = next((tmp_path / "data").rglob("*.tif"))
    with Image.open(image) as im:
        plane, tags = np.array(im), dict(im.tag_v2)
    plane[0, 0] += 1
    Image.fromarray(plane).save(image, tiffinfo=tags)
    assert prepare(path) != changed


def test_source_scope_external_background_and_original_numbers(tmp_path, monkeypatch):
    path, settings = settings_file(tmp_path)
    settings["samples"][0]["analysis_frame_range"] = [11, 60]
    inputs._write(path, settings)
    manifest = prepare(path, numbers=[7])
    raw = pd.read_csv(json.loads(manifest.read_text())["samples"][0]["centroids"], float_precision="round_trip")
    assert [raw.source_frame_1based.iloc[0], raw.source_frame_1based.iloc[-1]] == [11, 60]
    bg = raw[["source_frame_1based", "time_sec"]].copy()
    bg["background_intensity"] = np.arange(len(bg), dtype=float)
    bg.to_csv(tmp_path / "background.csv", index=False, float_format="%.17g")
    settings["samples"][0]["background"] = dict(csv="background.csv", rise_source_frame=45)
    inputs._write(path, settings)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            inputs, "_extract", lambda *args: pytest.fail("background changes must reuse image measurements")
        )
        bundle, _ = recalculate_manifest(prepare(path))
    assert bundle["settings"]["rise_source_frames"] == {"7": 45}
    assert "background_intensity" in bundle["observations"]
    bg.loc[1, "source_frame_1based"] = 13
    bg.to_csv(tmp_path / "background.csv", index=False, float_format="%.17g")
    with pytest.raises(ValueError, match="frames and timestamps"):
        prepare(path)
    settings["samples"][0]["background"] = None
    inputs._write(path, settings)
    full = json.loads(prepare(path, all_frames=True).read_text())
    assert len(pd.read_csv(full["samples"][0]["centroids"])) == 80


def test_calibrated_ccw_images_produce_finite_positive_standard_velocity(tmp_path):
    path, _ = settings_file(tmp_path, frames=240)
    bundle, _ = recalculate_manifest(prepare(path))
    observed = bundle["observations"]
    omega = observed.angular_velocity_rad_s.to_numpy()
    assert np.isfinite(omega).sum() > 150
    assert np.nanmedian(omega) == pytest.approx(2 * np.pi * 12, rel=0.1)
    assert np.isnan(observed.angular_velocity_rad_s.iloc[41])  # crossing an explicit re-selection is unavailable
    assert np.isfinite(observed.angular_velocity_rad_s.iloc[42])  # the next edge belongs entirely to the new track
    assert np.nanmedian(observed.center_x_um) == pytest.approx(6.4, abs=0.2)
    assert np.nanmedian(observed.center_y_um) == pytest.approx(6.4, abs=0.2)


def test_settings_order_source_day_and_sample_selection_must_be_explicit(tmp_path):
    path, settings = settings_file(tmp_path)
    settings["order_confirmed"] = False
    inputs._write(path, settings)
    with pytest.raises(ValueError, match="order"):
        prepare(path)
    with pytest.raises(ValueError, match="source"):
        inputs.prepare_measurements(path, tmp_path / "out", source="avi", day="day")
    with pytest.raises(ValueError, match="day"):
        inputs.prepare_measurements(path, tmp_path / "out", source="tiff", day="other")
    for selection in ([1], [7, 7], []):
        with pytest.raises(ValueError, match="selection"):
            inputs.selected_samples(settings, selection)


def test_interactive_choices_are_saved_and_a_cancel_preserves_old_settings(tmp_path, monkeypatch):
    path, settings = settings_file(tmp_path, gap=True)
    calls = []

    def pick(image):
        calls.append(True)
        y, x = np.nonzero(image > 2000)
        return [float(x.mean()), float(y.mean())]

    monkeypatch.setattr(inputs.io, "pick_target", pick)
    monkeypatch.setattr(inputs, "choose_threshold", lambda image, seed, initial: 2100.0)
    monkeypatch.setattr(inputs.io, "confirm_roi", lambda image, roi: inputs.ROI(15, 15, 34, 34))
    result = inputs.configure_settings(path)
    sample = result["samples"][0]
    assert sample["target_confirmed"] and sample["threshold"] == 2100
    assert set(sample["seeds"]) == {"1", "32", "42"} and len(calls) == 3
    saved = path.read_bytes()
    monkeypatch.setattr(inputs.io, "pick_target", lambda image: (_ for _ in ()).throw(ValueError("cancelled")))
    with pytest.raises(ValueError, match="cancelled"):
        inputs.configure_settings(path)
    assert path.read_bytes() == saved


def test_avi_uses_separate_method_and_native_calibration_and_refuses_count_mismatch(tmp_path):
    recording(tmp_path, frames=10)
    avi = tmp_path / "data/day/recording.avi"
    writer = cv2.VideoWriter(str(avi), getattr(cv2, "VideoWriter_fourcc")(*"FFV1"), 200, (64, 64))
    assert writer.isOpened()
    for _ in range(10):
        image = np.zeros((64, 64, 3), np.uint8)
        cv2.circle(image, (32, 32), 5, (255, 255, 255), -1)
        writer.write(image)
    writer.release()
    spec = inputs.draft_settings("day", "avi", tmp_path / "data", px2um=[0.02, 0.02])
    spec["order_confirmed"] = True
    path = tmp_path / "settings.json"
    inputs._write(path, spec)
    manifest = inputs.prepare_measurements(path, tmp_path / "outputs", source="avi", day="day")
    entry = json.loads(manifest.read_text())["samples"][0]
    raw = pd.read_csv(entry["centroids"])
    np.testing.assert_allclose(raw.x_um, raw.x_px * 0.02)
    assert raw.avi_frame_1based.to_list() == list(range(1, 11))
    assert inputs.AVI_METHOD in entry["centroids"] and inputs.TIFF_METHOD not in entry["centroids"]
    next((tmp_path / "data").rglob("*.tif")).unlink()
    with pytest.raises(ValueError, match="counts differ"):
        inputs.prepare_measurements(path, tmp_path / "outputs", source="avi", day="day")


def test_analysis_cli_generates_a_manifest_and_publishes_grouped_outputs(tmp_path, monkeypatch):
    path, _ = settings_file(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/04_repellent_response_analysis.py"
    spec = importlib.util.spec_from_file_location("image_analysis", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.param, "save_dir_bef", str(tmp_path / "outputs"))
    from utils.functions import repellent_result_export

    exported = []
    monkeypatch.setattr(
        repellent_result_export, "export_results", lambda destination, **bundle: exported.append((destination, bundle))
    )
    options = dict(
        day="day",
        baseline_ratio=0.5,
        sigma_threshold=3,
        min_consecutive=3,
        post_rise_center_mode=2,
        stop_activity_ratio=0.1,
        stop_min_duration_rotations=2,
        output_suffix=None,
        input_settings=str(path),
        input_source="tiff",
    )
    module.main(**options)
    assert exported[0][0] == tmp_path / "outputs/reanalysis_tiff/day/repellent_response"
    assert exported[0][1]["observations"].sample_no.eq(7).all()
    with pytest.raises(ValueError, match="explicit input-source"):
        module.main(**dict(options, input_source=None))
    with pytest.raises(ValueError, match="mutually exclusive"):
        module.main(**options, measurement_input="other.json")


def test_execution_inventory_includes_external_settings_tiffs_background_and_manifest_sources(tmp_path):
    path, settings = settings_file(tmp_path, frames=10)
    settings["samples"][0]["background"] = dict(csv="background.csv")
    (tmp_path / "background.csv").write_text("synthetic background")
    inputs._write(path, settings)
    args = Namespace(input_settings=str(path), input_samples=[7], input_source="tiff", output_suffix=None, day="day")
    saved = input_inventory(repellent_inputs(args))
    files = {item["path"] for item in saved["files"]}
    assert str(path) in files and str(tmp_path / "background.csv") in files
    assert len([name for name in files if name.endswith(".tif")]) == 10
    settings["samples"][0]["background"] = None
    inputs._write(path, settings)
    # The standalone TIFF manifest has different references, but the same path rules as its loader.
    descriptor = tmp_path / "tiff.json"
    frame = next((tmp_path / "data").rglob("*.tif"))
    descriptor.write_text(json.dumps(dict(samples=[dict(frames=[dict(path=str(frame))])])))
    assert set(tiff_inputs(Namespace(manifest=descriptor))) == {descriptor, frame}
    manifest = prepare(path)
    inventory = input_inventory(manifest_inputs(manifest, "measurement"))
    assert str(path) in {item["path"] for item in inventory["files"]}
    assert len([item for item in inventory["files"] if item["path"].endswith(".tif")]) == 10
