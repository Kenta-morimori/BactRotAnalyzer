"""Truth-backed ROI, target continuity, dropout and source-identity checks."""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from utils.functions import tiff_roi as roi
from utils.functions.tiff_roi_io import directory_rows, process


def image(objects=((30, 32, 5, 10000),), size=80):
    plane = np.full((size, size), 500, np.uint16)
    for x, y, r, value in objects:
        cv2.circle(plane, (x, y), r, value, -1)
    return plane


def rows(count):
    return [dict(source_frame_1based=i + 1, time_sec=i / 200, segment_id=0) for i in range(count)]


def test_auto_roi_contains_the_whole_calibration_orbit_not_the_distractor():
    sequence = []
    for i in range(220):
        angle = i * 0.2
        sequence.append(
            image(
                (
                    (40 + round(5 * np.cos(angle)), 40 + round(5 * np.sin(angle)), 5, 10000),
                    (70, 70, 8, 30000),
                ),
                size=96,
            )
        )
    found, summary = roi.infer_roi(sequence, rows(220), [45, 40])
    assert summary["calibration_frames"] == 200 and summary["detected_frames"] == 200
    assert found.x <= 28 and found.x + found.width >= 52
    assert found.y <= 28 and found.y + found.height >= 52
    assert found.x + found.width < 65 and found.y + found.height < 65
    for plane in sequence[:200]:
        _, _, objects = roi.binary_regions(plane, found, 2000)
        assert len(objects) == 1 and not objects[0].edge


@pytest.mark.parametrize("count,visible", [(19, 19), (100, 70)])
def test_auto_roi_requests_manual_setup_when_evidence_is_insufficient(count, visible):
    sequence = [image() if i < visible else image(()) for i in range(count)]
    found, summary = roi.infer_roi(sequence, rows(count), [30, 32])
    assert found is None and summary["reason"] == "manual_roi_required"


def test_calibration_never_crosses_timestamp_or_frame_boundaries():
    source = rows(50)
    source[10]["time_sec"] = 2
    found, summary = roi.infer_roi([image()] * 50, source, [30, 32])
    assert found is None and summary["calibration_frames"] == 10
    source = rows(50)
    source[10]["source_frame_1based"] = 30
    _, summary = roi.infer_roi([image()] * 50, source, [30, 32])
    assert summary["calibration_frames"] == 10


def test_target_is_seeded_not_largest_and_pixel_mean_uses_tenfold_centers():
    region = roi.ROI(10, 10, 60, 60)
    tracker = roi.Tracker(region)
    result = tracker.step(image(((30, 32, 5, 10000), (55, 50, 9, 30000))), 1, 0, 0, [30, 32])
    assert result.point is not None
    y, x = np.nonzero(result.mask)
    np.testing.assert_allclose(result.point, (np.array([x.mean(), y.mean()]) + 0.5) / 10 - 0.5 + [10, 10])
    assert np.linalg.norm(result.point - [30, 32]) < 0.1


def test_temporary_loss_remains_missing_but_does_not_disable_reacquisition():
    tracker = roi.Tracker(roi.ROI(10, 10, 60, 60))
    tracker.step(image(((30, 32, 5, 10000), (60, 50, 9, 30000))), 1, 0, 0, [30, 32])
    missing = tracker.step(image(((60, 50, 9, 30000),)), 2, 0.005, 0)
    assert missing.point is None and not missing.mask.any()
    recovered = tracker.step(image(((31, 32, 5, 10000), (60, 50, 9, 30000))), 3, 0.01, 0)
    assert recovered.reacquired and recovered.reason == "reacquired"
    assert np.linalg.norm(recovered.point - [31, 32]) < 0.1


def test_contact_stays_missing_until_separation_then_reacquires_target():
    tracker = roi.Tracker(roi.ROI(5, 10, 65, 60))
    tracker.step(image(((25, 32, 5, 10000), (40, 32, 5, 12000))), 1, 0, 0, [25, 32])
    joined = image(((28, 32, 5, 10000), (36, 32, 5, 12000)))
    first = tracker.step(joined, 2, 0.005, 0)
    assert first.point is None and first.reason == "merged"
    assert tracker.step(joined, 3, 0.01, 0).point is None
    separated = tracker.step(image(((25, 32, 5, 10000), (40, 32, 5, 12000))), 4, 0.015, 0)
    assert separated.reacquired and np.linalg.norm(separated.point - [25, 32]) < 0.1


def test_equal_candidates_are_not_assigned_a_target_identity():
    tracker = roi.Tracker(roi.ROI(5, 10, 65, 60))
    tracker.step(image(), 1, 0, 0, [30, 32])
    tracker.step(image(()), 2, 0.005, 0)
    result = tracker.step(image(((24, 32, 3, 10000), (36, 32, 3, 10000))), 3, 0.01, 0)
    assert result.point is None and result.reason == "ambiguous"


def test_roi_edge_is_flagged_and_not_automatically_expanded():
    fixed = roi.ROI(20, 20, 30, 30)
    tracker = roi.Tracker(fixed)
    tracker.step(image(), 1, 0, 0, [30, 32])
    result = tracker.step(image(((24, 32, 5, 10000),)), 2, 0.005, 0)
    assert result.reason == "roi_edge" and result.point is None
    assert tracker.roi == fixed
    assert tracker.step(image(), 3, 0.01, 0).reacquired


@pytest.mark.parametrize("frame,time,segment", [(3, 0.005, 0), (2, 1.0, 0), (2, 0.005, 1), (2, np.nan, 0)])
def test_recording_boundary_never_reuses_the_old_identity(frame, time, segment):
    tracker = roi.Tracker(roi.ROI(5, 10, 65, 60))
    tracker.step(image(), 1, 0, 0, [30, 32])
    assert tracker.step(image(), frame, time, segment).point is None
    assert tracker.step(image(), frame + 1, (time + 0.005) if np.isfinite(time) else 0.01, segment).point is None


def test_holes_and_disconnected_regions_keep_their_own_pixels():
    plane = image(())
    cv2.circle(plane, (35, 35), 12, 10000, -1)
    cv2.circle(plane, (35, 35), 7, 500, -1)
    cv2.circle(plane, (35, 35), 2, 10000, -1)
    fixed = roi.ROI(0, 0, 80, 80)
    _, _, objects = roi.binary_regions(plane, fixed, 2000)
    assert len(objects) == 2
    ring = max(objects, key=lambda obj: obj.area)
    assert not ring.mask[350, 350]
    y, x = np.nonzero(ring.mask)
    np.testing.assert_allclose(ring.point, (np.array([x.mean(), y.mean()]) + 0.5) / 10 - 0.5)


def manifest(tmp_path):
    source = []
    for index in range(4):
        path = tmp_path / f"source_{index:05}.tif"
        Image.fromarray(image() if index != 1 else image(())).save(path)
        source.append(
            dict(
                path=str(path),
                source_frame_1based=index + 1,
                time_sec=index * 0.005,
                source_timestamp=f"original-{index}",
                segment_id=0,
            )
        )
    return dict(
        samples=[
            dict(
                input_id="synthetic",
                threshold=2000,
                roi=[10, 10, 60, 60],
                seeds={"1": [30, 32]},
                frames=source,
            )
        ]
    )


def test_export_keeps_missing_rows_original_time_and_inputs_immutable(tmp_path):
    spec = manifest(tmp_path)
    before = {r["path"]: Path(r["path"]).read_bytes() for r in spec["samples"][0]["frames"]}
    output = process(spec, tmp_path / "out", video=False)
    table = pd.read_csv(output / "sample1/centroids.csv")
    assert table.source_frame_1based.tolist() == [1, 2, 3, 4]
    np.testing.assert_allclose(table.time_sec, [0, 0.005, 0.01, 0.015])
    assert table.source_timestamp.tolist() == [f"original-{i}" for i in range(4)]
    assert np.isnan(table.x_px.iloc[1]) and table.reacquired.tolist() == [False, False, True, False]
    assert json.loads((output / "run.json").read_text())["status"] == "complete"
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    csv_before = (output / "sample1/centroids.csv").read_bytes()
    with pytest.raises(FileExistsError):
        process(spec, output, video=False)
    assert (output / "sample1/centroids.csv").read_bytes() == csv_before


def test_source_suffixes_and_tiff_times_are_read_without_writing_caches(tmp_path):
    for suffix, stamp in [
        (0, "10/25/2026 12:00:00.000000"),
        (1, "10/25/2026 12:00:00.005000"),
        (5, "10/25/2026 12:00:01.000000"),
    ]:
        Image.fromarray(image()).save(tmp_path / f"source_{suffix:05}.tif", tiffinfo={306: stamp})
    source = directory_rows(tmp_path)
    assert [r["source_frame_1based"] for r in source] == [1, 2, 6]
    assert [r["time_sec"] for r in source] == [0, 0.005, 1]
    assert [r["segment_id"] for r in source] == [0, 0, 1]
    assert len(list(tmp_path.iterdir())) == 3


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg optional in unit-test environment")
def test_video_contains_each_source_observation_and_is_fully_decodable(tmp_path):
    output = process(manifest(tmp_path), tmp_path / "out", video=True, video_fps=10)
    cap = cv2.VideoCapture(str(output / "sample1/review.mp4"))
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape == (420, 960, 3)
        count += 1
    cap.release()
    assert count == 4
    assert json.loads((output / "sample1/presentation.json").read_text())["source_frames"] == [1, 2, 3, 4]


def test_one_click_setup_and_manual_roi_override_are_recorded(tmp_path, monkeypatch):
    from utils.functions import tiff_roi_io as io

    spec = manifest(tmp_path)["samples"][0]
    spec["seeds"] = {}
    spec.pop("roi")
    calls = []
    monkeypatch.setattr(io, "pick_target", lambda image: [30, 32])

    def manual(image, automatic):
        assert automatic is None  # Four frames are insufficient for auto ROI.
        calls.append(True)
        return roi.ROI(10, 10, 60, 60)

    monkeypatch.setattr(io, "confirm_roi", manual)
    fixed, seeds, calibration, _, _ = io.prepare(spec, interactive=True)
    assert calls and seeds == {"1": [30, 32]} and fixed == roi.ROI(10, 10, 60, 60)
    assert calibration["manual_override"]


def test_repeated_runs_are_reproducible_and_missing_time_never_compacts_rows(tmp_path):
    spec = manifest(tmp_path)
    spec["samples"][0]["frames"][1]["time_sec"] = None
    first = process(spec, tmp_path / "first", video=False)
    second = process(spec, tmp_path / "second", video=False)
    assert (first / "sample1/centroids.csv").read_bytes() == (second / "sample1/centroids.csv").read_bytes()
    table = pd.read_csv(first / "sample1/centroids.csv")
    assert len(table) == 4 and table.x_px.iloc[1:].isna().all()
    assert table.reason.iloc[1] == "invalid_time"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="Local video encoder")
def test_video_marks_time_and_source_gaps_even_if_segment_labels_are_unchanged(tmp_path):
    spec = manifest(tmp_path)
    spec["samples"][0]["frames"][2]["time_sec"] = 1.0
    spec["samples"][0]["frames"][3]["time_sec"] = 1.005
    output = process(spec, tmp_path / "out", video=True, video_fps=10)
    mapping = json.loads((output / "sample1/presentation.json").read_text())["source_frames"]
    assert mapping == [1, 2] + [None] * 10 + [3, 4]
    table = pd.read_csv(output / "sample1/centroids.csv")
    assert table.x_px.iloc[2:].isna().all()


@pytest.mark.parametrize("shape", ["circle", "ellipse", "asymmetric", "ring", "brightness"])
def test_fresh_synthetic_seeds_meet_agreed_position_and_coverage_bounds(shape):
    # Truth is used only to simulate the first foreground click and to score.
    # The algorithm receives grayscale images, that click and source identity.
    y, x = np.mgrid[:96, :96]
    for seed in range(4000, 4020):
        rng = np.random.default_rng(seed)
        phase = rng.uniform(0, 2 * np.pi)
        pictures = []
        truth = []
        masks = []
        for index in range(40):
            cx = 38 + 4 * np.cos(phase + index * 0.25)
            cy = 38 + 4 * np.sin(phase + index * 0.25)
            a, b = (8, 5) if shape == "ellipse" else (6, 6)
            mask = ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1
            if shape == "ring":
                mask &= (x - cx) ** 2 + (y - cy) ** 2 >= 3**2
            if shape == "asymmetric":
                mask &= ~((x > cx) & (y < cy))
            intensity = 12000 - index * 180 if shape == "brightness" else 12000
            plane = np.full((96, 96), 500, np.uint16)
            plane[mask] = intensity
            cv2.circle(plane, (76, 76), 10, 30000, -1)
            yy, xx = np.nonzero(mask)
            truth.append([xx.mean(), yy.mean()])
            masks.append(mask)
            pictures.append(plane)
        yy, xx = np.nonzero(masks[0])
        p = np.asarray(truth[0])
        closest = np.argmin((xx - p[0]) ** 2 + (yy - p[1]) ** 2)
        click = [int(xx[closest]), int(yy[closest])]
        fixed, calibration = roi.infer_roi(pictures, rows(40), click)
        assert fixed is not None
        tracker = roi.Tracker(fixed)
        estimates = []
        for index, plane in enumerate(pictures):
            result = tracker.step(plane, index + 1, index * 0.005, 0, click if index == 0 else None)
            estimates.append([np.nan, np.nan] if result.point is None else result.point)
        estimated_points = np.asarray(estimates)
        valid = np.isfinite(estimated_points).all(axis=1)
        assert valid.mean() >= 0.95
        error = np.linalg.norm(estimated_points[valid] - np.array(truth)[valid], axis=1)
        radius = np.sqrt(np.array([mask.sum() for mask in masks])[valid] / np.pi)
        assert np.percentile(error, 95) <= 0.5
        assert np.percentile(error / radius, 95) <= 0.1
        assert np.all(estimated_points[valid] < 60)  # never selected the brighter distractor


def test_unrelated_third_region_does_not_release_contact_lock():
    tracker = roi.Tracker(roi.ROI(5, 5, 70, 70))
    distant = (60, 60, 4, 20000)
    tracker.step(image(((25, 32, 5, 10000), (40, 32, 5, 12000), distant)), 1, 0, 0, [25, 32])
    joined = image(((28, 32, 5, 10000), (36, 32, 5, 12000), distant))
    assert tracker.step(joined, 2, 0.005, 0).reason == "merged"
    assert tracker.step(joined, 3, 0.01, 0).reason == "contact_unresolved"
    separated = tracker.step(image(((25, 32, 5, 10000), (40, 32, 5, 12000), distant)), 4, 0.015, 0)
    assert separated.reacquired and np.linalg.norm(separated.point - [25, 32]) < 0.1
