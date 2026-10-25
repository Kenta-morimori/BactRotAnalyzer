"""Synthetic-only checks of independent diagnostics and source identity."""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, TiffImagePlugin

from tests.synthetic.images import CASES, generate_images
from utils.functions import centroid_diagnostics as diagnostic
from utils.functions.get_centroid_coordinate import contours


def test_independent_mask_preserves_holes_and_excludes_enclosed_island():
    selected = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(selected, (40, 40), 26, 255, -1)
    cv2.circle(selected, (42, 40), 18, 0, -1)
    binary = selected.copy()
    cv2.circle(binary, (43, 40), 3, 255, -1)
    cv2.circle(binary, (80, 80), 10, 255, -1)
    _, mask, xy, matches = diagnostic.audit_frame(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), contours)
    assert matches
    np.testing.assert_array_equal(mask, selected > 0)
    np.testing.assert_allclose(xy, diagnostic.pixel_mean(selected), atol=1e-10)


def test_independent_audit_detects_wrong_centroid_and_preserves_missing_gate():
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    cv2.circle(image, (25, 25), 10, (255, 255, 255), -1)
    assert not diagnostic.audit_frame(image, lambda _: (0, 0, None))[3]
    image[:] = 0
    cv2.rectangle(image, (5, 5), (40, 40), (255, 255, 255), -1)
    _, selected, xy, matches = diagnostic.audit_frame(image, contours)
    assert selected.any() and np.isnan(xy).all() and matches


def test_verified_affine_includes_crop_and_pixel_center_offsets():
    points = np.array([[0.0, 0.0], [10, 20], [np.nan, np.nan]])
    assert np.isnan(diagnostic.original_coordinates(points, {"status": "unknown"})).all()
    transform: dict = dict(
        status="verified", evidence="synthetic crop", avi_to_tiff_affine=[[0.1, 0, 12.45], [0, 0.1, 9.45]]
    )
    result = diagnostic.original_coordinates(points, transform)
    np.testing.assert_allclose(result[:2], [[12.45, 9.45], [13.45, 11.45]])
    assert np.isnan(result[2]).all()
    transform["evidence"] = None
    assert np.isnan(diagnostic.original_coordinates(points, transform)).all()


def test_timeline_keeps_suffixes_gaps_invalid_times_and_unknown_avi_mapping(tmp_path):
    stamps = [
        "10/20/2026 16:00:00.000000",
        "10/20/2026 16:00:00.005000",
        "10/20/2026 16:00:01.010000",
        None,
        "10/20/2026 16:00:01.020000",
    ]
    paths = []
    for suffix, stamp in zip([0, 1, 3, 4, 5], stamps):
        path = tmp_path / f"frame_{suffix:05}.tif"
        tags = TiffImagePlugin.ImageFileDirectory_v2()
        if stamp:
            tags[306] = stamp
        Image.fromarray(np.zeros((8, 8), dtype=np.uint16)).save(path, tiffinfo=tags)
        paths.append(path)
    rows = diagnostic.timeline(paths[::-1], 0.005)
    assert [r["tiff_suffix_0based"] for r in rows] == [0, 1, 3, 4, 5]
    assert [r["segment_id"] for r in rows] == [0, 0, 1, 2, 3]
    assert rows[3]["time_sec"] is None
    assert all(r["avi_source_frame_1based"] is None for r in rows)
    assert diagnostic.representative_indices(rows) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("case", CASES)
def test_synthetic_truth_is_reproducible_and_has_distinct_definitions(case):
    first = generate_images(case, 3)
    second = generate_images(case, 3)
    np.testing.assert_array_equal(first.images, second.images)
    assert first.images.dtype == np.uint16
    assert first.masks.shape == (24, 2, 96, 96)
    np.testing.assert_array_equal(first.source_frames, np.arange(1, 25))
    for i in range(len(first.images)):
        np.testing.assert_allclose(
            first.isolated_centroids[i, 0], diagnostic.pixel_mean(first.masks[i, 0]), equal_nan=True
        )
    assert not np.array_equal(first.images, generate_images(case, 43).images)


def test_synthetic_motion_missing_and_occlusion_remain_explicit():
    stop = generate_images("stop", 0)
    np.testing.assert_allclose(stop.centers[8:16, 0], np.broadcast_to(stop.centers[8, 0], (8, 2)))
    gap = generate_images("gap", 0)
    assert gap.times[12] - gap.times[11] > 1 and gap.segments[12] == 1
    missing = generate_images("missing", 0)
    assert not missing.observable[8:16].any()
    assert np.isnan(missing.isolated_centroids[8:16, 0]).all()
    overlap = generate_images("occlusion", 0)
    assert not overlap.observable[8:16].any()
    assert np.isfinite(overlap.centers[8:16, 0]).all()


def test_uint16_threshold_is_not_reduced_to_eight_bits():
    image = np.array([[1000, 3000, 65000]], dtype=np.uint16)
    np.testing.assert_array_equal(diagnostic.diagnostic_mask(image, 2000), [[False, True, True]])


def test_protocol_blocks_comparison_and_split_changes():
    path = Path(__file__).resolve().parents[1] / "configs" / "centroid_evaluation.json"
    config = json.loads(path.read_text())
    diagnostic.validate_config(config)
    config["stage"] = "B"
    config["acceptance"]["status"] = "proposed_not_agreed"
    with pytest.raises(ValueError, match="Stage B"):
        diagnostic.validate_config(config)
    config["stage"] = "A"
    config["synthetic_seeds"]["validation"] = [0]
    with pytest.raises(ValueError, match="seed split"):
        diagnostic.validate_config(config)


def test_regression_reports_short_decode_instead_of_passing(tmp_path):
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), getattr(cv2, "VideoWriter_fourcc")(*"MJPG"), 200, (48, 48))
    assert writer.isOpened()
    for _ in range(3):
        frame = np.zeros((48, 48, 3), dtype=np.uint8)
        cv2.circle(frame, (24, 24), 9, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    rows, summary = diagnostic.regression(path, contours, 4, 200, np.array([0.02, 0.03]))
    assert not summary["frame_count_matches"]
    assert summary["pixel_mean_mismatches"] == 0
    np.testing.assert_allclose(rows[:, 1], [0, 0.005, 0.01])
    np.testing.assert_allclose(rows[:, 4:], rows[:, 2:4] * [0.02, 0.03])


@pytest.mark.parametrize(
    "script,arguments",
    [
        ("05_centroid_diagnostics.py", ["synthetic"]),
        ("06_centroid_comparison.py", ["synthetic"]),
        ("07_centroid_review.py", ["--comparison", "absent", "--manifest-root", "absent"]),
        (
            "08_registered_centroid_review.py",
            ["--dataset", "absent", "--diagnostics", "absent", "--prior-review", "absent", "--raw-centroids", "absent"],
        ),
    ],
)
def test_cli_refuses_public_output_before_writing(script, arguments):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("diagnostics_cli", root / "scripts" / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit):
        option = "--output" if script.startswith("08") else "--output-root"
        module.main(arguments + [option, str(root / "outputs" / "test_data" / "should_not_exist")])


def test_private_output_rejects_symlink_escape_and_public_alias(tmp_path):
    outputs = tmp_path / "repo" / "outputs"
    outputs.mkdir(parents=True)
    (outputs / "test_data").mkdir()
    (outputs / "private").mkdir()
    (outputs / "outside").symlink_to(tmp_path, target_is_directory=True)
    (outputs / "public_alias").symlink_to(outputs / "test_data", target_is_directory=True)
    for path in (outputs, outputs / "outside" / "run", outputs / "public_alias" / "run"):
        with pytest.raises(ValueError, match="private subdirectory"):
            diagnostic.private_output_root(outputs.parent, path)
    assert diagnostic.private_output_root(outputs.parent, outputs / "private") == outputs / "private"


def test_metrics_include_rejections_wrong_identity_and_unidentifiable_frames():
    frames = np.arange(1, 5)
    predicted = [[0, 0], [3, 4], [np.nan, np.nan], [1, 1]]
    truth = [[0, 0], [0, 0], [0, 0], [1, 1]]
    metrics = diagnostic.position_metrics(
        predicted,
        truth,
        [True, True, True, False],
        [5, 5, 5, 5],
        ["target", "distractor", "", "target"],
        frames,
        frames,
    )
    assert metrics["accepted_fraction"] == 0.75
    assert metrics["identifiable_accepted_fraction"] == 2 / 3
    assert metrics["wrong_target_count"] == metrics["unidentifiable_finite_count"] == 1
    assert metrics["position_native_px"]["maximum"] == 5
    assert metrics["position_over_radius"]["maximum"] == 1
    with pytest.raises(ValueError, match="source-frame"):
        diagnostic.position_metrics(predicted, truth, [True] * 4, [5] * 4, ["target"] * 4, frames[::-1], frames)
