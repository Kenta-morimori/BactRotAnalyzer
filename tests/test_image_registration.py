import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from utils.functions.image_registration import (
    crop_image,
    enlarge,
    optimal_threshold,
    reference_to_native,
    register_crop,
    verify_labels,
)


def test_saved_labels_reject_shifted_frames_and_timestamps():
    timeline = [
        dict(tiff_path="/synthetic/frame_00000.tif", source_timestamp="start"),
        dict(tiff_path="/synthetic/frame_00001.tif", source_timestamp="after gap"),
    ]
    labels = ["frame_00000.tif\nDateTime: start\n", "frame_00001.tif\nDateTime: after gap\n"]
    verify_labels(labels, timeline)
    with pytest.raises(ValueError, match="filename"):
        verify_labels(labels[::-1], timeline)
    with pytest.raises(ValueError, match="timestamp"):
        verify_labels([labels[0], labels[1].replace("after gap", "start")], timeline)
    with pytest.raises(ValueError, match="counts"):
        verify_labels(labels[:1], timeline)


def test_registration_recovers_crop_from_texture_without_centroids():
    rng = np.random.default_rng(82)
    image = rng.integers(0, 12000, (12, 13), dtype=np.uint16)
    expanded = enlarge(image, 3, "linear")
    reference = expanded[7:26, 9:30] > 6000
    best = register_crop(image, reference, scale=3, radius=3)[0]
    assert (best["interpolation"], best["x"], best["y"], best["iou"]) == ("linear", 9, 7, 1.0)
    transform = dict(best, scale=3, width=21, height=19)
    assert np.array_equal(crop_image(image, transform) > best["threshold"], reference)


def test_threshold_never_splits_equal_intensities():
    image = np.array([[1, 2, 2, 4]])
    threshold, score = optimal_threshold(image, np.array([[False, True, False, True]]))
    assert threshold == 1
    assert score == pytest.approx(2 / 3)


def test_pixel_center_transform_and_out_of_bounds():
    transform = dict(scale=10, interpolation="linear", x=20, y=30, width=10, height=10)
    assert np.allclose(reference_to_native([[0, 0], [9, 9]], transform), [[1.55, 2.55], [2.45, 3.45]])
    with pytest.raises(ValueError):
        crop_image(np.zeros((2, 2), np.uint16), transform)
    with pytest.raises(ValueError):
        optimal_threshold(np.zeros((2, 2)), np.ones((2, 2), bool))


def test_threshold_can_include_the_minimum_intensity_without_splitting_ties():
    image = np.ones((2, 2), dtype=np.uint16)
    reference = np.array([[True, True], [False, False]])
    threshold, score = optimal_threshold(image, reference)
    assert threshold < 1
    assert (image > threshold).all()
    assert score == 0.5


def test_failed_registered_review_records_failure_and_preserves_existing_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "registered_review_cli", root / "scripts/08_registered_centroid_review.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    setattr(module, "ROOT", tmp_path)
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "index.json").write_text("[]")
    output = tmp_path / "outputs" / "private" / "failed"
    arguments = [
        "--dataset",
        str(tmp_path / "absent"),
        "--diagnostics",
        str(tmp_path),
        "--prior-review",
        str(prior),
        "--raw-centroids",
        str(tmp_path),
        "--output",
        str(output),
    ]
    with pytest.raises(ValueError, match="No AVI"):
        module.main(arguments)
    before = (output / "run.json").read_bytes()
    assert json.loads(before)["status"] == "failed"
    with pytest.raises(FileExistsError):
        module.main(arguments)
    assert (output / "run.json").read_bytes() == before
