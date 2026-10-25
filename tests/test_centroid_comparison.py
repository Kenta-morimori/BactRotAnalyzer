"""Image-based failure modes for experimental candidates and evaluation gates."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.synthetic.images import generate_images
from utils.functions import centroid_comparison as comparison
from utils.functions import centroid_diagnostics as diagnostic
from utils.functions.get_centroid_coordinate import contours

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "centroid_evaluation.json").read_text())
METHOD = CONFIG["comparison"]["methods"][0]
SETTINGS = CONFIG["comparison"]["selector"]


def objects_at(*points):
    mask = np.zeros((80, 80), dtype=np.uint8)
    for point in points:
        cv2.circle(mask, point, 5, 1, -1)
    return comparison.components(mask > 0, 1)


def test_tracking_clears_identity_after_missing_and_requires_explicit_reidentification():
    tracker = comparison.ConstrainedSelector(SETTINGS)
    assert tracker.step(objects_at((30, 30)), 1, 0, 0, [30, 30])[0] is not None
    assert tracker.step([], 2, 0.005, 0)[0] is None
    assert tracker.step(objects_at((30, 30)), 3, 0.01, 0)[1] == "reidentification_required"
    assert tracker.step(objects_at((30, 30)), 4, 0.015, 0, [30, 30])[0] is not None


@pytest.mark.parametrize("frame,time,segment", [(3, 0.005, 0), (2, 1, 0), (2, 0.005, 1), (2, np.nan, 0), (2, 0, 0)])
def test_tracking_never_carries_identity_across_source_or_time_boundaries(frame, time, segment):
    tracker = comparison.ConstrainedSelector(SETTINGS)
    tracker.step(objects_at((30, 30)), 1, 0, 0, [30, 30])
    assert tracker.step(objects_at((30, 30)), frame, time, segment)[0] is None


def test_multiple_plausible_objects_and_edge_contact_are_missing():
    tracker = comparison.ConstrainedSelector(SETTINGS)
    tracker.step(objects_at((30, 30)), 1, 0, 0, [30, 30])
    assert tracker.step(objects_at((24, 30), (36, 30)), 2, 0.005, 0)[0] is None
    assert tracker.step(objects_at((0, 30)), 3, 0.01, 0, [0, 30])[0] is None


def test_nearest_tenfold_mask_and_centroid_keep_native_pixel_center_convention():
    data = generate_images("asymmetric", 42)
    method = dict(METHOD, scale=10, interpolation="nearest")
    native = comparison.segment(data.images[0], METHOD)
    scaled = comparison.segment(data.images[0], method)
    original = comparison.components(native, 1)[0]
    enlarged = comparison.components(scaled, 10)[0]
    np.testing.assert_array_equal(original.coverage, enlarged.coverage)
    np.testing.assert_allclose(original.center, enlarged.center, atol=1e-12)


def test_cache_separates_settings_inputs_and_detects_reuse(tmp_path):
    images = generate_images("circle", 42).images[:2]
    first, reused = comparison.cached_masks(images, METHOD, tmp_path)
    assert not reused
    again, reused = comparison.cached_masks(images, METHOD, tmp_path)
    assert reused
    np.testing.assert_array_equal(first, again)
    comparison.cached_masks(images, dict(METHOD, threshold=12000), tmp_path)
    comparison.cached_masks(images + 1, METHOD, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 3


def test_separation_failure_is_scored_independently_of_chosen_target():
    data = generate_images("partial_overlap", 42)
    descriptions = [comparison.components(comparison.segment(data.images[0], METHOD), 1)]
    metrics = comparison.separation_metrics(descriptions, data.masks[:1])
    assert metrics["merge_frames"] == 1


def test_no_seed_means_missing_without_oracle_recovery():
    data = generate_images("circle", 42)
    masks = np.stack([comparison.segment(image, METHOD) for image in data.images[:3]])
    result, _ = comparison.select_sequence(
        masks, METHOD, SETTINGS, data.source_frames[:3], data.times[:3], data.segments[:3], {}
    )
    assert np.isnan(result["constrained"]["points"]).all()
    assert np.isfinite(result["largest_external_contour"]["points"]).all()


def test_agreed_criteria_reject_pooled_accuracy_without_coverage_or_identity():
    metrics = diagnostic.position_metrics(
        [[0, 0], [np.nan, np.nan]], [[0, 0], [0, 0]], [True, True], [7, 7], ["target", ""], [1, 2], [1, 2]
    )
    assert comparison.acceptance_failures(metrics, CONFIG["acceptance"]) == ["accepted_fraction"]
    metrics["unidentifiable_finite_count"] = 1
    assert "unidentifiable_finite_output" in comparison.acceptance_failures(metrics, CONFIG["acceptance"])


def test_touching_without_hidden_pixels_remains_observable():
    data = generate_images("approach", 42)
    for index, mask in enumerate(data.masks[:, 0]):
        # The distractor overwrites target pixels with its distinct intensity.
        # A fully visible target remains scoreable even when masks are adjacent.
        fully_visible = np.all(data.images[index][mask] == 17000)
        assert data.observable[index] == fully_visible


@pytest.mark.parametrize("points", [((20, 30), (50, 30)), ((30, 20), (30, 50))])
def test_baseline_matches_normal_extraction_when_external_areas_tie(points):
    mask = np.zeros((80, 80), dtype=np.uint8)
    for point in points:
        cv2.circle(mask, point, 7, 255, -1)
    expected = contours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))[:2]
    result, _ = comparison.select_sequence(np.array([mask > 0]), METHOD, SETTINGS, [1], [0], [0], {})
    np.testing.assert_allclose(result["largest_external_contour"]["points"][0], expected, atol=1e-12)
