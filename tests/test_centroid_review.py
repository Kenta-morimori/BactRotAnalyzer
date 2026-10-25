"""Registration, missing data and video encoding for qualitative TIFF reviews."""

import shutil

import cv2
import numpy as np
import pytest

from utils.functions import centroid_review as review


def test_native_pixel_centers_stay_registered_after_display_zoom():
    assert review.display_point([0, 0], 5) == (2, 2)
    assert review.display_point([10, 12], 5) == (52, 62)
    image = np.full((24, 24), 1000, dtype=np.uint16)
    mask = np.zeros_like(image, dtype=float)
    mask[12, 10] = 1
    canvas = review.overlay(image, mask, mask, [10, 12], (0, 2000), zoom=5)
    np.testing.assert_array_equal(canvas[62, 52], review.COLORS["centroid"])


def test_missing_centroid_is_not_rendered_at_zero_or_carried_forward():
    image = np.zeros((24, 24), dtype=np.uint16)
    mask = np.zeros_like(image, dtype=float)
    missing = review.overlay(image, mask, mask, [np.nan, np.nan], (0, 1), zoom=5)
    assert not missing.any()
    observed = review.overlay(image, mask, mask, [0, 0], (0, 1), zoom=5)
    assert observed[2, 2, 2] == 255


def test_saved_results_require_exact_tiff_frame_and_timestamp_identity():
    rows = [
        dict(tiff_suffix_0based=0, time_sec=0.0, segment_id=0),
        dict(tiff_suffix_0based=8, time_sec=1.0, segment_id=1),
    ]
    saved = dict(
        source_frames=np.array([1, 9]),
        times=np.array([0, 1.0]),
        segments=np.array([0, 1]),
        points=np.zeros((2, 2)),
        selected_masks=np.zeros((2, 24, 24)),
        reasons=np.array(["accepted", "missing"]),
    )
    review.validate_saved(saved, rows, (24, 24))
    saved["source_frames"] = np.array([1, 2])
    with pytest.raises(ValueError, match="source frames"):
        review.validate_saved(saved, rows, (24, 24))
    saved["source_frames"] = np.array([1, 9])
    saved["times"] = np.array([0, 0.005])
    with pytest.raises(ValueError, match="timestamps"):
        review.validate_saved(saved, rows, (24, 24))


def test_method_spread_does_not_fabricate_missing_estimates():
    assert review.centroid_span([[np.nan, np.nan], [1, 2]]) is None
    assert review.centroid_span([[0, 0], [3, 4], [np.nan, np.nan]]) == 5


def test_display_contrast_is_fixed_for_the_window_and_preserves_uint16_range():
    images = np.array([[[1000, 10000]], [[1000, 20000]]], dtype=np.uint16)
    limits = review.display_limits(images)
    first, second = [review.gray_display(image, limits) for image in images]
    assert first[0, 0] == second[0, 0]
    assert first[0, 1] < second[0, 1]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="Local video encoder is optional")
def test_video_encoding_preserves_frame_count_and_refuses_overwrite(tmp_path):
    path = tmp_path / "review.mp4"
    frames = [np.full((24, 24, 3), value, np.uint8) for value in (0, 80, 160)]
    review.encode_video(path, frames, fps=10)
    capture = cv2.VideoCapture(str(path))
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    assert count == 3
    with pytest.raises(FileExistsError):
        review.encode_video(path, frames, fps=10)
