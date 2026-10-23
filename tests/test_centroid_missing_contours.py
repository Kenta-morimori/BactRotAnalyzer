import cv2
import numpy as np
import pandas as pd

from utils import param
from utils.functions import input_data
from utils.functions.get_centroid_coordinate import contours


def test_contours_returns_nan_when_no_region_exists():
    x, y, ellipse = contours(np.zeros((32, 32, 3), dtype=np.uint8))

    assert np.isnan(x)
    assert np.isnan(y)
    assert ellipse is None


def test_contours_returns_nan_for_too_small_contour():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[10, 10] = 255

    x, y, ellipse = contours(frame)

    assert np.isnan(x)
    assert np.isnan(y)
    assert ellipse is None


def test_contours_returns_centroid_and_ellipse_for_valid_contour():
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.ellipse(frame, (32, 32), (12, 8), 0, 0, 360, (255, 255, 255), -1)

    x, y, ellipse = contours(frame)

    assert np.isfinite(x)
    assert np.isfinite(y)
    assert ellipse is not None


def test_contours_uses_all_foreground_pixels_for_asymmetric_filled_region():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:45, 12:25] = 255
    mask[35:45, 25:52] = 255
    frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # Foreground brightness must not weight the binary area centroid.
    frame[10:25, 12:25] = 121
    expected_y, expected_x = np.nonzero(mask)

    x, y, ellipse = contours(frame)

    np.testing.assert_allclose([x, y], [expected_x.mean(), expected_y.mean()], atol=1e-12, rtol=0)
    assert ellipse is not None


def test_contours_excludes_holes_other_regions_and_islands_inside_holes():
    selected = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(selected, (32, 40), 24, 255, -1)
    cv2.circle(selected, (37, 40), 17, 0, -1)
    mask = selected.copy()
    # This island has no connection to the selected ring.
    cv2.circle(mask, (42, 40), 3, 255, -1)
    # More white pixels, but a smaller external contour area than the ring.
    cv2.circle(mask, (76, 76), 18, 255, -1)
    expected_y, expected_x = np.nonzero(selected)

    x, y, ellipse = contours(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))

    np.testing.assert_allclose([x, y], [expected_x.mean(), expected_y.mean()], atol=1e-12, rtol=0)
    assert ellipse is not None


def test_contours_does_not_fall_back_when_largest_contour_has_too_few_points():
    frame = np.zeros((96, 96, 3), dtype=np.uint8)
    cv2.rectangle(frame, (5, 5), (45, 45), (255, 255, 255), -1)
    cv2.circle(frame, (75, 75), 10, (255, 255, 255), -1)

    x, y, ellipse = contours(frame)

    assert np.isnan(x)
    assert np.isnan(y)
    assert ellipse is None


def test_input_centroid_coordinate_preserves_internal_nan(tmp_path, monkeypatch):
    output_day = tmp_path / "day"
    output_day.mkdir()
    pd.DataFrame({"x_1": [1.0, np.nan, 3.0], "y_1": [4.0, np.nan, 6.0]}).to_csv(
        output_day / "centroid_coordinate.csv", index=False
    )
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))

    x_list, y_list = input_data.input_centroid_coordinate("day")

    assert x_list[0].iloc[[0, 2]].tolist() == [1.0, 3.0]
    assert y_list[0].iloc[[0, 2]].tolist() == [4.0, 6.0]
    assert np.isnan(x_list[0].iloc[1])
    assert np.isnan(y_list[0].iloc[1])
