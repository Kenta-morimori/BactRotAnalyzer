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
