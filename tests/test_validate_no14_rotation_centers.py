import importlib.util
from pathlib import Path

import numpy as np
import pytest
import cv2


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "validation" / "validate_no14_rotation_centers.py"
SPEC = importlib.util.spec_from_file_location("validate_no14_rotation_centers", SCRIPT_PATH)
validation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validation)


def test_window_center_recovers_synthetic_circle_center():
    time = np.arange(0.0, 3.0, 0.005)
    center = np.array([1.5, 2.0])
    radius = 0.25
    x = center[0] + radius * np.cos(2 * np.pi * 20.0 * time)
    y = center[1] + radius * np.sin(2 * np.pi * 20.0 * time)

    cx, cy, count, fit_ok, points = validation.estimate_window_center(time, x, y, len(time) - 1, 0.5)

    assert fit_ok
    assert count == len(points)
    assert count >= 5
    assert cx == pytest.approx(center[0], abs=1e-6)
    assert cy == pytest.approx(center[1], abs=1e-6)


def test_window_center_reports_missing_for_insufficient_or_nan_points():
    time = np.arange(0.0, 0.02, 0.005)
    x = np.array([1.0, np.nan, 1.1, np.nan])
    y = np.array([2.0, np.nan, 2.1, np.nan])

    cx, cy, count, fit_ok, points = validation.estimate_window_center(time, x, y, len(time) - 1, 0.5)

    assert not fit_ok
    assert np.isnan(cx)
    assert np.isnan(cy)
    assert count == len(points) == 2


def test_validation_centroid_is_extracted_from_the_displayed_avi_frame():
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(frame, (30, 45), 10, (255, 255, 255), -1)

    x_um, y_um = validation.extract_centroid_from_avi_frame(frame, 0.02, 0.02)

    assert x_um == pytest.approx(30 * 0.02, abs=0.03)
    assert y_um == pytest.approx(45 * 0.02, abs=0.03)
