"""Scientific diagnostics retain time blocks and distinguish geometry from validity."""

import cv2
import numpy as np

from utils.functions import rotation_center_diagnostics as diagnostics
from utils.functions.image_diagnostics import measure_contour
from utils.functions.rotation_center_candidates import estimate_center


def test_revisiting_arc_does_not_increase_coverage():
    phase = np.linspace(-np.pi / 6, np.pi / 6, 50)
    arc = np.column_stack((np.cos(phase), np.sin(phase)))
    revisited = np.concatenate((arc, arc[::-1], arc))
    coverage, gap = diagnostics.angular_coverage(revisited, np.zeros(2))
    np.testing.assert_allclose([coverage, gap], [60, 300], atol=1e-10)


def test_missing_rows_do_not_shift_cross_validation_time_blocks(monkeypatch):
    phase = np.linspace(0, 20 * np.pi, 400)
    points = np.array([1.3, 1.45]) + np.column_stack((np.cos(phase), np.sin(phase))) * [0.3, 0.18]
    points[:50] = np.nan
    points[220:230] = np.nan
    fit = estimate_center(points, "ellipse_constrained")
    omitted_trials = []

    def record_fit(selected, method):
        omitted_trials.append(selected.copy())
        return estimate_center(selected, method)

    monkeypatch.setattr(diagnostics, "estimate_center", record_fit)
    result = diagnostics.block_cross_validation(points, "ellipse_constrained", fit=fit)
    assert [len(trial) for trial in omitted_trials] == [300] * 4
    assert [np.isnan(trial[:, 0]).sum() for trial in omitted_trials] == [10, 60, 50, 60]
    assert result["block_cv_computationally_ok"] == result["block_cv_valid"] == 4
    assert result["center_max_shift_R"] < 1e-5
    assert result["block_cv_p95_R"] < 1e-5


def test_missing_coordinates_leave_spectral_scores_unavailable():
    phase = np.linspace(0, 20 * np.pi, 400)
    points = np.column_stack((np.cos(phase), np.sin(phase))) * [0.3, 0.18]
    points[100] = np.nan
    result = diagnostics.evaluate_center(points, "ellipse_constrained", sample_rate_hz=200)
    assert result["n_input"] == 400 and result["n_used"] == 399
    assert np.isnan(result["rotation_quadrature_score"])
    assert np.isnan(result["peak_energy_fraction"])
    assert result["scientific_assessment"] == "not_assessed"


def test_contour_metrics_keep_empty_split_and_border_states():
    blank = np.zeros((80, 80), np.uint8)
    empty = measure_contour(blank)
    assert empty.count == 0 and empty.contour is None and np.isnan(empty.area)
    image = blank.copy()
    cv2.rectangle(image, (0, 10), (25, 35), 255, -1)
    cv2.rectangle(image, (60, 60), (65, 65), 255, -1)
    measured = measure_contour(image)
    assert measured.count == 2 and measured.border_touch
    assert measured.area == 625
    assert 0 < measured.circularity <= 1
