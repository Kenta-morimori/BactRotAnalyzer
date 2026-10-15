import numpy as np
import pytest

from utils.functions.rotation_center_candidates import (
    ELLIPSE_METHODS,
    ellipse_distance,
    estimate_center,
)


def trajectory(noise=0.0):
    phase = np.linspace(0, 8 * np.pi, 400, endpoint=False)
    theta = 0.4
    basis = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    points = np.array([1.3, 1.45]) + (np.column_stack((np.cos(phase), np.sin(phase))) * [0.3, 0.18]) @ basis.T
    return points + np.random.default_rng(38).normal(0, noise, points.shape)


@pytest.mark.parametrize("method", ELLIPSE_METHODS)
def test_known_ellipse_center_and_distance(method):
    points = trajectory()
    result = estimate_center(points, method)
    assert result.computationally_ok
    assert result.center is not None and result.axes is not None and result.ellipse is not None
    np.testing.assert_allclose(result.center, [1.3, 1.45], atol=1e-6)
    np.testing.assert_allclose(sorted(result.axes), [0.18, 0.3], atol=1e-6)
    assert np.max(np.abs(ellipse_distance(points, result.ellipse))) < 1e-6


@pytest.mark.parametrize("method", ("ellipse_constrained", "geometric_ellipse"))
def test_translation_invariance_and_missing_rows(method):
    points = trajectory(0.02)
    points[30:40] = np.nan
    original = points.copy()
    shift = np.array([2.0, -1.0])
    first = estimate_center(points, method)
    shifted = estimate_center(points + shift, method)
    assert first.center is not None and shifted.center is not None
    assert first.n_input == 400 and first.n_used == 390
    np.testing.assert_allclose(points, original, equal_nan=True)
    np.testing.assert_allclose(shifted.center - shift, first.center, atol=2e-5)


def test_legacy_origin_sensitivity_is_recorded():
    points = trajectory(0.08)
    first = estimate_center(points, "legacy")
    shifted = estimate_center(points + [2, -1], "legacy")
    assert first.center is not None and shifted.center is not None
    assert np.linalg.norm(first.center - (shifted.center - [2, -1])) > 1e-3


def test_failure_states_do_not_mean_physical_validity():
    assert estimate_center(np.full((50, 2), np.nan), "ellipse_constrained").status == "insufficient_points"
    assert estimate_center(np.zeros((50, 2)), "geometric_ellipse").status == "degenerate"
    result = estimate_center(trajectory(0.08), "geometric_ellipse", max_nfev=1)
    assert result.status == "not_converged" and result.converged is False
    assert result.ellipse is not None and not result.computationally_ok
    phase = np.linspace(-1, 1, 100)
    hyperbola = np.column_stack((np.cosh(phase), np.sinh(phase)))
    result = estimate_center(hyperbola, "legacy")
    assert result.status == "nonellipse" and result.center is not None and result.ellipse is None


@pytest.mark.parametrize("method", ELLIPSE_METHODS)
def test_short_arc_and_stopped_cloud_return_computational_results(method):
    # Good computation cannot certify a center from a short arc or a stopped cloud.
    rng = np.random.default_rng(38)
    phase = np.linspace(-0.3, 0.3, 360)
    arc = [1.3, 1.45] + np.column_stack((np.cos(phase), np.sin(phase))) * [0.3, 0.18]
    for points in (arc + rng.normal(0, 0.02, arc.shape), [1.6, 1.45] + rng.normal(0, 0.008, (360, 2))):
        result = estimate_center(points, method)
        assert result.n_input == result.n_used == 360
        assert result.status in ("computed", "nonellipse", "boundary", "not_converged")
        if result.center is not None:
            assert np.linalg.norm(result.center - [1.3, 1.45]) > 0.2


def test_invalid_api_inputs():
    with pytest.raises(ValueError, match="shape"):
        estimate_center(np.zeros(10), "legacy")
    with pytest.raises(ValueError, match="Unknown"):
        estimate_center(trajectory(), "unknown")
