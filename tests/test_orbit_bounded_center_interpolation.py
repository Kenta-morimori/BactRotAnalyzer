import numpy as np
import pytest

from utils.functions.orbit_center.bounded_center_interpolation import (
    interpolate_center_anchors,
)


def complete(t, c, query, **kwargs):
    n = len(t)
    return interpolate_center_anchors(
        np.arange(1, n + 1),
        t,
        c,
        run_id=kwargs.pop("run_id", np.zeros(n, int)),
        requested=query,
        max_gap_s=kwargs.pop("max_gap_s", 0.011),
        max_span_s=kwargs.pop("max_span_s", 0.05),
        **kwargs
    )


def test_real_time_convex_interpolation_preserves_anchors_and_no_end_hold():
    t = np.array([0, 0.01, 0.015, 0.025, 0.03])
    c = np.full((5, 2), np.nan)
    c[[1, 3]] = [[2, 6], [8, 0]]
    before = c.copy()
    r = complete(t, c, np.isfinite(c).all(axis=1))
    np.testing.assert_array_equal(c, before)
    np.testing.assert_array_equal(r["center"][[1, 3]], c[[1, 3]])
    np.testing.assert_allclose(r["center"][2], [4, 4])
    assert np.isnan(r["center"][[0, 4]]).all()
    assert r["left_support_row"][2] == 1 and r["right_support_row"][2] == 3


def test_requested_failure_is_not_filled_or_crossed():
    c = np.full((7, 2), np.nan)
    c[[0, 6]] = [[0, 0], [6, 6]]
    query = np.array([1, 0, 0, 1, 0, 0, 1], bool)
    r = complete(np.arange(7) * 0.005, c, query)
    assert np.isnan(r["center"][1:6]).all()
    assert r["reason"][3] == "requested_estimate_unavailable"


@pytest.mark.parametrize("boundary", ["identity", "missing", "time", "frame"])
def test_original_boundaries_cannot_be_crossed(boundary):
    t = np.arange(7) * 0.005
    f = np.arange(1, 8)
    ids = np.zeros(7, int)
    c = np.full((7, 2), np.nan)
    c[[0, 6]] = [[0, 0], [6, 6]]
    if boundary == "identity":
        ids[3:] = 1
    if boundary == "missing":
        ids[3] = -1
    if boundary == "time":
        t[3:] += 0.1
    if boundary == "frame":
        f[3:] += 1
    r = interpolate_center_anchors(
        f, t, c, run_id=ids, requested=np.isfinite(c).all(axis=1), max_gap_s=0.006, max_span_s=0.2
    )
    assert np.isnan(r["center"][1:6]).all()


def test_long_interval_stays_missing_and_unrequested_finite_center_is_error():
    c = np.full((9, 2), np.nan)
    c[[0, 8]] = [[0, 0], [8, 8]]
    query = np.isfinite(c).all(axis=1)
    r = complete(np.arange(9) * 0.01, c, query)
    assert np.isnan(r["center"][1:8]).all()
    assert r["reason"][4] == "bracketing_span_exceeds_limit"
    with pytest.raises(ValueError):
        complete(np.arange(9) * 0.01, c, np.zeros(9, bool))
