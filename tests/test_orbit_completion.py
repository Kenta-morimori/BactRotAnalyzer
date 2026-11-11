"""Support provenance and source barriers for mean-center completion."""

import numpy as np
import pytest

from utils.functions.orbit_center.completion import complete_centers


def example():
    t = np.arange(21) * 0.005
    frame = np.arange(1, 22)
    xy = np.c_[2 + np.cos(np.arange(21)), 3 + np.sin(np.arange(21))]
    center = np.full((21, 2), np.nan)
    center[[2, 3, 4, 16, 17, 18]] = [[1, 2], [2, 3], [3, 4], [7, 8], [8, 9], [9, 10]]
    return t, frame, xy, center, np.isfinite(center).all(1), np.zeros(21, int)


@pytest.mark.parametrize("points", [1, 3, 5])
def test_mean_support_time_and_shortfall(points):
    t, f, p, c, direct, ids = example()
    v = complete_centers(t, f, p, c, direct, ids, points=points, max_gap_s=0.005001)
    block = next(b for b in v["blocks"] if b["start_row"] == 5)
    left, right = block["left_rows"], block["right_rows"]
    expected = np.mean(c[left], 0) + (np.mean(c[right], 0) - np.mean(c[left], 0)) * (
        (t[10] - np.mean(t[left])) / (np.mean(t[right]) - np.mean(t[left]))
    )
    np.testing.assert_allclose(v["center"][10], expected)
    np.testing.assert_array_equal(v["center"][direct], c[direct])
    assert block["support_shortfall"] == (points == 5)
    assert v["status"][0] == "one_sided_mean_hold"


def test_gap_missing_identity_and_no_support_are_not_bridged():
    t, f, p, c, direct, ids = example()
    t[10:] += 2
    v = complete_centers(t, f, p, c, direct, ids, max_gap_s=0.005001)
    assert np.isnan(v["omega"][10])
    assert v["status"][9] == v["status"][10] == "one_sided_mean_hold"
    p[10] = np.nan
    v = complete_centers(t, f, p, c, direct, ids, max_gap_s=0.005001)
    assert np.isnan(v["center"][10]).all() and np.isnan(v["omega"][10:12]).all()
    v = complete_centers(t, f, p, np.full_like(c, np.nan), direct, ids, max_gap_s=0.005001)
    assert np.isnan(v["center"]).all()
    p[10] = 1
    t = np.arange(21) * 0.005
    ids[10:] = 1
    v = complete_centers(t, f, p, c, direct, ids, max_gap_s=0.005001)
    assert np.isnan(v["omega"][10])


def test_interpolated_points_and_distant_blocks_do_not_supply_means():
    t, f, p, c, direct, ids = example()
    c[5] = 100
    v = complete_centers(t, f, p, c, direct, ids, max_gap_s=0.005001)
    block = next(b for b in v["blocks"] if b["start_row"] == 6)
    assert block["left_rows"] == [2, 3, 4]
    c[6:10] = np.nan
    c[10] = 200
    direct[10] = True
    v = complete_centers(t, f, p, c, direct, ids, max_gap_s=0.005001)
    block = next(b for b in v["blocks"] if b["start_row"] == 11)
    assert block["left_rows"] == [10] and block["support_shortfall"]
