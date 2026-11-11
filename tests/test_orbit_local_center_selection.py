"""Local prediction selection retains original identities and moving-center truth."""

import numpy as np

from tests.synthetic.orbit_controls import make_radial_control
from utils.functions.orbit_center.local_center_selection import estimate_local_center


def settings(n):
    return dict(
        max_gap_s=0.005001,
        windows_s=[0.2, 0.4, 0.8],
        valid=np.ones(n, bool),
        segment_ids=np.zeros(n, int),
        relative_error_limit=0.05,
        relative_residual_limit=0.3,
        gaussian_se_multiplier=2.45,
    )


def test_accelerating_center_and_ellipse_radius_ramp_have_same_local_validation_rows():
    t = np.arange(161) * 0.005
    f = np.arange(1, len(t) + 1)
    c = np.c_[0.2 * t, 0.08 * t * t]
    radius = 0.7 + 0.2 * t
    phase = 2 * np.pi * 17 * t
    xy = c + radius[:, None] * np.c_[np.cos(phase), -0.4 * np.sin(phase)]
    query = np.zeros(len(t), bool)
    query[[0, 80, 160]] = True
    result = estimate_local_center(f, t, xy, query_mask=query, **settings(len(t)))
    np.testing.assert_allclose(result["center"][query], c[query], atol=1e-7)
    assert np.isnan(result["center"][~query]).all()
    for i in np.flatnonzero(query):
        row = result["records"][i]
        assert row["local_validation_first_source_frame"] <= f[i] <= row["local_validation_last_source_frame"]
        assert row["support_first_source_frame"] <= f[i] <= row["support_last_source_frame"]


def test_missing_observation_and_identity_break_split_every_support():
    t = np.arange(121) * 0.005
    f = np.arange(1, len(t) + 1)
    xy = np.c_[np.cos(2 * np.pi * 24 * t), -np.sin(2 * np.pi * 24 * t)]
    xy[60] = np.nan
    query = np.zeros(len(t), bool)
    query[[20, 60, 100]] = True
    before = xy.copy()
    result = estimate_local_center(f, t, xy, query_mask=query, **settings(len(t)))
    np.testing.assert_array_equal(xy, before)
    assert np.isnan(result["center"][60]).all() and np.isnan(result["computed_center"][60]).all()
    assert np.isnan(result["center"][~query]).all()
    for i in [20, 100]:
        row = result["records"][i]
        for candidate in row["candidates"]:
            assert not candidate["support_first_source_frame"] < 61 < candidate["support_last_source_frame"]
    assert result["run_id"][20] != result["run_id"][100]


def test_validation_guard_does_not_cross_a_source_identity_boundary():
    t = np.arange(161) * 0.005
    f = np.arange(1, len(t) + 1)
    xy = np.c_[np.cos(2 * np.pi * 24 * t), -np.sin(2 * np.pi * 24 * t)]
    seg = np.r_[np.zeros(80, int), np.ones(81, int)]
    xy[80:] += [2, 1]
    query = np.zeros(len(t), bool)
    query[[30, 110]] = True
    params = settings(len(t))
    params["segment_ids"] = seg
    result = estimate_local_center(f, t, xy, query_mask=query, validation_gap_rows=4, **params)
    assert np.isfinite(result["center"][query]).all()
    for i in np.flatnonzero(query):
        row = result["records"][i]
        assert row["validation_gap_rows"] == 4
        for candidate in row["candidates"]:
            assert not candidate["support_first_source_frame"] < 81 <= candidate["support_last_source_frame"]


def test_uncertain_predictor_does_not_hide_an_identified_center_candidate():
    z = make_radial_control(
        "colored_radius", speed_rev_s=9, dt_s=0.005, duration_s=8, amplitude=0.3, axis_ratio=1, seed=20261111
    )
    query = np.zeros(len(z["time"]), bool)
    query[[450]] = True
    params = settings(len(query))
    params["windows_s"] = [0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    r = estimate_local_center(
        z["source_frame"], z["time"], z["raw_xy"], query_mask=query, validation_gap_rows=4, **params
    )
    row = r["records"][450]
    eligible = [
        c
        for c in row["candidates"]
        if c["status"] == "computed" and c["relative_error_scale"] <= 0.05 and c["relative_residual"] <= 0.3
    ]
    assert eligible
    assert row["status"] == "candidate_supported"
    assert row["selection_pool_role"] == "identified_center_candidates"
    assert np.isfinite(r["center"][450]).all()


def test_chunked_query_evaluation_preserves_full_source_fit_and_missing_barriers():
    t = np.arange(121) * 0.005
    f = np.arange(1, len(t) + 1)
    rng = np.random.default_rng(41)
    xy = np.c_[0.1 * t + np.cos(2 * np.pi * 17 * t), -0.05 * t - 0.5 * np.sin(2 * np.pi * 17 * t)]
    xy += rng.normal(0, 0.002, xy.shape)
    xy[60] = np.nan
    params = settings(len(t))
    params["windows_s"] = [0.2, 0.4]
    expected = estimate_local_center(f, t, xy, validation_gap_rows=4, **params)
    computed = np.full_like(xy, np.nan)
    assessed = computed.copy()
    for first, last in [(0, 41), (41, 82), (82, len(t))]:
        query = np.zeros(len(t), bool)
        query[first:last] = True
        actual = estimate_local_center(f, t, xy, query_mask=query, validation_gap_rows=4, **params)
        computed[query] = actual["computed_center"][query]
        assessed[query] = actual["center"][query]
        np.testing.assert_array_equal(actual["run_id"], expected["run_id"])
        for i in np.flatnonzero(query):
            assert actual["records"][i]["status"] == expected["records"][i]["status"]
            assert actual["records"][i]["reason"] == expected["records"][i]["reason"]
    np.testing.assert_array_equal(computed, expected["computed_center"])
    np.testing.assert_array_equal(assessed, expected["center"])
