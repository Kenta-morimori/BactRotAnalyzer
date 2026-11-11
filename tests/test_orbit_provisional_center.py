import numpy as np

from utils.functions.orbit_center.local_center_selection import estimate_local_center
from utils.functions.orbit_center.provisional_center import (
    configuration,
    estimate_provisional_center,
)


def data():
    t = np.arange(161) * 0.005
    f = np.arange(1, len(t) + 1)
    truth = np.c_[0.2 * t, -0.1 * t]
    p = truth + np.c_[np.cos(2 * np.pi * 24 * t), -0.5 * np.sin(2 * np.pi * 24 * t)]
    return f, t, p, truth


def test_reference_wrapper_preserves_direct_results_and_real_time_short_fills():
    f, t, p, truth = data()
    before = p.copy()
    valid = np.ones(len(t), bool)
    ids = np.zeros(len(t), int)
    r = estimate_provisional_center(f, t, p, valid=valid, segment_ids=ids, frame_period_s=0.005, chunk_size=7)
    cfg = configuration()["parameters"]
    keys = [
        "windows_s",
        "center_degrees",
        "relative_error_limit",
        "relative_residual_limit",
        "gaussian_se_multiplier",
        "validation_gap_rows",
    ]
    original = estimate_local_center(
        f,
        t,
        p,
        valid=valid,
        segment_ids=ids,
        max_gap_s=0.005001,
        query_mask=r["evaluated"],
        **{k: cfg[k] for k in keys}
    )
    np.testing.assert_array_equal(r["conditional_center"], original["center"])
    np.testing.assert_allclose(r["completed_center"], truth, atol=1e-8)
    np.testing.assert_array_equal(p, before)
    assert r["interpolated"].any() and r["unqueried_center_fit_validity"] == "not assessed"


def test_full_audit_and_original_missing_identity_barriers_are_preserved():
    f, t, p, truth = data()
    p[80] = np.nan
    valid = np.isfinite(p).all(axis=1)
    ids = np.r_[np.zeros(81, int), np.ones(80, int)]
    r = estimate_provisional_center(
        f, t, p, valid=valid, segment_ids=ids, frame_period_s=0.005, evaluation_step=1, chunk_size=21
    )
    assert r["evaluated"].sum() == valid.sum()
    assert np.isnan(r["completed_center"][80]).all() and r["run_id"][20] != r["run_id"][120]
    for i in np.flatnonzero(r["interpolated"]):
        a, b = r["left_support_row"][i], r["right_support_row"][i]
        assert r["run_id"][a] == r["run_id"][i] == r["run_id"][b]
        assert r["evaluated"][i] and np.isfinite(r["conditional_center"][i]).all()


def test_configuration_copy_cannot_mutate_reference_parameters():
    cfg = configuration()
    cfg["parameters"]["windows_s"][0] = 999
    assert configuration()["parameters"]["windows_s"][0] == 0.2


def test_model_choice_review_keeps_numerical_evidence_but_blocks_interpolation(monkeypatch):
    import utils.functions.orbit_center.provisional_center as module

    n = 12
    t = np.arange(n) * 0.005
    f = np.arange(1, n + 1)
    p = np.c_[np.cos(2 * np.pi * 24 * t), -np.sin(2 * np.pi * 24 * t)]

    def estimator(frame, time, xy, *, query_mask, **kwargs):
        c = np.full((n, 2), np.nan)
        c[query_mask] = 0
        records = []
        for i in range(n):
            window = 0.4 if i < 4 else 6.4
            records.append(
                dict(
                    source_row=i,
                    status="candidate_supported",
                    reason=None,
                    selected_window_s=window,
                    selected_center_degree=1,
                    candidates=[
                        dict(
                            status="computed",
                            window_s=0.4,
                            center_degree=1,
                            center=np.array([0.1, 0]) if i >= 4 else np.zeros(2),
                            projected_major_semiaxis=1.0,
                        ),
                        dict(
                            status="computed",
                            window_s=6.4,
                            center_degree=1,
                            center=np.zeros(2),
                            projected_major_semiaxis=1.0,
                        ),
                    ],
                )
            )
        return dict(run_id=np.zeros(n, int), computed_center=c, center=c.copy(), records=records)

    monkeypatch.setattr(module, "estimate_local_center", estimator)
    r = module.estimate_provisional_center(
        f, t, p, valid=np.ones(n, bool), segment_ids=np.zeros(n, int), frame_period_s=0.005, evaluation_step=1
    )
    assert r["review_model_switch"][4] and np.isfinite(r["conditional_center"][4]).all()
    assert np.isnan(r["completed_center"][4]).all()
    assert not r["interpolated"].any()


def test_parameter_default_evaluation_step_is_used_unless_explicitly_overridden():
    f, t, p, _ = data()
    kwargs = dict(
        valid=np.ones(len(t), bool),
        segment_ids=np.zeros(len(t), int),
        frame_period_s=0.005,
        parameters={"default_evaluation_step_frames": 1},
    )
    dense = estimate_provisional_center(f, t, p, **kwargs)
    assert dense["evaluated"].all() and dense["evaluation_step"] == 1
    coarse = estimate_provisional_center(f, t, p, evaluation_step=20, **kwargs)
    assert coarse["evaluated"].sum() < len(t) and coarse["evaluation_step"] == 20
