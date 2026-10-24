"""Final references exist independently of stopped-velocity quality."""

import numpy as np
import pandas as pd

from tests.synthetic.stops import Case, generate
from utils.functions.repellent_sign_reference import analyze_sample
from utils.functions.rotation_center_output import reference_counts
from utils.functions.stopped_orbit_reference import estimate_stopped_reference


def test_final_center_survives_candidate_boundary_and_quality_failure():
    for kind in ("static", "near_center"):
        data = generate(Case(kind, kind=kind), 10400)
        _, frame, episodes = analyze_sample(data["time"], data["observed"])
        interpolated = frame.final_reference_source.eq("stop_interpolated")
        assert interpolated.sum() > 30 and len(episodes) == 1
        boundary = interpolated & frame.stop_velocity_reasons.eq("candidate_boundary")
        assert boundary.any()
        assert frame.loc[boundary, "final_reference_x_um"].notna().all()
        assert frame.loc[boundary, "stop_angular_velocity_raw_rad_s"].isna().all()
        primary = frame.final_reference_source.eq("v2")
        np.testing.assert_equal(
            frame.loc[primary, "final_reference_x_um"].to_numpy(), frame.loc[primary, "reference_x_um"].to_numpy()
        )
        low = frame.stop_velocity_quality.eq("low_confidence")
        if kind == "near_center":
            assert low.any()
            assert frame.loc[low, "stop_angular_velocity_raw_rad_s"].notna().all()
            assert frame.loc[low, "stop_angular_velocity_rad_s"].isna().all()
            assert frame.loc[low, "measurement_center_x_um"].isna().all()
        good = frame.stop_velocity_quality.eq("qualified")
        np.testing.assert_equal(
            frame.loc[good, "stop_angular_velocity_raw_rad_s"].to_numpy(),
            frame.loc[good, "stop_angular_velocity_rad_s"].to_numpy(),
        )
        assert frame.loc[interpolated, "angular_velocity_rad_s"].isna().all()
        counts = reference_counts(frame)
        assert counts["final_reference_denominator"] == len(frame)
        assert sum(counts[f"final_reference_{s}_count"] for s in ("v2", "stop_interpolated", "missing")) == len(frame)
        assert (
            sum(counts[f"stop_quality_{s}_count"] for s in ("qualified", "low_confidence", "unavailable"))
            == counts["stop_quality_denominator"]
        )


def test_undefined_angle_and_multiple_quality_reasons(monkeypatch):
    # Known stationary reference, with a tiny radial excursion during the stop.
    import utils.functions.stopped_orbit_reference as module

    time = np.arange(201) / 200
    points = np.column_stack((0.01 + 0.02 * time, np.zeros(len(time))))
    anchors = np.arange(20, 181, 5)
    frame = pd.DataFrame(
        dict(
            segment_id=0,
            frame_0based=anchors,
            time_sec=time[anchors],
            rotation_sign=np.nan,
            rotation_state="unknown",
            decision_reason="no_rotation",
            window_frames=41,
            reference_x_um=np.nan,
            reference_y_um=np.nan,
        )
    )
    event = dict(
        episode_id=0,
        segment_id=0,
        start_s=0.1,
        end_s=0.9,
        duration_s=0.825,
        rows=np.arange(len(frame)),
        rotation_motion_scale_um=0.001,
    )
    monkeypatch.setattr(module, "label_candidates", lambda *args: (np.full(len(frame), "stop_candidate"), [event]))
    monkeypatch.setattr(
        module,
        "_anchor",
        lambda *args: dict(time=0.0 if args[4] == "left" else 1.0, center=np.zeros(2), spread=0.1, points=20),
    )
    result, _ = estimate_stopped_reference(time, points, frame=frame)
    low = result.stop_velocity_quality.eq("low_confidence")
    assert low.any()
    assert result.loc[low, "stop_velocity_reasons"].eq("unresolved_lever_arm;inconsistent_radial_motion").all()
    assert np.allclose(result.loc[low, "stop_angular_velocity_raw_rad_s"], 0)
    result, _ = estimate_stopped_reference(time, np.zeros_like(points), frame=frame)
    assert result.stop_angular_velocity_raw_rad_s.isna().all()
    assert result.stop_velocity_reasons.str.contains("undefined_angle").any()
    assert result.final_reference_x_um.notna().all()


def test_unavailable_centers_and_stable_disabled_empty_schema():
    for case in (Case("gap", kind="gap"), Case("one", kind="one_sided"), Case("long", stop_s=9)):
        data = generate(case, 10400)
        _, frame, _ = analyze_sample(data["time"], data["observed"])
        assert not frame.final_reference_source.eq("stop_interpolated").any()
        assert frame.stop_angular_velocity_raw_rad_s.isna().all()
    data = generate(Case("missing"), 10400)
    data["observed"][720:725] = np.nan
    _, frame, _ = analyze_sample(data["time"], data["observed"])
    assert not frame.final_reference_source.eq("stop_interpolated").any()
    _, disabled, _ = analyze_sample(data["time"], data["observed"], stop_measurement_enabled=False)
    assert disabled.stop_velocity_reasons.eq("disabled").all()
    _, empty, _ = analyze_sample(np.arange(3) / 200, np.zeros((3, 2)))
    assert empty.empty and set(disabled.columns) == set(empty.columns)
    counts = reference_counts(empty)
    assert counts["final_reference_denominator"] == counts["stop_quality_denominator"] == 0
    assert np.isnan(counts["stop_quality_qualified_fraction"])
