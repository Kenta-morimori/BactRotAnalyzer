"""Stop measurement must preserve abstention and keep observations separate."""

import numpy as np

from tests.synthetic.stops import Case, generate
from utils.functions.stopped_orbit_reference import estimate_stopped_reference


def test_stop_measurement_is_not_forced_to_zero_and_v2_stays_missing():
    data = generate(Case("quiet"), 10400)
    frame, events = estimate_stopped_reference(data["time"], data["observed"])
    measured = frame.stop_angular_velocity_rad_s.notna()
    assert measured.sum() > 20 and len(events) == 1
    assert frame.loc[measured, "rotation_sign"].isna().all()
    assert (abs(frame.loc[measured, "stop_angular_velocity_rad_s"]) > 1e-6).any()
    assert frame.loc[measured, "motion_label"].eq("stop_candidate").all()


def test_coincident_center_is_labeled_but_not_measured():
    data = generate(Case("center", kind="center"), 10400)
    frame, events = estimate_stopped_reference(data["time"], data["observed"])
    assert len(events) == 1
    assert frame.motion_label.eq("stop_candidate").sum() > 20
    assert frame.stop_angular_velocity_rad_s.isna().all()
    assert frame.measurement_reason.eq("unresolved_lever_arm").any()


def test_shrinking_radius_rotation_is_not_called_stopped():
    data = generate(Case("shrink", kind="radius_shrink"), 10400)
    frame, _ = estimate_stopped_reference(data["time"], data["observed"])
    assert not frame.motion_label.eq("stop_candidate").any()
    assert frame.motion_label.eq("compact_periodic_motion").any()


def test_timestamp_gap_never_gets_one_interpolated_episode():
    data = generate(Case("gap", kind="gap"), 10400)
    frame, events = estimate_stopped_reference(data["time"], data["observed"])
    for event in events.itertuples():
        assert not event.start_s < 3.6 < event.end_s
    measured = frame.stop_angular_velocity_rad_s.notna()
    assert np.isfinite(frame.loc[measured, "measurement_center_x_um"]).all()
    # Each side of a gap lacks one bracketing rotation reference.
    assert not measured.any()


def test_one_sided_stop_requires_brackets_for_interpolation():
    data = generate(Case("last_stop", kind="one_sided"), 10400)
    result, events = estimate_stopped_reference(data["time"], data["observed"], strategy="interpolate")
    assert not events.empty
    assert result.stop_angular_velocity_rad_s.isna().all()
    assert result.measurement_reason.eq("missing_right_reference").any()


def test_long_stop_is_explicitly_deferred():
    data = generate(Case("long", stop_s=9), 10400)
    result, _ = estimate_stopped_reference(data["time"], data["observed"])
    assert result.measurement_reason.eq("stop_too_long").any()
    assert result.stop_angular_velocity_rad_s.isna().all()


def test_reference_search_skips_isolated_points_before_dense_rotation():
    import pandas as pd

    from utils.functions.stopped_orbit_reference import StopReferenceConfig, _anchor

    time = np.arange(0, 8, 0.005)
    points = np.column_stack((1.3 + 0.3 * np.cos(2 * np.pi * 24 * time), 1.45 + 0.18 * np.sin(2 * np.pi * 24 * time)))
    clocks = np.r_[4.4, 4.425, np.arange(5.5, 6.21, 0.025)]
    frame = pd.DataFrame(
        dict(
            segment_id=0,
            rotation_sign=-1,
            time_sec=clocks,
            frame_0based=np.round(clocks / 0.005).astype(int),
            reference_x_um=1.3,
            reference_y_um=1.45,
        )
    )
    event = dict(segment_id=0, start_s=3.0, end_s=4.2, rotation_motion_scale_um=0.2)
    anchor = _anchor(time, points, frame, event, "right", "v2", StopReferenceConfig())
    assert anchor is not None and anchor["points"] >= 12
    assert anchor["time"] >= 5.5
