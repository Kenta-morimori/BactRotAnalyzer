import numpy as np

from utils.functions.local_orbit_angular_velocity import estimate_local_orbit_velocity


def test_noiseless_orbit_has_production_sign_and_approximate_speed():
    time = np.arange(1000) / 200
    omega = 20.0
    phase = omega * time
    points = np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase))) + [1.3, 1.45]

    result = estimate_local_orbit_velocity(time, points)

    assert len(result) > 180
    assert (result.signed_angular_velocity_rad_s < 0).all()  # CCW is negative
    assert 15 < result.signed_angular_velocity_rad_s.abs().median() < 28


def test_timestamp_gap_is_never_bridged():
    first = np.arange(400) / 200
    second = 10 + np.arange(400) / 200
    time = np.r_[first, second]
    phase = 20 * time
    points = np.column_stack((np.cos(phase), np.sin(phase)))

    result = estimate_local_orbit_velocity(time, points)

    assert set(result.segment_id) == {0, 1}
    assert result.loc[result.segment_id.eq(0), "time_sec"].max() < first[-1]
    assert result.loc[result.segment_id.eq(1), "time_sec"].min() > second[0]
    assert not result.time_sec.between(first[-1], second[0]).any()


def test_reversal_changes_sign_away_from_transition():
    time = np.arange(1200) / 200
    phase = np.where(time < 3, 20 * time, 60 - 20 * (time - 3))
    points = np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))

    result = estimate_local_orbit_velocity(time, points)
    before = result.loc[result.time_sec.between(1, 2), "signed_angular_velocity_rad_s"]
    after = result.loc[result.time_sec.between(4, 5), "signed_angular_velocity_rad_s"]

    assert (before < 0).mean() > 0.95
    assert (after > 0).mean() > 0.95
