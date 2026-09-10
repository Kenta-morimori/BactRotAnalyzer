import numpy as np

from utils.functions import repellent_response


def _rotating_then_stopped_series(stopped: bool = True):
    frames = 1800
    dt = 0.02
    time = np.arange(frames, dtype=float) * dt
    rise_idx = 600
    stop_idx = 1000
    angle = 2.0 * np.pi * 8.0 * time
    x = 1.0 + 0.4 * np.cos(angle)
    y = 2.0 + 0.4 * np.sin(angle)
    if stopped:
        x[stop_idx:] = x[stop_idx]
        y[stop_idx:] = y[stop_idx]
    return time, x, y, rise_idx, stop_idx


def test_detected_stop_freezes_center_at_last_active_rotation():
    time, x, y, rise_idx, expected_stop_start = _rotating_then_stopped_series()
    stop_idx = repellent_response.detect_rotation_stop_indices(
        [time], [x], [y], [rise_idx], activity_ratio=0.2, min_duration_rotations=1.0
    )[0]

    assert np.isfinite(stop_idx)
    assert expected_stop_start <= stop_idx < expected_stop_start + 250

    cx_std = np.linspace(1.0, 2.0, len(time))
    cy_std = np.linspace(2.0, 3.0, len(time))
    cx, cy = repellent_response.apply_post_rise_center_strategy(
        [time], [x], [y], [cx_std], [cy_std], [rise_idx], stop_indices=[stop_idx]
    )
    detected = int(stop_idx)
    assert np.allclose(cx[0][detected:], cx[0][detected])
    assert np.allclose(cy[0][detected:], cy[0][detected])


def test_continued_rotation_and_missing_data_do_not_trigger_stop():
    time, x, y, rise_idx, _ = _rotating_then_stopped_series(stopped=False)
    continuous = repellent_response.detect_rotation_stop_indices([time], [x], [y], [rise_idx])[0]
    assert np.isnan(continuous)

    cx_std = np.linspace(1.0, 2.0, len(time))
    cy_std = np.linspace(2.0, 3.0, len(time))
    without_stop = repellent_response.apply_post_rise_center_strategy(
        [time], [x], [y], [cx_std], [cy_std], [rise_idx]
    )
    with_undetected_stop = repellent_response.apply_post_rise_center_strategy(
        [time], [x], [y], [cx_std], [cy_std], [rise_idx], stop_indices=[continuous]
    )
    assert np.allclose(without_stop[0][0], with_undetected_stop[0][0])
    assert np.allclose(without_stop[1][0], with_undetected_stop[1][0])

    x[rise_idx:] = np.nan
    y[rise_idx:] = np.nan
    missing = repellent_response.detect_rotation_stop_indices([time], [x], [y], [rise_idx])[0]
    assert np.isnan(missing)


def test_manual_stop_index_overrides_automatic_detection_and_is_validated():
    time, x, y, rise_idx, _ = _rotating_then_stopped_series(stopped=False)
    manual_idx = 1200
    stops, sources = repellent_response.resolve_rotation_stop_indices(
        [time], [rise_idx], [float("nan")], [manual_idx]
    )
    assert stops == [float(manual_idx)]
    assert sources == ["manual"]

    automatic_stops, automatic_sources = repellent_response.resolve_rotation_stop_indices(
        [time], [rise_idx], [1300.0], []
    )
    assert automatic_stops == [1300.0]
    assert automatic_sources == ["auto"]

    cx_std = np.linspace(1.0, 2.0, len(time))
    cy_std = np.linspace(2.0, 3.0, len(time))
    cx, cy = repellent_response.apply_post_rise_center_strategy(
        [time], [x], [y], [cx_std], [cy_std], [rise_idx], stop_indices=stops
    )
    assert np.allclose(cx[0][manual_idx:], cx[0][manual_idx])
    assert np.allclose(cy[0][manual_idx:], cy[0][manual_idx])
    assert np.isclose(cx[0][manual_idx], 1.0, atol=0.02)
    assert np.isclose(cy[0][manual_idx], 2.0, atol=0.02)

    with np.testing.assert_raises(ValueError):
        repellent_response.resolve_rotation_stop_indices([time], [rise_idx], [float("nan")], [rise_idx])
    with np.testing.assert_raises(ValueError):
        repellent_response.resolve_rotation_stop_indices([time], [rise_idx], [float("nan")], [len(time)])
