"""Known-onset synthetic validation, separate from exploratory seeds/real references."""

import numpy as np
import pytest

from utils.functions.background_rise import RiseSettings, detect_sustained_rise


def signal(dt=0.05, seed=42079):
    time = np.arange(0, 50, dt)
    baseline = 100 + np.random.default_rng(seed).normal(0, 0.25, len(time))
    return time, baseline


@pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
@pytest.mark.parametrize("kind", ["abrupt", "gradual", "very_gradual", "multistage", "decay"])
def test_known_onset_returns_original_observation_not_peak(dt, kind):
    t, y = signal(dt)

    def ramp(start, duration, height):
        return height * np.clip((t - start) / duration, 0, 1)

    if kind == "abrupt":
        y += ramp(15, 0.1, 20)
    elif kind == "gradual":
        y += ramp(15, 8, 20)
    elif kind == "very_gradual":
        y += ramp(15, 15, 20)
    elif kind == "multistage":
        y += ramp(15, 0.1, 8) + ramp(18, 5, 12)
    else:
        y += 20 * np.exp(-np.maximum(t - 15, 0) / 30) * (t >= 15)
    frames = np.arange(2001, 2001 + len(t))
    original = y.copy()
    result = detect_sustained_rise(y, t + 80, source_frames=frames)
    assert result["status"] == "detected", result
    assert abs(result["rise_time_sec"] - 95) <= 0.5
    assert result["rise_source_frame"] == frames[result["rise_index"]]
    assert result["rise_time_sec"] == t[result["rise_index"]] + 80
    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize("kind", ["flat", "drift", "fall", "single_spike", "spike_train", "two_second_peak"])
def test_endpoint_outlier_and_drift_rejection(kind):
    t, y = signal()
    if kind == "drift":
        y += 0.4 * t
    elif kind == "fall":
        y -= 0.4 * t
    elif kind == "single_spike":
        y[300] += 100
    elif kind == "spike_train":
        y[200:800:41] += 100
    elif kind == "two_second_peak":
        y[(t >= 15) & (t < 17)] += 40
    result = detect_sustained_rise(y, t)
    assert result["status"] == "not_detected", result
    assert not np.isfinite(result["rise_index"])
    assert result["rise_source_frame"] is None and result["rise_time_sec"] is None


@pytest.mark.parametrize("kind", ["baseline", "followup", "all_missing", "no_time"])
def test_insufficient_evidence_is_unavailable(kind):
    t, y = signal()
    if kind == "baseline":
        y += 20 * (t >= 1)
    elif kind == "followup":
        y += 20 * (t >= 46)
    elif kind == "all_missing":
        y[:] = np.nan
    else:
        t[:] = np.nan
    result = detect_sustained_rise(y, t)
    assert result["status"] == "not_detected"
    assert result["rise_time_sec"] is None


@pytest.mark.parametrize("kind", ["missing_value", "missing_time", "time_gap", "source_gap", "segment"])
def test_onset_in_unavailable_region_cannot_be_reconstructed(kind):
    t, y = signal()
    y += 20 * (t >= 15)
    frames = np.arange(1, len(t) + 1)
    segments = np.zeros(len(t), int)
    if kind == "missing_value":
        y[(t >= 14) & (t <= 16)] = np.nan
    elif kind == "missing_time":
        t[(t >= 14) & (t <= 16)] = np.nan
    elif kind == "time_gap":
        t[t >= 15] += 8
    elif kind == "source_gap":
        frames[t >= 15] += 100
    else:
        segments[t >= 15] = 1
    result = detect_sustained_rise(y, t, source_frames=frames, segment_ids=segments)
    assert result["status"] == "not_detected", result
    assert "gap" in result["reason"] or "continuous" in result["reason"]


def test_multiple_separated_changes_are_ambiguous():
    t, y = signal()
    y += 20 * ((t >= 12) & (t < 26)) + 40 * (t >= 33)
    result = detect_sustained_rise(y, t)
    assert result["status"] == "ambiguous" and result["reason"] == "multiple_candidates"
    assert len(result["candidates"]) == 2 and result["rise_time_sec"] is None


def test_gap_before_independently_supported_change_keeps_source_mapping():
    t, y = signal()
    y += 20 * (t >= 25)
    y[50] = np.nan
    result = detect_sustained_rise(y, t, source_frames=np.arange(101, 101 + len(t)))
    assert result["status"] == "detected"
    assert result["rise_source_frame"] == 101 + result["rise_index"]
    assert abs(result["rise_time_sec"] - 25) <= 0.25
    assert result["continuous_runs"] == 2


def test_invalid_parameters_and_alignment_fail_explicitly():
    with pytest.raises(ValueError):
        RiseSettings(persistence_sec=0)
    with pytest.raises(ValueError):
        RiseSettings(persistence_fraction=1.2)
    with pytest.raises(ValueError):
        detect_sustained_rise([1, 2], [0])
    with pytest.raises(ValueError):
        detect_sustained_rise([1, 2], [0, 1], source_frames=[1, 1])


def test_public_api_requires_clock_and_preserves_explicit_legacy_values():
    from utils.functions.repellent_response import (
        _detect_rise_index_legacy,
        detect_rise_index,
        detect_rise_points,
    )

    t, y = signal()
    y += 20 * (t >= 15)
    unavailable = detect_rise_index(y)
    assert unavailable["reason"] == "original_timestamps_required"
    old = detect_rise_index(y, method="legacy", sigma_threshold=2.5)
    assert old == _detect_rise_index_legacy(y, sigma_threshold=2.5)
    row = detect_rise_points([y], time_list=[t], source_frame_list=[np.arange(501, 501 + len(t))])[0]
    assert row["sample_no"] == 1 and row["num_frames"] == len(y)
    assert row["rise_source_frame"] == row["rise_index"] + 501


def test_manual_time_config_is_distinct_from_retained_automatic_diagnostics(monkeypatch):
    from utils import param
    from utils.functions.repellent_response import (
        add_rise_time_to_results,
        detect_rise_index,
    )

    t, y = signal()
    y += 20 * (t >= 15)
    row = detect_rise_index(y, times=t)
    monkeypatch.setattr(param, "get_flag_use_manual_rise_time", lambda *a, **k: True)
    monkeypatch.setattr(param, "get_manual_rise_time_sec_config", lambda *a: [16.0])
    add_rise_time_to_results("synthetic", [row], [t])
    assert row["reason"] == "manual_time_config" and row["status"] == "provided"
    assert row["rise_time"] == 16.0
    assert row["automatic_detection"]["rise_time_sec"] == 15.0
