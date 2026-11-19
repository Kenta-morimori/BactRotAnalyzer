import numpy as np
import pandas as pd

from utils.functions.poster_repellent_prepost import apply_masks, sign_change_summary


def _synthetic_series():
    return pd.DataFrame(
        {
            "source_frame_1based": [1, 2, 3, 4, 5, 7, 8, 9],
            "time_sec": [0, 0.01, 0.02, 0.03, 0.04, 0.06, 0.07, 0.08],
            "centroid_x": [1.0, 2.0, 100.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "centroid_y": [1.0] * 8,
            "center_x": [0.0] * 8,
            "center_y": [0.0] * 8,
            "angular_velocity_rad_s": [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            "background_intensity": [1.0] * 8,
        }
    )


def test_mask_preserves_frame_and_time_but_breaks_velocity_pairs():
    original = _synthetic_series()
    result = apply_masks(original, [[3, 3]])
    assert result.source_frame_1based.tolist() == original.source_frame_1based.tolist()
    assert result.time_sec.tolist() == original.time_sec.tolist()
    assert np.isnan(result.loc[2, "centroid_x"])
    assert np.isnan(result.loc[2, "angular_velocity_rad_s"])
    assert np.isnan(result.loc[3, "angular_velocity_rad_s"])
    # The source-frame gap 5 -> 7 also invalidates the first velocity after it.
    assert np.isnan(result.loc[5, "angular_velocity_rad_s"])


def test_sign_change_rate_does_not_bridge_missing_or_frame_gap():
    raw = _synthetic_series()
    clean = apply_masks(raw, [[3, 3]])
    result, bins = sign_change_summary(clean, 0, 0.09)
    # Frame 3 and the following velocity are missing. Frame 7 starts after a
    # recording gap. Only frames 8 -> 9 remain sign-eligible.
    assert result["sign_change_count"] == 1
    assert result["eligible_pair_count"] == 1
    assert np.isclose(result["eligible_seconds"], 0.01)
    assert np.isclose(result["sign_changes_per_observed_s"], 100.0)
    assert bins.sign_change_count.sum() == 1
