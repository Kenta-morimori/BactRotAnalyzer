"""Interpolate unrequested center evaluations, with explicit failure barriers.

An invalid requested estimate is not an uncomputed evaluation. It remains
missing and splits interpolation support, as do original acquisition/identity
boundaries. No end holds, extrapolation, coordinate smoothing or seed changes.
"""

import numpy as np


def interpolate_center_anchors(frame, time, center, *, run_id, requested, max_gap_s, max_span_s):
    f, t, c = np.asarray(frame, float), np.asarray(time, float), np.asarray(center, float)
    n = len(t)
    ids, query = np.asarray(run_id), np.asarray(requested, bool)
    if f.shape != (n,) or c.shape != (n, 2) or ids.shape != (n,) or query.shape != (n,):
        raise ValueError("aligned original frames, times, centers and requests required")
    if not np.isfinite([max_gap_s, max_span_s]).all() or min(max_gap_s, max_span_s) <= 0:
        raise ValueError("explicit positive acquisition and interpolation spans required")
    finite = np.isfinite(c).all(axis=1)
    valid = (ids >= 0) & np.isfinite(f) & np.isfinite(t)
    if np.any(finite & (~query | ~valid)):
        raise ValueError("finite centers must be requested valid original observations")
    result = np.full((n, 2), np.nan)
    result[finite] = c[finite]
    interpolated = np.zeros(n, bool)
    left, right = np.full(n, -1, int), np.full(n, -1, int)
    left[finite] = right[finite] = np.flatnonzero(finite)
    reason = np.full(n, "no_bracketing_support", object)
    reason[~valid] = "unavailable_original_observation"
    reason[valid & query & ~finite] = "requested_estimate_unavailable"
    reason[finite] = "direct_estimate"
    # Known failures are barriers, not points to erase before searching anchors.
    usable = valid & ~(query & ~finite)
    previous = None
    for i in range(n):
        connected = (
            i > 0
            and usable[i - 1]
            and usable[i]
            and ids[i] == ids[i - 1]
            and f[i] == f[i - 1] + 1
            and 0 < t[i] - t[i - 1] <= max_gap_s
        )
        if not connected:
            previous = None
        if not usable[i] or not finite[i]:
            continue
        if previous is not None and i > previous + 1:
            middle = np.arange(previous + 1, i)
            if t[i] - t[previous] <= max_span_s:
                weight = (t[middle] - t[previous]) / (t[i] - t[previous])
                result[middle] = c[previous] + weight[:, None] * (c[i] - c[previous])
                interpolated[middle] = True
                left[middle], right[middle] = previous, i
                reason[middle] = "linear_time_interpolation"
            else:
                reason[middle] = "bracketing_span_exceeds_limit"
        previous = i
    return dict(center=result, interpolated=interpolated, reason=reason, left_support_row=left, right_support_row=right)
