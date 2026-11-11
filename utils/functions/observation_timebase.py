"""Original-row continuity for repellent metrics."""

import numpy as np


def nominal_interval(time):
    """Median positive *adjacent* timestamp difference; never bridge a NaN."""
    dt = np.diff(np.asarray(time, dtype=float))
    positive = dt[np.isfinite(dt) & (dt > 0)]
    return float(np.median(positive)) if len(positive) else np.nan


def continuous_edges(time, *, source_frames=None, segment_ids=None, max_dt=None):
    t = np.asarray(time, float)
    dt = np.diff(t)
    limit = 1.5 * nominal_interval(t) if max_dt is None else max_dt
    edges = np.isfinite(dt) & (dt > 0) & (dt <= limit)
    if source_frames is not None:
        frames = np.asarray(source_frames, float)
        if len(frames) != len(t):
            raise ValueError("source frame/time lengths differ")
        edges &= np.diff(frames) == 1
    if segment_ids is not None:
        segments = np.asarray(segment_ids)
        if len(segments) != len(t):
            raise ValueError("segment/time lengths differ")
        edges &= segments[:-1] == segments[1:]
    return edges
