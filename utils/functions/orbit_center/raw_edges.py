"""Source identity and adjacent CCW angular measurement; no smoothing."""

import hashlib

import numpy as np


def input_hash(frame, time, xy):
    digest = hashlib.sha256()
    for item in [frame, time, xy]:
        array = np.asarray(item, dtype="<f8")
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def edge_omega(frame, time, xy, center, *, max_gap_s):
    """CCW-positive polar edge angle/time; invalid endpoints remain missing.

    Dynamic centers are an explicit diagnostic arm, not a production estimator.
    Moving the reference between frames contributes to the measured angle.
    """
    f, t, p = np.asarray(frame), np.asarray(time, float), np.asarray(xy, float)
    c = np.asarray(center, float)
    if p.shape != (len(t), 2) or f.shape != t.shape:
        raise ValueError("unaligned source arrays")
    if c.shape == (2,):
        c = np.broadcast_to(c, p.shape)
    if c.shape != p.shape or not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("invalid center shape or gap policy")
    r = p - c
    valid = np.isfinite(r).all(axis=1) & np.isfinite(t) & np.isfinite(f)
    valid &= np.linalg.norm(r, axis=1) > 0
    dt = np.diff(t)
    edges = valid[:-1] & valid[1:] & (np.diff(f) == 1) & (dt > 0) & (dt <= max_gap_s)
    cross = r[:-1, 0] * r[1:, 1] - r[:-1, 1] * r[1:, 0]
    dot = np.sum(r[:-1] * r[1:], axis=1)
    out = np.full(len(t), np.nan)
    out[1:][edges] = -np.arctan2(cross[edges], dot[edges]) / dt[edges]
    return out
