"""Provisional mean-support completion inside continuous observation runs."""

from typing import Any

import numpy as np

from .raw_edges import edge_omega


def blocks(mask, ids):
    indices = np.flatnonzero(mask)
    cuts = np.flatnonzero((np.diff(indices) != 1) | (np.diff(ids[indices]) != 0)) + 1
    return [part for part in np.split(indices, cuts) if len(part)]


def complete_centers(time, frame, xy, baseline, direct, ids, *, points=3, max_gap_s):
    """Average up to N direct supports from each immediately adjacent finite block.

    Interpolate between mean centers at their mean timestamps. At an observed
    run end, hold the available side's mean. Never alter a finite baseline
    center or fill a missing observation; record support shortfalls explicitly.
    """
    if type(points) is not int or points < 1:
        raise ValueError("completion points must be a positive integer")
    t, f, xy, baseline = (np.asarray(v, float) for v in (time, frame, xy, baseline))
    ids, direct = np.asarray(ids), np.asarray(direct, bool)
    n = len(t)
    if any(v.shape != (n,) for v in (f, ids, direct)) or xy.shape != (n, 2) or baseline.shape != xy.shape:
        raise ValueError("aligned observations, centers and direct-support flags required")
    if not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError("positive explicit acquisition gap required")
    valid = (ids >= 0) & np.isfinite(xy).all(1) & np.isfinite(t) & np.isfinite(f)
    finite = np.isfinite(baseline).all(1) & valid
    direct = direct & finite
    run = np.full(n, -1, int)
    rid = -1
    for i in range(n):
        if not valid[i]:
            continue
        linked = (
            i > 0
            and valid[i - 1]
            and ids[i] == ids[i - 1]
            and f[i] == f[i - 1] + 1
            and 0 < t[i] - t[i - 1] <= max_gap_s
        )
        if not linked:
            rid += 1
        run[i] = rid
    output = np.full_like(baseline, np.nan)
    output[finite] = baseline[finite]
    status = np.full(n, "unavailable", object)
    status[finite] = "existing"
    supports: list[dict[str, Any]] = []
    for rows in blocks(valid & ~finite, run):
        start, end = int(rows[0]), int(rows[-1])
        left: list[int] = []
        right: list[int] = []
        for step, j, target in ((-1, start - 1, left), (1, end + 1, right)):
            while 0 <= j < n and run[j] == run[start] and finite[j]:
                if direct[j]:
                    target.append(j)
                if len(target) == points:
                    break
                j += step
        left.sort()
        right.sort()
        lc = np.mean(baseline[left], axis=0) if left else None
        rc = np.mean(baseline[right], axis=0) if right else None
        lt = float(np.mean(t[left])) if left else None
        rt = float(np.mean(t[right])) if right else None
        mode = "unavailable"
        if left and right:
            assert lc is not None and rc is not None and lt is not None and rt is not None
            output[rows] = lc + (rc - lc) * ((t[rows] - lt) / (rt - lt))[:, None]
            mode = "two_sided_mean_linear"
        elif left or right:
            output[rows] = lc if left else rc
            mode = "one_sided_mean_hold"
        status[rows] = mode
        supports.append(
            dict(
                block_id=len(supports),
                start_row=start,
                end_row=end,
                start_s=float(t[start]),
                end_s=float(t[end]),
                duration_s=float(t[end] - t[start]),
                run_id=int(run[start]),
                status=mode,
                requested_points=points,
                left_rows=left,
                right_rows=right,
                left_frames=f[left].astype(int).tolist(),
                right_frames=f[right].astype(int).tolist(),
                left_times_s=t[left].tolist(),
                right_times_s=t[right].tolist(),
                left_mean_time_s=lt,
                right_mean_time_s=rt,
                left_mean_um=None if lc is None else lc.tolist(),
                right_mean_um=None if rc is None else rc.tolist(),
                support_shortfall=bool((left and len(left) < points) or (right and len(right) < points)),
            )
        )
    omega = edge_omega(f, t, xy, output, max_gap_s=max_gap_s)
    same = (run[1:] >= 0) & (run[1:] == run[:-1])
    omega[1:][~same] = np.nan
    boundary = np.zeros(n, bool)
    boundary[1:] = (status[1:] != status[:-1]) & same & np.isfinite(omega[1:])
    return dict(center=output, omega=omega, status=status, boundary=boundary, blocks=supports, run=run)
