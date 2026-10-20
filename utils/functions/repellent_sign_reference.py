"""V2 production computation, with optional separate stopped-orbit measurement."""

import numpy as np
from scipy.signal import savgol_filter

from utils import param
from utils.functions.local_orbit_angular_velocity import _continuous_runs
from utils.functions.sign_reference_center_v2 import _phase, estimate_sign_reference_v2
from utils.functions.stopped_orbit_reference import estimate_stopped_reference


def analyze_sample(time, points, *, source_start=1, stop_measurement_enabled=True):
    """Compute one sample without writing files or modifying the primary V2 series."""
    time, points = np.asarray(time, float), np.asarray(points, float)
    frame = estimate_sign_reference_v2(time, points)
    anchors = frame.frame_0based.to_numpy(int)
    refs = frame[["reference_x_um", "reference_y_um"]].to_numpy(float)
    centers = np.full(points.shape, np.nan)
    centers[anchors] = refs
    accepted = frame.rotation_sign.notna().to_numpy()
    omega = np.full(len(frame), np.nan)
    angle = np.full(len(frame), np.nan)
    for start, end in _continuous_runs(time, points):
        smooth = savgol_filter(points[start:end], min(5, (end - start - 1) // 2 * 2 + 1), 2, axis=0)
        for j in np.flatnonzero((anchors >= start) & (anchors < end) & accepted):
            anchor = anchors[j]
            half = int(frame.window_frames.iloc[j]) // 2
            lower, upper = anchor - half, anchor + half + 1
            clock = time[lower:upper]
            sign, _, travel = _phase(smooth[lower - start : upper - start], clock, refs[j])
            omega[j] = sign * travel / (clock[-1] - clock[0])
            angle[j] = np.arctan2(*(points[anchor] - refs[j])[::-1])
    frame["angular_velocity_rad_s"] = omega
    frame["angle_rad"] = angle
    frame["source_frame_1based"] = anchors + source_start
    frame["observed_x_um"] = points[anchors, 0]
    frame["observed_y_um"] = points[anchors, 1]
    # Always supply a stable output schema, including empty/disabled samples.
    stopped, episodes = estimate_stopped_reference(time, points, frame=frame, enabled=stop_measurement_enabled)
    for column in stopped.columns.difference(frame.columns):
        frame[column] = stopped[column]
    return centers, frame, episodes


def analyze(day, time_list, x_list, y_list, stop_indices=None, source_starts=None, stop_measurement_enabled=None):
    """Adapt pure computations to the existing point-sampled downstream analysis.

    ``stop_indices`` is retained for call compatibility with older adapters.
    Stop measurement never supplies primary centers, signs, or velocities.
    """
    from utils.functions.rotation_center_output import save_results

    if stop_measurement_enabled is None:
        stop_measurement_enabled = param.stop_measurement_enabled
    result: dict[str, list] = {
        key: []
        for key in (
            "valid_indices",
            "time_list",
            "angle_list",
            "angular_velocity_list",
            "signed_angular_velocity_list",
            "source_frame_list",
        )
    }
    cx_all, cy_all, samples = [], [], []
    for sample, (clock, x, y) in enumerate(zip(time_list, x_list, y_list)):
        time = np.asarray(clock, float)
        points = np.column_stack((x, y)).astype(float)
        source_start = source_starts[sample] if source_starts else 1
        centers, frame, episodes = analyze_sample(
            time,
            points,
            source_start=source_start,
            stop_measurement_enabled=stop_measurement_enabled,
        )
        cx_all.append(centers[:, 0].tolist())
        cy_all.append(centers[:, 1].tolist())
        samples.append((time, points, frame, episodes))
        result["valid_indices"].append(sample)
        if frame.empty:
            # Keep the source sample and its clock visible in primary outputs.
            # No accepted window means every angle and velocity is missing.
            missing = np.full(len(time), np.nan).tolist()
            result["time_list"].append(time.tolist())
            result["angle_list"].append(missing)
            result["angular_velocity_list"].append(missing.copy())
            result["signed_angular_velocity_list"].append(missing.copy())
            result["source_frame_list"].append((np.arange(len(time)) + source_start).tolist())
        else:
            primary_time = []
            primary_angle = []
            primary_omega = []
            primary_source_frame = []
            previous_time = float("nan")
            previous_segment = -1
            for row in frame.itertuples():
                if previous_segment >= 0 and row.segment_id != previous_segment:
                    # One explicit missing point breaks all downstream windows
                    # spanning a coordinate or timestamp gap between runs.
                    primary_time.append((previous_time + row.time_sec) / 2)
                    primary_angle.append(np.nan)
                    primary_omega.append(np.nan)
                    primary_source_frame.append(np.nan)
                primary_time.append(row.time_sec)
                primary_angle.append(row.angle_rad)
                primary_omega.append(row.angular_velocity_rad_s)
                primary_source_frame.append(row.source_frame_1based)
                previous_time = row.time_sec
                previous_segment = row.segment_id
            result["time_list"].append(primary_time)
            result["angle_list"].append(primary_angle)
            result["angular_velocity_list"].append(primary_omega)
            result["signed_angular_velocity_list"].append(primary_omega.copy())
            result["source_frame_list"].append(primary_source_frame)
    save_results(day, samples, stop_measurement_enabled=stop_measurement_enabled)
    return cx_all, cy_all, result
