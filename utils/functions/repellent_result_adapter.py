"""Adapt existing analysis arrays to the grouped writer without refitting them."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from utils.functions.repellent_output import response_root


def collect_results(
    day,
    ranges,
    raw_frames,
    clocks,
    xs,
    ys,
    centers_x,
    centers_y,
    rotation,
    backgrounds,
    rises,
    estimator,
    *,
    source_ids=None,
):
    frames = []
    evaluated = {sample: k for k, sample in enumerate(rotation["valid_indices"])}
    for i, (clock, x, y, cx, cy) in enumerate(zip(clocks, xs, ys, centers_x, centers_y)):
        start = int(ranges[i]["source_start_frame_1based"])
        source = np.arange(len(clock)) + start
        raw = raw_frames[i].iloc[start - 1 : start - 1 + len(clock)]
        frame = pd.DataFrame(
            dict(
                sample_no=i + 1,
                source_frame_1based=source,
                time_sec=clock,
                raw_x_um=x,
                raw_y_um=y,
                center_x_um=cx,
                center_y_um=cy,
                background_intensity=backgrounds[i],
                angular_velocity_rad_s=np.nan,
            )
        )
        frame["centroid_missing_reason"] = np.where(
            np.isfinite(frame[["raw_x_um", "raw_y_um"]]).all(axis=1), "", "not_detected"
        )
        for column in ("missing_reason", "reason"):
            if column in raw:
                frame["centroid_missing_reason"] = np.where(
                    np.isfinite(frame[["raw_x_um", "raw_y_um"]]).all(axis=1),
                    "",
                    raw[column].fillna("").to_numpy(),
                )
                break
        frame["source_id"] = source_ids[i] if source_ids is not None else raw.attrs.get("source_id", f"sample_{i + 1}")
        frame["center_status"] = np.where(
            np.isfinite(frame[["center_x_um", "center_y_um"]]).all(axis=1), "direct", "unavailable"
        )
        if i in evaluated:
            k = evaluated[i]
            signed = rotation.get("signed_angular_velocity_list", rotation["angular_velocity_list"])
            velocity = np.asarray(signed[k], float)
            if "source_frame_list" in rotation:
                right = np.asarray(rotation["source_frame_list"][k], float)
            else:
                times = np.asarray(rotation["time_list"][k], float)
                if len(velocity) == len(times) - 1:
                    times = times[1:]
                if len(times) != len(velocity):
                    raise ValueError("angular velocity/time alignment is unavailable")
                # Exact source timestamps only; never infer missing source frame identities.
                unique = {float(t): int(f) for t, f in zip(clock, source) if np.isfinite(t)}
                if len(unique) != int(np.isfinite(clock).sum()):
                    raise ValueError("ambiguous duplicate source timestamps")
                right = np.asarray([unique.get(float(t), np.nan) for t in times])
            if len(right) != len(velocity):
                raise ValueError("angular velocity/source frame alignment differs")
            for f, value in zip(right, velocity):
                if np.isfinite(f):
                    if f != int(f) or not start <= f < start + len(clock):
                        raise ValueError("velocity source frame is outside selected observations")
                    frame.loc[int(f) - start, "angular_velocity_rad_s"] = -value  # existing backend is CW+
        # Preserve separate stop measurements; never use them for primary rotation statistics.
        if "reference_samples" in rotation:
            reference = rotation["reference_samples"][i][2]
            extras = [c for c in reference if c.startswith(("stop_", "measurement_", "final_reference_"))]
            for column in extras:
                frame[column] = np.nan if pd.api.types.is_numeric_dtype(reference[column]) else ""
                values = reference[column]
                if column.startswith("stop_angular_velocity"):
                    values = -values
                for source_no, value in zip(reference.source_frame_1based, values):
                    frame.loc[int(source_no) - start, column] = value
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True)
    cut = {
        str(i + 1): int(ranges[i]["source_start_frame_1based"]) + int(r["rise_index"])
        for i, r in enumerate(rises)
        if np.isfinite(r.get("rise_index", np.nan))
    }
    settings = dict(
        center_estimator=estimator,
        sign_convention="CCW_positive_image_y_down",
        input_backend_sign="CW_positive; numeric sign converted once at export adapter",
        series_role=f"existing_{estimator}_primary",
        physical_reversal_assessment="not_assessed",
        rise_source_frames=cut,
        rise_results=rises,
        source_frame_ranges=ranges,
        measurement_processing="existing backend unchanged; export performs no smoothing or fitting",
    )
    backend_settings = Path(response_root(day)) / "rotation_center/settings.json"
    if backend_settings.exists():
        settings["backend_settings"] = {
            k: v for k, v in json.loads(backend_settings.read_text()).items() if k != "run_id"
        }
    fluctuation, spectrum = _fluctuation_tables(response_root(day), observations)
    frequency = _switching_table(response_root(day), observations, cut)
    return dict(
        observations=observations, settings=settings, fluctuation=fluctuation, spectrum=spectrum, frequency=frequency
    )


def _switching_table(root, observations, rise_frames):
    """Retain existing primary-series counts/CW rates, without calling them physical reversals."""
    base = Path(root) / "00_all_rotational_analysis/angular_velocity"
    count_path, cw_path = base / "switching_count.csv", base / "cw_rate.csv"
    if not count_path.exists():
        return None
    counts = pd.read_csv(count_path, float_precision="round_trip")
    clockwise = pd.read_csv(cw_path, float_precision="round_trip") if cw_path.exists() else pd.DataFrame()
    rows = []
    for sample, original in observations.groupby("sample_no", sort=False):
        cut = rise_frames.get(str(sample))
        rise = original.loc[original.source_frame_1based.eq(cut)]
        key = f"No.{sample}"
        if rise.empty or key + "_time" not in counts or not np.isfinite(rise.time_sec.iloc[0]):
            continue
        t = original.time_sec.to_numpy(float)
        dt = np.diff(t)
        positive = dt[np.isfinite(dt) & (dt > 0)]
        limit = 1.5 * np.median(positive) if len(positive) else np.nan
        runs = np.r_[0, np.cumsum(~(np.isfinite(dt) & (dt > 0) & (dt <= limit)))]
        cw = dict(zip(clockwise[key + "_time"], clockwise[key + "_cw_rate"])) if key + "_time" in clockwise else {}
        for clock, count in zip(counts[key + "_time"], counts[key + "_count"]):
            if not np.isfinite(clock) or clock - 0.5 < rise.time_sec.iloc[0]:
                continue
            inside = np.flatnonzero((t >= clock - 0.5) & (t <= clock + 0.5))
            run = int(runs[inside[0]]) if len(inside) and np.all(runs[inside] == runs[inside[0]]) else np.nan
            rows.append(
                dict(
                    sample_no=sample,
                    time_sec=clock,
                    window_start_sec=clock - 0.5,
                    window_end_sec=clock + 0.5,
                    window_width_sec=1.0,
                    source_run_id=run,
                    window_start_source_frame_1based=(
                        int(original.source_frame_1based.iloc[inside[0]]) if len(inside) else np.nan
                    ),
                    window_end_source_frame_1based=(
                        int(original.source_frame_1based.iloc[inside[-1]]) if len(inside) else np.nan
                    ),
                    diagnostic_switch_count=count,
                    diagnostic_zero_crossing_rate_hz=count,
                    cw_fraction=cw.get(clock, np.nan),
                    frequency_hz=np.nan,
                    status="existing_primary_sign_change_statistic",
                    physical_reversal_assessment="not_assessed",
                )
            )
    return pd.DataFrame(rows) if rows else None


def _fluctuation_tables(root, observations):
    base = Path(root) / "02_pre_rise_fluctuation"
    mapping_path = base / "sample_index_map.csv"
    if not mapping_path.exists():
        return None, None
    mapping = pd.read_csv(mapping_path)
    mapping = dict(zip(mapping.pre_rise_fluctuation_no, mapping.original_sample_no))
    folder = base / "fluctuation_analysis/SD-time-series"
    sd_path, fft_path = folder / "SD_time_series.csv", folder / "SD-time-series_fft.csv"
    stats, spectra = [], []
    if sd_path.exists():
        sd = pd.read_csv(sd_path, float_precision="round_trip")
        for column in sd:
            match = re.fullmatch(r"No\.(\d+)_([\d.]+)s_sd", column)
            if not match:
                continue
            no, width = int(match[1]), float(match[2])
            sample = int(mapping[no])
            original = observations.loc[observations.sample_no.eq(sample)]
            # Preserve legacy metric times. Unmatched regenerated times have unknown frame identity.
            lookup = dict(zip(original.time_sec, original.source_frame_1based))
            time = sd[f"No.{no}_time"]
            count_key = f"No.{no}_{match[2]}s_data_num"
            count = sd[count_key] if count_key in sd else pd.Series(np.nan, index=sd.index)
            used = time.notna() | sd[column].notna() | count.notna()
            stats.append(
                pd.DataFrame(
                    dict(
                        sample_no=sample,
                        source_frame_1based=time[used].map(lookup),
                        time_sec=time[used],
                        sd_window_sec=width,
                        sd=sd[column][used],
                    )
                )
            )
    if fft_path.exists():
        fft = pd.read_csv(fft_path, float_precision="round_trip")
        for column in fft:
            match = re.fullmatch(r"No\.(\d+)_([\d.]+)s_freq", column)
            if match:
                no, width = int(match[1]), float(match[2])
                amp = fft[column.replace("_freq", "_Amp")]
                used = fft[column].notna() | amp.notna()
                spectra.append(
                    pd.DataFrame(
                        dict(
                            sample_no=int(mapping[no]),
                            sd_window_sec=width,
                            frequency_hz=fft[column][used],
                            amplitude=amp[used],
                        )
                    )
                )
    return (
        pd.concat(stats, ignore_index=True) if stats else None,
        pd.concat(spectra, ignore_index=True) if spectra else None,
    )
