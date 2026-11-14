"""Phase-grouped local outputs from source-aligned, already computed results.

This module formats measurements; it does not fit centers, fill observations,
filter velocities, or decide physical reversals. Tables retain their source rows.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import param

ALL = "00_all_rotational_analysis"
PRE = "02_pre_rise_fluctuation"
POST = "03_post_rise_analysis"
KEYS = ["sample_no", "source_frame_1based", "time_sec"]
REQUIRED = KEYS + ["raw_x_um", "raw_y_um", "center_x_um", "center_y_um", "angular_velocity_rad_s"]
STATUS_COLORS = {
    "short_interpolation": ("#79a987", "Short interpolation"),
    "two_sided_mean_linear": ("#d55e00", "Provisional two-sided fill"),
    "one_sided_mean_hold": ("#d6a021", "One-sided hold"),
    "unavailable": ("#b74242", "Center unavailable"),
}


def plot_log_spectrum(axis, frequency, amplitude, **kwargs):
    """Log-log display only: preserve excluded bins in saved numerical tables."""
    f, a = np.asarray(frequency, float), np.asarray(amplitude, float)
    if f.shape != a.shape:
        raise ValueError("frequency/amplitude lengths differ")
    valid = np.isfinite(f) & np.isfinite(a) & (f > 0) & (a > 0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.plot(f[valid], a[valid], **kwargs)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Amplitude")
    axis.grid(True, which="both", alpha=0.2)
    if not valid.any():
        axis.text(0.5, 0.5, "No positive finite spectrum", ha="center", transform=axis.transAxes)
    return valid


def _validate(table):
    missing = set(REQUIRED) - set(table)
    if missing:
        raise ValueError(f"missing result columns: {sorted(missing)}")
    if table.empty or table["sample_no"].isna().any():
        raise ValueError("at least one identified sample is required")
    if table.duplicated(["sample_no", "source_frame_1based"]).any():
        raise ValueError("duplicate sample/source frame")
    for _, group in table.groupby("sample_no", sort=False):
        frames = group.source_frame_1based.to_numpy(float)
        if not np.isfinite(frames).all() or (frames <= 0).any() or (frames != np.floor(frames)).any():
            raise ValueError("positive integer source frames required")
        if (np.diff(frames) <= 0).any():
            raise ValueError("source frames must retain their original increasing order")


def _edges(group):
    if group.empty:
        return np.empty(0, dtype=bool)
    t = group.time_sec.to_numpy(float)
    f = group.source_frame_1based.to_numpy(float)
    dt = np.diff(t)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    limit = (
        float(group.max_gap_s.iloc[0])
        if "max_gap_s" in group
        else (1.5 * np.median(positive) if len(positive) else np.nan)
    )
    edges = np.isfinite(dt) & (dt > 0) & (dt <= limit) & (np.diff(f) == 1)
    for key in ("source_run_id", "target_id"):
        if key in group:
            v = group[key].to_numpy()
            edges &= (v[:-1] == v[1:]) & pd.notna(v[:-1]) & pd.notna(v[1:])
    return edges


def _parts(group, columns):
    values = group[columns].to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & np.isfinite(group.time_sec.to_numpy(float))
    for prefix in ("raw", "center", "center_before"):
        pair = [f"{prefix}_x_um", f"{prefix}_y_um"]
        if any(c in columns for c in pair) and set(pair).issubset(group):
            valid &= np.isfinite(group[pair].to_numpy(float)).all(axis=1)
    edges = _edges(group)
    current: list[int] = []
    for i in range(len(group)):
        if not valid[i] or (current and not edges[i - 1]):
            if current:
                yield np.asarray(current)
                current = []
        if valid[i]:
            current.append(i)
    if current:
        yield np.asarray(current)


def _line(axis, group, columns, *, xy=False, **kwargs):
    first = True
    label = kwargs.pop("label", None)
    kwargs.setdefault("markersize", 2)
    for part in _parts(group, columns):
        vals = group.iloc[part]
        x = vals[columns[0]] if xy else vals.time_sec
        y = vals[columns[1]] if xy else vals[columns[0]]
        axis.plot(x, y, label=label if first else "_nolegend_", marker="." if len(part) == 1 else None, **kwargs)
        first = False


def _bands(axis, group):
    t = group.time_sec.to_numpy(float)
    finite_time = np.isfinite(t)
    status = group.center_status.to_numpy()
    edges = _edges(group)
    for key, (color, label) in STATUS_COLORS.items():
        mask = (status == key) & finite_time
        indices = np.flatnonzero(mask)
        cuts = (
            np.flatnonzero((np.diff(indices) != 1) | (~edges[indices[:-1]])) + 1 if len(indices) else np.array([], int)
        )
        for k, part in enumerate(np.split(indices, cuts)):
            if len(part):
                if len(part) == 1:
                    axis.axvline(t[part[0]], color=color, alpha=0.3, linewidth=1)
                else:
                    axis.axvspan(t[part[0]], t[part[-1]], color=color, alpha=0.13, label=label if k == 0 else None)
    for i in np.flatnonzero(~edges):
        if np.isfinite(t[i : i + 2]).all() and t[i + 1] > t[i]:
            axis.axvspan(t[i], t[i + 1], color="#c8ccd0", alpha=0.55)
    if "centroid_missing_reason" in group:
        reasons = group.centroid_missing_reason.fillna("")
        for reason in reasons[reasons.ne("")].unique():
            times = t[reasons.eq(reason).to_numpy() & finite_time]
            axis.plot(
                times,
                np.full(len(times), 0.03),
                "|",
                transform=axis.get_xaxis_transform(),
                label=f"Centroid missing: {reason}",
                markersize=7,
            )


def _pages(groups, size):
    for start in range(0, len(groups), size):
        yield start // size + 1, groups[start : start + size]


def _page_name(name, page, count):
    path = Path(name)
    return path.with_name(f"{path.stem}_part{page:02d}{path.suffix}") if count > 1 else path


def _figure_end(fig, filename):
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.text(
        0.01,
        0.004,
        "Positions: μm | angular velocity: CCW positive | original source frames/times retained",
        fontsize=8,
    )
    fig.savefig(filename, dpi=150)
    plt.close(fig)


def _coordinate_figures(groups, save, size, labeled, *, before=False):
    count = math.ceil(len(groups) / size)
    suffix = "_labeled" if labeled else ""
    for page, batch in _pages(groups, size):
        fig, axes = plt.subplots(2 * len(batch), 1, figsize=(15, 4.6 * len(batch)), squeeze=False)
        for i, (no, group) in enumerate(batch):
            for j, coord in enumerate("xy"):
                ax = axes[2 * i + j, 0]
                _line(ax, group, [f"raw_{coord}_um"], color="#aab2b8", linewidth=0.6, label="Original centroid")
                if before:
                    _line(
                        ax,
                        group,
                        [f"center_before_{coord}_um"],
                        color="#246ca0",
                        linewidth=0.9,
                        label="Center candidate before correction",
                        linestyle="--",
                    )
                _line(
                    ax,
                    group,
                    [f"center_{coord}_um"],
                    color="#d55e00",
                    linewidth=1.0,
                    label="Final reference incl. provisional completion",
                )
                if labeled:
                    _bands(ax, group)
                ax.set(
                    title=f"No.{no}: {coord} / centroid and reference center",
                    ylabel=f"{coord} (μm)",
                    xlabel="Original time (s)",
                )
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7, loc="upper left", ncol=3)
        name = (
            f"details/center_before_after{suffix}.png"
            if before
            else f"{ALL}/centroid_coordinate/centroid_center_time_series{suffix}.png"
        )
        save(
            _page_name(name, page, count),
            fig,
            "positions before/after" if before else "centroid and center time series",
        )


def _standard_figures(groups, save, size):
    count = math.ceil(len(groups) / size)
    for labeled in (False, True):
        _coordinate_figures(groups, save, size, labeled)
        suffix = "_labeled" if labeled else ""
        for page, batch in _pages(groups, size):
            cols = min(2, len(batch))
            fig, axes = plt.subplots(
                math.ceil(len(batch) / cols),
                cols,
                figsize=(6.5 * cols, 4.2 * math.ceil(len(batch) / cols)),
                squeeze=False,
            )
            for ax in axes.flat:
                ax.set_visible(False)
            for ax, (no, group) in zip(axes.flat, batch):
                ax.set_visible(True)
                _line(ax, group, ["raw_x_um", "raw_y_um"], xy=True, color="#aab2b8", linewidth=0.35, label="Centroid")
                _line(
                    ax,
                    group,
                    ["center_x_um", "center_y_um"],
                    xy=True,
                    color="#d55e00",
                    linewidth=1,
                    label="Reference center",
                )
                if labeled:
                    for status, (color, label) in STATUS_COLORS.items():
                        selected = group.loc[group.center_status.eq(status)]
                        ax.scatter(selected.center_x_um, selected.center_y_um, s=3, color=color, label=label)
                ax.set(title=f"No.{no}: full-record XY", xlabel="x (μm)", ylabel="y (μm)", aspect="equal")
                ax.invert_yaxis()
                ax.legend(fontsize=7)
            save(
                _page_name(f"{ALL}/centroid_coordinate/trajectory_with_center{suffix}.png", page, count),
                fig,
                "XY centroid and reference centers",
            )
            fig, axes = plt.subplots(len(batch), 1, figsize=(15, 3.0 * len(batch)), squeeze=False)
            for ax, (no, group) in zip(axes.flat, batch):
                _line(ax, group, ["angular_velocity_rad_s"], color="#246ca0", linewidth=0.7)
                ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
                if labeled:
                    _bands(ax, group)
                ax.set(title=f"No.{no}: angular velocity", xlabel="Original time (s)", ylabel="rad/s")
                ax.grid(alpha=0.2)
                handles, _ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(fontsize=7, ncol=3)
            save(
                _page_name(f"{ALL}/angular_velocity/angular_velocity_time_series{suffix}.png", page, count),
                fig,
                "angular velocity; zero reference",
            )


def export_results(
    root,
    observations,
    *,
    settings,
    supports=None,
    fluctuation=None,
    spectrum=None,
    candidates=None,
    frequency=None,
    fft_features=None,
):
    """Write the compact layout, with all samples in each table/panel figure.

    ``settings`` must declare the supplied series role and sign. Optional tables
    are precomputed measurements; absent analyses are recorded, never fabricated.
    Only files owned by a previous manifest are replaced/removed on a successful
    export; unknown files and inputs are preserved.
    """
    if settings.get("sign_convention") != "CCW_positive_image_y_down":
        raise ValueError("explicit CCW-positive input required; exporter never flips measurements")
    if not settings.get("series_role"):
        raise ValueError("explicit primary/provisional series role required")
    _validate(observations)
    if settings.get("angular_sampling") == "adjacent_source_frames":
        for _, group in observations.groupby("sample_no", sort=False):
            required = group[["raw_x_um", "raw_y_um", "center_x_um", "center_y_um", "time_sec"]].to_numpy(float)
            valid = np.isfinite(required).all(axis=1)
            valid &= np.linalg.norm(required[:, :2] - required[:, 2:4], axis=1) > 0
            allowed = np.r_[False, _edges(group) & valid[:-1] & valid[1:]]
            if np.any(np.isfinite(group.angular_velocity_rad_s.to_numpy(float)) & ~allowed):
                raise ValueError("finite adjacent angular velocity at an unavailable observation/reference edge")
    table = observations.copy(deep=True)
    if "center_status" not in table:
        table["center_status"] = np.where(
            table[["center_x_um", "center_y_um"]].notna().all(axis=1), "direct", "unavailable"
        )
    if "centroid_missing_reason" not in table:
        table["centroid_missing_reason"] = np.where(
            table[["raw_x_um", "raw_y_um"]].isna().any(axis=1), "unavailable_unspecified", ""
        )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    old_manifest = root / "output_manifest.csv"
    previous = pd.read_csv(old_manifest).path.tolist() if old_manifest.exists() else []
    owned = []

    def write(relative, data, purpose):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False, float_format="%.17g")
        owned.append(dict(path=str(relative), purpose=purpose, kind="table"))

    def save(relative, fig, purpose):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        _figure_end(fig, path)
        owned.append(dict(path=str(relative), purpose=purpose, kind="figure"))

    groups = list(table.groupby("sample_no", sort=False))
    size = int(settings.get("samples_per_page", param.repellent_samples_per_page))
    if size < 1:
        raise ValueError("samples_per_page must be positive")
    sample_cols = [c for c in ("sample_no", "source_id", "centroid_method") if c in table]
    write("sample_index_map.csv", table[sample_cols].drop_duplicates(), "original sample/input identities")
    write("00_time_list/time_list.csv", table[KEYS], "unchanged source frame/time rows")
    raw_cols = KEYS + [c for c in ("raw_x_um", "raw_y_um", "centroid_missing_reason", "source_run_id") if c in table]
    write(
        f"{ALL}/centroid_coordinate/centroid_time_series.csv", table[raw_cols], "original centroid observations in μm"
    )
    write(
        f"{ALL}/centroid_coordinate/centroid_missing_reasons.csv",
        table[KEYS + ["centroid_missing_reason"]],
        "reason retained for every source row",
    )
    center_cols = (
        KEYS + ["center_x_um", "center_y_um"] + [c for c in table if c.startswith(("measurement_", "final_reference_"))]
    )
    write(
        f"{ALL}/center_coordinate/center_coordinate.csv",
        table[center_cols],
        "supplied reference centers; separate stop references retained",
    )
    write(
        f"{ALL}/center_coordinate/center_status.csv",
        table[KEYS + ["center_status"] + [c for c in table if c.startswith(("center_estimate_", "center_review_"))]],
        "direct/short/provisional/one-sided/unavailable",
    )
    support_cols = KEYS + [c for c in table if c.startswith("support_")]
    write(
        f"{ALL}/center_coordinate/center_support.csv",
        supports if supports is not None else table[support_cols],
        "support provenance; no synthetic support points",
    )
    angular_cols = KEYS + [
        c
        for c in (
            "angular_velocity_rad_s",
            "primary_angular_velocity_rad_s",
            "uncompleted_angular_velocity_rad_s",
            "completion_boundary_edge",
        )
        if c in table
    ]
    angular_cols += [c for c in table if c.startswith("stop_")]
    write(
        f"{ALL}/angular_velocity/angular_velocity_time_series.csv",
        table[angular_cols],
        "supplied angular velocity values, not filtered",
    )
    _standard_figures(groups, save, size)
    ranges, rises = [], []
    rise_map = settings.get("rise_source_frames", {})
    for no, group in groups:
        cut = rise_map.get(str(no))
        for phase, mask in (
            ("all", np.ones(len(group), bool)),
            ("pre", group.source_frame_1based.lt(cut) if cut is not None else np.zeros(len(group), bool)),
            ("post", group.source_frame_1based.ge(cut) if cut is not None else np.zeros(len(group), bool)),
        ):
            selected = group.loc[mask]
            ranges.append(
                dict(
                    sample_no=no,
                    phase=phase,
                    status="selected" if len(selected) else "unavailable",
                    reason="" if len(selected) else "rise_not_provided_or_empty_range",
                    source_start_frame_1based=selected.source_frame_1based.iloc[0] if len(selected) else np.nan,
                    source_end_frame_1based=selected.source_frame_1based.iloc[-1] if len(selected) else np.nan,
                    start_time_sec=selected.time_sec.iloc[0] if len(selected) else np.nan,
                    end_time_sec=selected.time_sec.iloc[-1] if len(selected) else np.nan,
                )
            )
        rise = group.loc[group.source_frame_1based.eq(cut)] if cut is not None else group.iloc[:0]
        rises.append(
            dict(
                sample_no=no,
                rise_source_frame_1based=cut,
                rise_time_sec=rise.time_sec.iloc[0] if len(rise) else np.nan,
                status="provided" if cut is not None else "unavailable",
            )
        )
        detection = settings.get("image_input", {}).get("rise_results", {}).get(str(no))
        if detection is not None:
            rises[-1].update(rise_detection_method=detection["method"], rise_detection_status=detection["status"])
    write("00_time_list/analysis_frame_ranges.csv", pd.DataFrame(ranges), "source-frame phase boundaries")
    write("01_brightness_change/rise_summary.csv", pd.DataFrame(rises), "provided rise boundary; no redetection")
    if "background_intensity" in table:
        write(
            "01_brightness_change/background_intensity_time_series.csv",
            table[KEYS + ["background_intensity"]],
            "original background measurements",
        )
        for page, batch in _pages(groups, size):
            fig, axes = plt.subplots(len(batch), 1, figsize=(15, 3 * len(batch)), squeeze=False)
            for ax, (no, group) in zip(axes.flat, batch):
                _line(ax, group, ["background_intensity"], color="#666", linewidth=0.7)
                cut = rise_map.get(str(no))
                rise = group.loc[group.source_frame_1based.eq(cut)]
                if len(rise) and np.isfinite(rise.time_sec.iloc[0]):
                    ax.axvline(rise.time_sec.iloc[0], color="#d55e00", linestyle="--")
                ax.set(title=f"No.{no}: background intensity", xlabel="Original time (s)", ylabel="Intensity")
            save(
                _page_name(
                    "01_brightness_change/background_intensity_time_series.png", page, math.ceil(len(groups) / size)
                ),
                fig,
                "background and provided rise time",
            )
    status = {"background": "provided" if "background_intensity" in table else "not_provided"}
    if fluctuation is not None:
        write(
            f"{PRE}/fluctuation_analysis/SD_time_series.csv",
            fluctuation,
            "provided SD; window and original times preserved",
        )
        _metric_figures(fluctuation, "sd", "SD", f"{PRE}/fluctuation_analysis/SD_time_series.png", save, size)
    status["pre_rise_fluctuation"] = "provided" if fluctuation is not None else "not_provided"
    if spectrum is not None:
        write(
            f"{PRE}/fluctuation_analysis/SD_fft.csv", spectrum, "unmodified spectrum, including zero/nonpositive bins"
        )
        _spectrum_figures(spectrum, save, size)
    status["fft"] = "provided_loglog_display" if spectrum is not None else "not_provided"
    if fft_features is not None:
        write(
            f"{PRE}/fluctuation_analysis/FFT_features.csv", fft_features, "FFT support, ratios and unavailable reasons"
        )
    if candidates is None:
        candidates = pd.DataFrame(
            [dict(sample_no=no, status="not_assessed", physical_reversal_assessment="not_assessed") for no, _ in groups]
        )
    write(
        f"{POST}/switching_analysis/reversal_candidates.csv",
        candidates,
        "candidate status; not automatically physical reversals",
    )
    if frequency is None:
        frequency = pd.DataFrame([dict(sample_no=no, frequency_hz=np.nan, status="not_assessed") for no, _ in groups])
    write(
        f"{POST}/switching_analysis/reversal_frequency.csv", frequency, "supplied frequency; unassessed remains missing"
    )
    _post_figures(groups, candidates, frequency, save, size, rise_map)
    status["physical_reversal"] = settings.get("physical_reversal_assessment", "not_assessed")
    if settings.get("details", param.repellent_details_enabled):
        if {"center_before_x_um", "center_before_y_um"}.issubset(table):
            write(
                "details/center_before_correction.csv",
                table[KEYS + [c for c in table if c.startswith("center_before_")]],
                "center candidates before correction; see settings definition",
            )
            for labeled in (False, True):
                _coordinate_figures(groups, save, size, labeled, before=True)
        _zoom_figures(groups, settings.get("zoom_ranges", []), save)
    status["tracking_examples"] = "not_provided"
    result_settings = dict(
        settings,
        execution_time=datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
        output_schema="phase_grouped_v1",
        status="success",
        analysis_status=status,
        sample_numbers=[int(no) for no, _ in groups],
        coordinate_unit="um",
        fft_axes="loglog",
        samples_per_page=size,
        details=bool(settings.get("details", param.repellent_details_enabled)),
    )
    result_settings = _json_safe(result_settings)
    (root / "settings.json").write_text(
        json.dumps(result_settings, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    owned.append(
        dict(path="settings.json", purpose="effective settings, provenance and execution time", kind="settings")
    )
    gallery = []
    for row in owned:
        name = html.escape(row["path"])
        gallery.append(f'<li><a href="{name}">{name}</a></li>')
        if row["kind"] == "figure":
            gallery.append(
                f'<a href="{name}"><img loading="lazy" src="{name}" alt="{html.escape(row["purpose"])}"></a>'
            )
    (root / "index.html").write_text(
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>忌避応答解析の出力一覧</title><style>body{max-width:1400px;margin:24px auto;font-family:system-ui;line-height:1.6;padding:0 20px}img{max-width:100%;margin:12px 0 30px}li{overflow-wrap:anywhere}</style><h1>忌避応答解析：出力一覧</h1><p>図は複数個体を集約。位置μm・角速度CCW＋。補完の状態はラベル付き図を参照。</p><p>Series role: '
        + html.escape(settings["series_role"])
        + ". Physical reversal: "
        + html.escape(status["physical_reversal"])
        + ".</p><ul>"
        + "".join(gallery)
        + "</ul></html>"
    )
    owned.append(dict(path="index.html", purpose="local output gallery", kind="index"))
    write(
        "output_manifest.csv",
        pd.DataFrame(owned + [dict(path="output_manifest.csv", purpose="complete output inventory", kind="manifest")]),
        "output inventory",
    )
    current_paths = {r["path"] for r in owned}
    for relative in previous:
        stale = root / relative
        if (
            relative not in current_paths
            and stale.is_relative_to(root)
            and ".." not in Path(relative).parts
            and stale.is_file()
        ):
            stale.unlink()
    return result_settings


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _metric_figures(data, column, ylabel, filename, save, size):
    groups = list(data.groupby("sample_no", sort=False))
    for page, batch in _pages(groups, size):
        fig, axes = plt.subplots(len(batch), 1, figsize=(15, 3 * len(batch)), squeeze=False)
        for ax, (no, group) in zip(axes.flat, batch):
            for width, part in group.groupby("sd_window_sec", sort=False):
                _line(ax, part, [column], linewidth=0.7, label=f"{width:g} s")
            ax.set(title=f"No.{no}: fluctuation before rise", xlabel="Original time (s)", ylabel=ylabel)
            ax.legend(fontsize=8, ncol=6)
        save(_page_name(filename, page, math.ceil(len(groups) / size)), fig, "pre-rise fluctuation")


def _spectrum_figures(data, save, size):
    groups = list(data.groupby("sample_no", sort=False))
    for page, batch in _pages(groups, size):
        fig, axes = plt.subplots(len(batch), 1, figsize=(13, 4 * len(batch)), squeeze=False)
        for ax, (no, group) in zip(axes.flat, batch):
            for width, part in group.groupby("sd_window_sec", sort=False):
                plot_log_spectrum(ax, part.frequency_hz, part.amplitude, linewidth=0.8, label=f"SD {width:g} s")
            ax.set_title(f"No.{no}: SD spectrum before rise / log-log")
            ax.legend(fontsize=8, ncol=3)
        save(
            _page_name(f"{PRE}/fluctuation_analysis/SD_fft_loglog.png", page, math.ceil(len(groups) / size)),
            fig,
            "FFT log-log; numerical spectrum unchanged",
        )


def _post_figures(groups, candidates, frequency, save, size, rise_map):
    for page, batch in _pages(groups, size):
        fig, axes = plt.subplots(len(batch), 1, figsize=(15, 3 * len(batch)), squeeze=False)
        for ax, (no, group) in zip(axes.flat, batch):
            cut = rise_map.get(str(no))
            part = group.loc[group.source_frame_1based.ge(cut)] if cut is not None else group.iloc[:0]
            _line(ax, part, ["angular_velocity_rad_s"], linewidth=0.6, color="#246ca0")
            selected = candidates.loc[candidates.sample_no.eq(no)]
            if "time_sec" in selected and len(part):
                selected = selected.loc[selected.time_sec.between(part.time_sec.min(), part.time_sec.max())]
                ax.plot(
                    selected.time_sec,
                    np.full(len(selected), 0.025),
                    "|",
                    color="#8b49a7",
                    transform=ax.get_xaxis_transform(),
                    markersize=4,
                    label="Diagnostic sign-change candidates; not confirmed reversals",
                )
                ax.legend(fontsize=7, loc="upper right")
            ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
            if not len(part):
                ax.text(0.5, 0.5, "Not executed: rise boundary unavailable", transform=ax.transAxes, ha="center")
            ax.set(
                title=f"No.{no}: post-rise angular velocity / physical reversals not inferred by exporter",
                xlabel="Original time (s)",
                ylabel="rad/s",
            )
        save(
            _page_name(f"{POST}/switching_analysis/switching_time_series.png", page, math.ceil(len(groups) / size)),
            fig,
            "post-rise velocity and switching assessment",
        )
        fig, axes = plt.subplots(len(batch), 1, figsize=(15, 3 * len(batch)), squeeze=False)
        for ax, (no, _) in zip(axes.flat, batch):
            part = frequency.loc[frequency.sample_no.eq(no)]
            if {"time_sec", "frequency_hz"}.issubset(part) and part.frequency_hz.notna().any():
                _line(ax, part, ["frequency_hz"], linewidth=0.7)
            elif {"time_sec", "diagnostic_zero_crossing_rate_hz"}.issubset(part):
                _frequency_line(ax, part, "diagnostic_zero_crossing_rate_hz")
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, "Physical reversal frequency: not assessed", transform=ax.transAxes, ha="center")
            ax.set(title=f"No.{no}: reversal frequency", xlabel="Original time (s)", ylabel="Hz")
        save(
            _page_name(f"{POST}/switching_analysis/reversal_frequency.png", page, math.ceil(len(groups) / size)),
            fig,
            "reversal frequency or explicit unassessed status",
        )


def _frequency_line(axis, group, column):
    label = "Diagnostic zero-crossing rate; physical reversals not assessed"
    if "source_frame_1based" in group:
        _line(axis, group, [column], color="#8b49a7", linewidth=0.7, label=label)
        return
    # Window centers need not coincide with an original frame. Never invent a
    # source frame for them: use explicit window-support runs to break lines.
    t, value = group.time_sec.to_numpy(float), group[column].to_numpy(float)
    valid = np.isfinite(t) & np.isfinite(value)
    run = group.source_run_id.to_numpy() if "source_run_id" in group else np.zeros(len(group))
    valid &= pd.notna(run)
    ix = np.flatnonzero(valid)
    positive = np.diff(t)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    limit = 1.5 * np.median(positive) if len(positive) else np.nan
    split = (np.diff(ix) != 1) | (run[ix[1:]] != run[ix[:-1]])
    split |= (np.diff(t[ix]) <= 0) | (np.diff(t[ix]) > limit)
    for number, part in enumerate(np.split(ix, np.flatnonzero(split) + 1)):
        if len(part):
            axis.plot(
                t[part],
                value[part],
                color="#8b49a7",
                linewidth=0.7,
                marker="." if len(part) == 1 else None,
                label=label if number == 0 else None,
            )


def _zoom_figures(groups, ranges, save):
    views = []
    for view in ranges:
        for no, group in groups:
            if no == view["sample_no"]:
                selected = group.loc[group.time_sec.between(view["start_s"], view["end_s"])]
                if len(selected):
                    views.append((view, selected))
    for page, batch in _pages(views, 3):
        fig, axes = plt.subplots(len(batch), 1, figsize=(15, 3.3 * len(batch)), squeeze=False)
        for ax, (view, group) in zip(axes.flat, batch):
            _line(ax, group, ["angular_velocity_rad_s"], color="#246ca0", linewidth=0.8)
            _bands(ax, group)
            ax.axhline(0, color="black", linestyle="--", linewidth=0.9)
            ax.set(
                title=f"No.{view['sample_no']}: {view.get('label', 'stop/reversal candidate')}",
                xlabel="Original time (s)",
                ylabel="rad/s",
            )
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7)
        save(
            _page_name("details/stop_reversal_zooms.png", page, math.ceil(len(views) / 3)),
            fig,
            "source-aligned stop/reversal candidate zooms",
        )


def export_bundle(bundle_path, root):
    """Read a portable JSON descriptor; relative table paths resolve at the JSON."""
    path = Path(bundle_path)
    bundle = json.loads(path.read_text())
    tables = {}
    for name in ("observations", "supports", "fluctuation", "spectrum", "candidates", "frequency", "fft_features"):
        if name in bundle:
            source = Path(bundle[name])
            if not source.is_absolute():
                source = path.parent / source
            tables[name] = pd.read_csv(source, float_precision="round_trip")
    if "observations" not in tables:
        raise ValueError("bundle observations table is required")
    return export_results(root, settings=bundle["settings"], **tables)
