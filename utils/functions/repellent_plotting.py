"""Plots consume saved reference/quality values and preserve missing observations."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from utils.functions.repellent_output import annotate_figure, response_root

PRIMARY = "#286b9c"
STOP = "#d38128"


def continuous_parts(time, values, segments=None):
    """Yield finite consecutive slices, including singletons; never compact NaNs."""
    time, values = np.asarray(time, float), np.asarray(values, float)
    if not len(time):
        return
    values = values.reshape(len(time), -1)
    finite = np.isfinite(time) & np.isfinite(values).all(axis=1)
    dt = np.diff(time)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    nominal = np.median(positive) if len(positive) else np.inf
    breaks = ~finite[:-1] | ~finite[1:] | ~np.isfinite(dt) | (dt <= 0) | (dt > 1.5 * nominal)
    if segments is not None:
        breaks |= np.diff(np.asarray(segments)) != 0
    starts = np.r_[0, np.flatnonzero(breaks) + 1]
    ends = np.r_[np.flatnonzero(breaks) + 1, len(time)]
    for lower, upper in zip(starts, ends):
        if finite[lower]:
            yield int(lower), int(upper)


def plot_series(axis, time, values, *, segments=None, xy=False, **kwargs):
    label = kwargs.pop("label", None)
    kwargs.setdefault("marker", ".")
    kwargs.setdefault("markersize", 2)
    color = kwargs.pop("color", None)
    time, values = np.asarray(time, float), np.asarray(values, float)
    for number, (lower, upper) in enumerate(continuous_parts(time, values, segments)):
        part = values[lower:upper]
        x, y = (part[:, 0], part[:, 1]) if xy else (time[lower:upper], part)
        (line,) = axis.plot(x, y, label=label if number == 0 else "_nolegend_", color=color, **kwargs)
        color = line.get_color()


def coordinate_component(points, component):
    """An unavailable coordinate pair breaks both component traces."""
    points = np.asarray(points, float)
    return np.where(np.isfinite(points).all(axis=1), points[:, component], np.nan)


def recording_gaps(axes, time, points):
    time = np.asarray(time, float)
    parts = list(continuous_parts(time, points))
    intervals = [(time[a - 1], time[b]) for (_, a), (b, _) in zip(parts, parts[1:])]
    finite_time = time[np.isfinite(time)]
    if len(finite_time):
        if not parts:
            intervals.append((finite_time[0], finite_time[-1]))
        else:
            if parts[0][0] > 0:
                intervals.append((finite_time[0], time[parts[0][0]]))
            if parts[-1][1] < len(time):
                intervals.append((time[parts[-1][1] - 1], finite_time[-1]))
    for axis in axes:
        for lower, upper in intervals:
            if np.isfinite(lower) and np.isfinite(upper) and lower < upper:
                axis.axvspan(lower, upper, color="#dddddd", alpha=0.6, zorder=0)
        axis.grid(alpha=0.15)


def candidate_bands(axes, episodes, bounds=None):
    for event in episodes.itertuples():
        lower, upper = event.start_s, event.end_s
        if bounds is not None:
            lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
        if lower <= upper:
            for axis in axes:
                axis.axvspan(lower, upper, color=STOP, alpha=0.12, zorder=0)


def reference_series(axis, frame, *, component=None, components=False):
    columns = ["final_reference_x_um", "final_reference_y_um"]
    values = frame[columns].to_numpy(float) if component is None else frame[columns[component]].to_numpy(float)
    groups = (
        [("v2", PRIMARY, "V2 reference"), ("stop_interpolated", STOP, "Interpolated reference")]
        if components
        else [("", PRIMARY, "Final reference")]
    )
    for source, color, label in groups:
        selected = values.copy()
        if source:
            selected[~frame.final_reference_source.eq(source).to_numpy()] = np.nan
        plot_series(
            axis,
            frame.time_sec,
            selected,
            segments=frame.segment_id,
            xy=component is None,
            color=color,
            lw=1,
            label=label,
        )


def velocities(axis, frame):
    plot_series(
        axis,
        frame.time_sec,
        frame.angular_velocity_rad_s,
        segments=frame.segment_id,
        color=PRIMARY,
        lw=0.65,
        label="Primary V2",
    )
    for quality, marker, style, label in (
        ("qualified", ".", "-", "Stop: qualified"),
        ("low_confidence", "x", "None", "Stop: low confidence"),
    ):
        values = frame.stop_angular_velocity_raw_rad_s.where(frame.stop_velocity_quality.eq(quality))
        plot_series(
            axis,
            frame.time_sec,
            values,
            segments=frame.segment_id,
            color=STOP,
            marker=marker,
            linestyle=style,
            lw=0.8,
            markersize=3,
            label=label,
        )
    handles = [
        Line2D([], [], color=PRIMARY, label="Primary V2"),
        Line2D([], [], color=STOP, marker=".", label="Stop: qualified"),
        Line2D([], [], color=STOP, marker="x", linestyle="None", label="Stop: low confidence"),
        Line2D([], [], color="none", label="Unavailable: no value"),
    ]
    axis.legend(handles=handles, loc="upper right", fontsize=7)
    axis.set_ylabel("CW+ ω (rad/s)")


def save_figure(fig, path, *, day=None, samples=None):
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    annotate_figure(fig, day, samples)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def phase_figures(day, samples, rise_times, *, source_starts=None, rise_indices=None, sample_numbers=None):
    """Render phase subsets of the saved tables, with the standard panel pagination."""
    from utils.functions.make_graph import _page_path, _sample_pages

    sample_numbers = list(range(1, len(samples) + 1)) if sample_numbers is None else list(sample_numbers)
    if len(sample_numbers) != len(samples) or len(set(sample_numbers)) != len(samples):
        raise ValueError("sample_numbers must contain one distinct original sample number per sample")
    for phase, subdir in [
        ("all", "00_all_rotational_analysis"),
        ("pre", "02_pre_rise_fluctuation"),
        ("post", "03_post_rise_analysis"),
    ]:
        out = response_root(day) / subdir
        subsets = []
        for sample_index, (time, points, frame, episodes) in enumerate(samples):
            sample_no = sample_numbers[sample_index]
            time, points = np.asarray(time, float), np.asarray(points, float)
            rise = rise_times[sample_index]
            start = source_starts[sample_index] if source_starts else 1
            cut = (
                int(rise_indices[sample_index])
                if rise_indices is not None and np.isfinite(rise_indices[sample_index])
                else (int(np.searchsorted(time, rise)) if np.isfinite(rise) else len(time))
            )
            # Evaluation rows use the same original-frame boundary as the observations,
            # including when the rise frame has no timestamp.
            table_cut = int(np.searchsorted(frame.frame_0based.to_numpy(int), cut))
            selected = np.zeros(len(time), bool)
            chosen = np.zeros(len(frame), bool)
            if phase == "all":
                selected[:], chosen[:] = True, True
            elif phase == "pre":
                selected[:cut], chosen[:table_cut] = True, True
            else:
                selected[cut:], chosen[table_cut:] = True, True
            clock, observed, table = time[selected], points[selected], frame.loc[chosen].copy()
            tables = out / "rotation_center" / f"no{sample_no:02d}"
            tables.mkdir(parents=True, exist_ok=True)
            table.to_csv(tables / "timeseries.csv", index=False)
            bounds = (float(np.nanmin(clock)), float(np.nanmax(clock))) if np.isfinite(clock).any() else None
            events = (
                episodes
                if bounds is None
                else episodes[(episodes.end_s >= bounds[0]) & (episodes.start_s <= bounds[1])]
            )
            if not len(clock):
                events = episodes.iloc[:0]
            events.to_csv(tables / "stop_episodes.csv", index=False)
            sources = np.flatnonzero(selected) + start
            source_label = f"{sources[0]}-{sources[-1]}" if len(sources) else "empty"
            title = f"No.{sample_no} | {phase} | source frames {source_label}"
            subsets.append((clock, observed, table, events, bounds, rise, title))
        for page, lower, upper in _sample_pages(len(samples)):
            for center_only in (False, True):
                for components in (False, True):
                    fig, axes = plt.subplots(upper - lower, 3, figsize=(16, max(4.8, (upper - lower) * 4)))
                    axes = np.atleast_2d(axes)
                    for row, (clock, observed, table, events, bounds, rise, title) in enumerate(subsets[lower:upper]):
                        for component, axis in enumerate(axes[row, 1:]):
                            if not center_only:
                                plot_series(
                                    axis,
                                    clock,
                                    coordinate_component(observed, component),
                                    color="#bfc6cb",
                                    lw=0.35,
                                    label="Centroid",
                                )
                            reference_series(axis, table, component=component, components=components)
                            axis.set(xlabel="Time (s)", ylabel=["x (µm)", "y (µm)"][component])
                            if np.isfinite(rise) and phase == "all":
                                axis.axvline(rise, color="red", ls="--", lw=0.7)
                        if not center_only:
                            plot_series(
                                axes[row, 0], clock, observed, xy=True, color="#bfc6cb", lw=0.35, label="Centroid"
                            )
                        reference_series(axes[row, 0], table, components=components)
                        axes[row, 0].set(xlabel="x (µm)", ylabel="y (µm)", aspect="equal", title=title)
                        candidate_bands(axes[row, 1:], events, bounds)
                        recording_gaps(axes[row, 1:], clock, observed)
                        handles, _ = axes[row, 0].get_legend_handles_labels()
                        if handles:
                            axes[row, 0].legend(fontsize=7)
                    name = "rotation_center" if center_only else "centroid_before"
                    if components:
                        name += "_components"
                    folder = out / "centroid_coordinate"
                    folder.mkdir(exist_ok=True)
                    save_figure(
                        fig,
                        _page_path(str(folder), name + ".png", len(samples), page),
                        day=day,
                        samples=sample_numbers[lower:upper],
                    )
            fig, axes = plt.subplots(upper - lower, 1, figsize=(14, max(4, (upper - lower) * 3.5)))
            for axis, (clock, observed, table, events, bounds, rise, title) in zip(
                np.atleast_1d(axes), subsets[lower:upper]
            ):
                velocities(axis, table)
                candidate_bands([axis], events, bounds)
                recording_gaps([axis], clock, observed)
                axis.set(xlabel="Time (s); grey: missing observations / recording gaps", title=title)
            fig.suptitle("Stop values are excluded from primary statistics")
            folder = out / "angular_velocity"
            folder.mkdir(exist_ok=True)
            save_figure(
                fig,
                _page_path(str(folder), "angular_velocity_with_stop.png", len(samples), page),
                day=day,
                samples=sample_numbers[lower:upper],
            )
