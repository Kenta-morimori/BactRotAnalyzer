import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import param
from utils.features import IGNORE_PLOT_COLS, ROTATION_FEATURES, SD_WIDTH_DEPEND_COLS
from utils.functions import rot_df_manage


def _get_axis(axs, idx):
    # Normalize axes access: handle both 2D grids and single-axes cases.
    ax_arr = axs.ravel() if hasattr(axs, "ravel") else np.array([axs])
    return ax_arr[idx]


def _compute_ylim_for_columns(df, day_list, columns):
    # Compute a shared y-range across related columns (aligns scales for fair comparison).
    vals_min, vals_max = np.inf, -np.inf
    for col in columns:
        s = df[df["day"].isin(day_list)][col].dropna()
        if len(s) == 0:
            continue
        vmin, vmax = float(np.min(s)), float(np.max(s))
        vals_min = min(vals_min, vmin)
        vals_max = max(vals_max, vmax)

    if not np.isfinite(vals_min) or not np.isfinite(vals_max):
        return None

    if np.isclose(vals_min, vals_max):
        # Avoid degenerate scale when all values are equal.
        eps = 1e-6 if vals_min == 0 else abs(vals_min) * 1e-3
        return (vals_min - eps, vals_max + eps)

    # Add small padding for readability.
    pad = (vals_max - vals_min) * 0.05
    return (vals_min - pad, vals_max + pad)


def _add_plot_rot_param_comparison(ax, plot_col, df, day_list, plot_labels, font_size, ylim=None):
    # Draw per-day scatter for one column; annotate each point with its index.
    df_selected = df[df["day"].isin(day_list)][["day", plot_col]].dropna(subset=[plot_col])

    labels = plot_labels if plot_labels is not None else day_list
    if len(labels) != len(day_list):
        raise ValueError("plot_labels と day_list の長さが一致していません。")

    for d_i, day in enumerate(day_list):
        plot_df = df_selected[df_selected["day"] == day]
        if len(plot_df) == 0:
            continue
        x = np.full(len(plot_df), d_i)
        y = plot_df[plot_col].to_numpy()
        ax.scatter(x, y, label=labels[d_i])
        # Put data index near each point for traceability in downstream checks.
        for idx, y_val in enumerate(y):
            ax.text(d_i + 0.05, y_val, str(idx + 1), fontsize=font_size * 0.7, va="center", ha="left")

    xticks = np.arange(len(day_list))
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels, fontsize=font_size, rotation=0)
    ax.set_xlim(-0.5, len(day_list) - 0.5)
    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_title(plot_col, fontsize=font_size)
    ax.set_ylabel("Value", fontsize=font_size)
    ax.tick_params(axis="both", which="major", labelsize=font_size)

    if ylim is not None:
        ax.set_ylim(*ylim)


def plot_rot_param_comparison(
    day_list,
    df,
    save_label,
    plot_labels=None,
    cols=3,
    rows=4,
    fig_size_x=20,
    fig_size_y=23,
    font_size=20,
):
    # Orchestrates paging, shared scales per base feature, and final image saving.
    save_dir = f"{param.save_dir_bef}/{save_label}"
    os.makedirs(save_dir, exist_ok=True)
    width_time_list = param.SD_window_width_list

    # Decide how many panels each base feature needs (width-dependent ones expand).
    cols_name = [c for c in ROTATION_FEATURES.__annotations__.keys() if c not in IGNORE_PLOT_COLS]
    plot_col_num_dict = {c: (len(width_time_list) if c in SD_WIDTH_DEPEND_COLS else 1) for c in cols_name}

    capacity = cols * rows
    page_idx = 1
    panel_idx_on_page = 0
    fig, axs = plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))

    def _hide_unfilled_axes(axs, filled, capacity):
        # Keep output clean: hide unused slots on a page.
        ax_arr = axs.ravel() if hasattr(axs, "ravel") else np.array([axs])
        for i in range(filled, capacity):
            ax_arr[i].set_visible(False)

    def _save_and_reset(fig, axs, page_idx, filled):
        # Finalize current page (hide empty slots, layout, save), then start a new one.
        _hide_unfilled_axes(axs, filled, capacity)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/rot_param_comparison_{page_idx}.png")
        plt.close(fig)
        return plt.subplots(rows, cols, figsize=(fig_size_x, fig_size_y))

    for base_col, num_panels in plot_col_num_dict.items():
        # Page break if remaining capacity is insufficient for this group.
        if panel_idx_on_page + num_panels > capacity:
            fig, axs = _save_and_reset(fig, axs, page_idx, panel_idx_on_page)
            page_idx += 1
            panel_idx_on_page = 0

        if base_col in SD_WIDTH_DEPEND_COLS:
            # Align y across all widths for the same base feature.
            derived_cols = [f"{base_col}_{w}s" for w in width_time_list]
            if param.flag_share_y_axis_across_width_time:
                ylim = _compute_ylim_for_columns(df, day_list, derived_cols)
            else:
                ylim = None

            for derived_col in derived_cols:
                ax = _get_axis(axs, panel_idx_on_page)
                _add_plot_rot_param_comparison(ax, derived_col, df, day_list, plot_labels, font_size, ylim=ylim)
                panel_idx_on_page += 1
        else:
            ax = _get_axis(axs, panel_idx_on_page)
            _add_plot_rot_param_comparison(ax, base_col, df, day_list, plot_labels, font_size, ylim=None)
            panel_idx_on_page += 1

    # last page
    if panel_idx_on_page > 0:
        fig, axs = _save_and_reset(fig, axs, page_idx, panel_idx_on_page)


def main(day_list, plot_labels=None):
    # Build combined dataframe across days and dispatch plotting.
    save_label = "-".join(day_list)

    rot_df_list = []
    for day in day_list:
        add_rot_df = rot_df_manage.get_rot_df(day)
        add_rot_df["day"] = day
        rot_df_list.append(add_rot_df)
    rot_df_all = pd.concat(rot_df_list, ignore_index=True)

    plot_rot_param_comparison(day_list, rot_df_all, save_label, plot_labels=plot_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", type=str, help="data labels")
    parser.add_argument("--plot-labels", nargs="+", type=str, help="data labels")

    args = parser.parse_args()
    day_list = args.days
    plot_labels = args.plot_labels

    main(day_list, plot_labels)
