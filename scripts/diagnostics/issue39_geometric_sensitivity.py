"""Check geometric ellipse sensitivity to optimizer configuration."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diagnostics.issue39_validation import (  # noqa: E402
    OUT,
    ROOT,
    TRUE_RADIUS,
    synthetic_points,
)
from utils.functions.rotation_center_candidates import (  # noqa: E402
    estimate_center,
    robust_radius,
)


def cases() -> list[tuple[str, int | None, np.ndarray, np.ndarray | None]]:
    samples: list[tuple[str, int | None, np.ndarray, np.ndarray | None]] = []
    for scenario in ("ellipse", "thick_ellipse", "short_arc", "stopped_cloud"):
        for seed in range(4200, 4204):
            points, centers = synthetic_points(scenario, seed)
            samples.append((scenario, seed, points, centers.mean(axis=0)))
    source = pd.read_csv(ROOT / "outputs/repellent-response/23/raw_centroid/2026_0802_182430.csv")
    if not np.array_equal(source.source_frame_1based, np.arange(1, 18433)):
        raise ValueError("No.14 source frame alignment changed")
    for start in (20.789, 64.862, 82.500):
        selected = source.loc[(source.time_sec >= start) & (source.time_sec < start + 5.080367)]
        samples.append((f"No.14 {start:.3f}s", None, selected[["x_um", "y_um"]].to_numpy(float), None))
    return samples


def sensitivity() -> pd.DataFrame:
    rows = []
    for scenario, seed, points, truth in cases():
        radius = robust_radius(points)
        for initial in ("ellipse", "median"):
            for bounds_scale in (0.75, 1.0, 2.0):
                for max_nfev in (40, 80):
                    fit = estimate_center(
                        points,
                        "geometric_ellipse",
                        initial_center=initial,
                        bounds_scale=bounds_scale,
                        max_nfev=max_nfev,
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "initial_center": initial,
                            "bounds_scale": bounds_scale,
                            "max_nfev": max_nfev,
                            "status": fit.status,
                            "center_x": np.nan if fit.center is None else fit.center[0],
                            "center_y": np.nan if fit.center is None else fit.center[1],
                            "center_error_R": (
                                np.nan
                                if fit.center is None or truth is None
                                else np.linalg.norm(fit.center - truth) / TRUE_RADIUS
                            ),
                            "radius": radius,
                        }
                    )
    result = pd.DataFrame(rows)
    reference = result.loc[
        (result.initial_center == "ellipse") & (result.bounds_scale == 1.0) & (result.max_nfev == 80),
        ["scenario", "seed", "center_x", "center_y"],
    ].rename(columns={"center_x": "reference_x", "center_y": "reference_y"})
    result = result.merge(reference, on=["scenario", "seed"], validate="many_to_one")
    result["shift_from_default_R"] = (
        np.hypot(result.center_x - result.reference_x, result.center_y - result.reference_y) / result.radius
    )
    return result.drop(columns=["reference_x", "reference_y"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sensitivity().to_csv(OUT / "geometric_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
