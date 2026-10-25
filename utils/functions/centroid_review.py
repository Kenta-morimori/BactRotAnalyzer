"""Native-TIFF overlay movies for qualitative inspection, not new measurements."""

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

COLORS = {"foreground": (0, 220, 255), "selected": (255, 220, 0), "centroid": (0, 0, 255)}


def display_limits(images, percentile=99.99):
    """Use one display-only contrast scale throughout a source window."""
    low, high = np.percentile(images, [0.5, percentile])
    return float(low), float(max(high, low + 1))


def gray_display(image, limits):
    low, high = limits
    return np.clip((image.astype(float) - low) * 255 / (high - low), 0, 255).astype(np.uint8)


def display_point(point, zoom):
    """Place a native pixel-center coordinate on a nearest-neighbor display."""
    return tuple(np.rint((np.asarray(point, dtype=float) + 0.5) * zoom - 0.5).astype(int))


def validate_saved(saved, rows, shape):
    """Never silently pair a saved result with different TIFF rows or times."""
    expected_frames = np.array([row["tiff_suffix_0based"] + 1 for row in rows])
    expected_times = np.array([np.nan if row["time_sec"] is None else row["time_sec"] for row in rows])
    if not np.array_equal(saved["source_frames"], expected_frames):
        raise ValueError("Saved source frames do not match the TIFF timeline")
    if not np.allclose(saved["times"], expected_times, atol=1e-9, rtol=0, equal_nan=True):
        raise ValueError("Saved timestamps do not match the TIFF timeline")
    if not np.array_equal(saved["segments"], [row["segment_id"] for row in rows]):
        raise ValueError("Saved recording segments do not match the TIFF timeline")
    n = len(rows)
    if saved["points"].shape != (n, 2) or saved["selected_masks"].shape != (n, *shape):
        raise ValueError("Saved coordinates/masks do not share the original image axes")
    if len(saved["reasons"]) != n:
        raise ValueError("Missing per-frame decision reasons")
    if not np.isfinite(saved["selected_masks"]).all():
        raise ValueError("Invalid mask coverage")


def centroid_span(points):
    """Sensitivity across methods; explicitly not a true position-error bound."""
    valid = np.asarray(points, dtype=float)
    valid = valid[np.isfinite(valid).all(axis=1)]
    if len(valid) < 2:
        return None
    return float(np.linalg.norm(valid[:, None] - valid[None, :], axis=-1).max())


def text(canvas, value, xy, color=(235, 235, 235), scale=0.46):
    cv2.putText(canvas, value, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def overlay(image, all_coverage, selected, point, limits, zoom=5):
    """Preserve raw texture, draw both masks, and omit missing centroids."""
    gray = cv2.resize(gray_display(image, limits), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for coverage, color, alpha in ((all_coverage, COLORS["foreground"], 0.10), (selected, COLORS["selected"], 0.12)):
        coverage = cv2.resize(coverage.astype(np.float32), gray.shape[::-1], interpolation=cv2.INTER_NEAREST)
        weight = np.clip(coverage, 0, 1)[..., None] * alpha
        canvas = np.rint(canvas * (1 - weight) + np.array(color) * weight).astype(np.uint8)
        outlines, _ = cv2.findContours((coverage > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, outlines, -1, color, 1)
    if np.isfinite(point).all():
        cv2.drawMarker(canvas, display_point(point, zoom), COLORS["centroid"], cv2.MARKER_CROSS, 11, 1)
    return canvas


def review_frame(image, frame_results, row, methods, limits, context_limits, replay=1, repeats=3, zoom=5):
    """Two selector rows, raw TIFF plus all five segmentation candidates."""
    side = image.shape[1] * zoom
    height = image.shape[0] * zoom
    cell_header, header, footer = 70, 90, 58
    canvas = np.full((header + 2 * (cell_header + height) + footer, side * (len(methods) + 1), 3), 22, np.uint8)
    stamp = row.get("time_sec")
    time_label = f"{stamp:.6f} s" if stamp is not None else "unavailable"
    text(
        canvas,
        f"Sample {row.get('sample_no', '?')} | Original TIFF suffix {row['tiff_suffix_0based']} (0-based) | TIFF time {time_label} | segment {row['segment_id']}",
        (12, 22),
        scale=0.57,
    )
    text(
        canvas,
        f"Review only | replay {replay}/{repeats} | original TIFF coordinates | uniform 10 fps playback, nominal 20x slow",
        (12, 45),
    )
    text(
        canvas, "Yellow: all foreground  Cyan: selected region  Red +: saved centroid  Missing stays missing", (12, 68)
    )
    selectors = ("largest_external_contour", "constrained")
    for r, selector in enumerate(selectors):
        y = header + r * (cell_header + height)
        raw_limits = limits if r == 0 else context_limits
        title = "Original TIFF (fixed contrast)" if r == 0 else "Original TIFF (context contrast)"
        text(canvas, title, (8, y + 20))
        text(
            canvas,
            "No centroid/target annotation" if r == 0 else "Bright pixels may clip for context",
            (8, y + 43),
            scale=0.42,
        )
        raw = cv2.resize(gray_display(image, raw_limits), (side, height), interpolation=cv2.INTER_NEAREST)
        canvas[y + cell_header : y + cell_header + height, :side] = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        for column, method in enumerate(methods, 1):
            x = column * side
            result = frame_results[(method["name"], selector)]
            point = result["point"]
            text(canvas, method["name"], (x + 6, y + 17), scale=0.40)
            text(canvas, "Largest external contour" if r == 0 else "Constrained tracker", (x + 6, y + 35), scale=0.42)
            reason = str(result["reason"])
            value = f"x={point[0]:.3f} y={point[1]:.3f}" if np.isfinite(point).all() else "MISSING: " + reason
            text(canvas, value, (x + 6, y + 55), (150, 240, 150) if np.isfinite(point).all() else (110, 180, 255), 0.36)
            canvas[y + cell_header : y + cell_header + height, x : x + side] = overlay(
                image, result["all_coverage"], result["selected"], point, limits, zoom
            )
    span = centroid_span([frame_results[(m["name"], "largest_external_contour")]["point"] for m in methods])
    sensitivity = "unavailable" if span is None else f"{span:.3f} native px"
    text(
        canvas,
        f"Largest-selection centroid span across methods: {sensitivity} (sensitivity only, NOT true error)",
        (10, canvas.shape[0] - 34),
    )
    text(
        canvas,
        "No AVI overlay: TIFF-to-AVI mapping remains unverified. Videos are for visual review, not analysis input.",
        (10, canvas.shape[0] - 12),
    )
    return canvas


def card(shape, message):
    canvas = np.full(shape, 18, dtype=np.uint8)
    for i, line in enumerate(message):
        text(canvas, line, (35, 110 + i * 46), (120, 200, 255), 0.85)
    return canvas


def encode_video(path, frames, fps=10):
    """Stream H.264 for native app playback; refuse to overwrite an artifact."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the local review videos")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    iterator = iter(frames)
    first = next(iterator)
    height, width = first.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("H.264 review dimensions must be even")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    with path.with_suffix(".ffmpeg.log").open("wb") as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        assert process.stdin is not None
        try:
            process.stdin.write(first.tobytes())
            for frame in iterator:
                if frame.shape != first.shape or frame.dtype != np.uint8:
                    raise ValueError("Inconsistent review frame shape/dtype")
                process.stdin.write(frame.tobytes())
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError(f"Video encoding failed: {path.with_suffix('.ffmpeg.log')}")
        except BaseException:
            process.stdin.close()
            process.terminate()
            process.wait()
            raise


def write_gallery(output, entries):
    """A local-only media index; it never sends measured images to a service."""
    import html

    sections = []
    for entry in entries:
        name = entry["name"]
        sections.append(
            f'<section><h2>{html.escape(name)}</h2><p>{html.escape(entry["description"])}</p>'
            f'<video controls preload="none" poster="{name}.png" src="{name}.mp4"></video>'
            f'<p><a href="{name}.json">Frame mapping and sensitivity</a></p></section>'
        )
    content = '<!doctype html><meta charset="utf-8"><title>Local TIFF centroid review</title>'
    content += "<style>body{font:16px system-ui;background:#181818;color:#eee;margin:24px}video{width:100%;max-width:1920px}section{margin:36px 0}a{color:#8df}</style>"
    content += "<h1>元TIFF・候補マスク・重心の比較</h1><p>ローカル実測データ。黄色：全領域、シアン：選択領域、赤：重心。上段：最大外輪郭、下段：追跡。誤差の真値・採用判定ではありません。</p>"
    Path(output, "index.html").write_text(content + "".join(sections), encoding="utf-8")
    Path(output, "index.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
