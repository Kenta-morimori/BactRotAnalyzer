"""Private, persistent batch TIFF proposals, review movies and explicit approval."""

import copy
import html
import json
import threading
import uuid
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd

from utils.functions import tiff_roi_io as images
from utils.functions.centroid_review import (
    display_limits,
    encode_video,
    gray_display,
    text,
)
from utils.functions.repellent_input import (
    _hash,
    _load,
    _observations,
    _resolve,
    _write,
    initial_threshold,
    selected_samples,
)
from utils.functions.tiff_roi import (
    ROI,
    Tracker,
    binary_regions,
    infer_roi,
    nominal_max_dt,
)
from utils.run_context import REPOSITORY, analysis_run, require_clean_git

METHOD = "central-target-fixed-roi-batch-review-v1"
FIELDS = ("input_id", "tiff_directory", "px2um_x", "px2um_y", "analysis_frame_range", "threshold", "roi", "seeds")


class ReviewCancelled(Exception):
    pass


def conditions(sample):
    return {k: sample.get(k) for k in FIELDS}


def implementation():
    folder = Path(__file__).parent
    return dict(
        method=METHOD,
        code={
            name: _hash(folder / name)
            for name in ("repellent_review.py", "repellent_input.py", "tiff_roi.py", "tiff_roi_io.py")
        },
        runtime={name: version(name) for name in ("numpy", "opencv-python", "pandas", "Pillow", "tifffile")},
    )


def required_starts(rows):
    maximum = nominal_max_dt(rows)
    return [
        r
        for i, r in enumerate(rows)
        if r["time_sec"] is not None and (i == 0 or images.recording_boundary(rows[i - 1], r, maximum))
    ]


def central_seed(image, threshold):
    """Choose a foreground point in the uniquely closest region, never a hole centroid."""
    height, width = image.shape
    _, _, objects = binary_regions(image, ROI(0, 0, width, height), threshold)
    if not objects:
        return None, "no_central_candidate"
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    distances = np.array([np.linalg.norm(o.point - center) for o in objects])
    winners = np.flatnonzero(np.isclose(distances, distances.min(), rtol=0, atol=1e-12))
    if len(winners) != 1:
        return None, "ambiguous_central_candidate"
    chosen = objects[int(winners[0])]
    if chosen.edge:
        return None, "central_candidate_on_image_edge"
    y, x = np.nonzero(chosen.mask)
    points = (np.column_stack([x, y]) + 0.5) / 10 - 0.5
    point = points[np.argmin(np.linalg.norm(points - chosen.point, axis=1))]
    return point.tolist(), "central_candidate"


def propose(sample, rows):
    candidate = copy.deepcopy(sample)
    candidate["threshold"] = initial_threshold(sample)
    candidate["seeds"] = dict(sample.get("seeds") or {})
    images.validate_sample(dict(candidate, frames=rows))
    issues, origins = [], {}
    starts = required_starts(rows)
    if not starts:
        issues.append("no_valid_timestamp")
    for row in starts:
        key = str(row["source_frame_1based"])
        if key in candidate["seeds"]:
            origins[key] = "saved"
            continue
        seed, reason = central_seed(images.load_image(row["path"]), candidate["threshold"])
        origins[key] = reason
        if seed is None:
            issues.append(f"frame {key}: {reason}")
        else:
            candidate["seeds"][key] = seed
    calibration = dict(reason="saved_roi")
    if candidate.get("roi") is None:
        if starts:
            first = starts[0]
            start = rows.index(first)
            seed = candidate["seeds"].get(str(first["source_frame_1based"]))
            if seed is not None:
                subset = rows[start : start + 200]
                roi, calibration = infer_roi(
                    [images.load_image(r["path"]) for r in subset], subset, seed, candidate["threshold"]
                )
                if roi is not None:
                    candidate["roi"] = [roi.x, roi.y, roi.width, roi.height]
        if candidate.get("roi") is None:
            issues.append("automatic_roi_unavailable")
    if candidate.get("roi") is not None:
        roi = ROI(*candidate["roi"])
        for frame, seed in candidate["seeds"].items():
            row = next(r for r in rows if str(r["source_frame_1based"]) == frame)
            if row["time_sec"] is None:
                issues.append(f"frame {frame}: invalid_seed_time")
                continue
            result = Tracker(roi, candidate["threshold"]).step(
                images.load_image(row["path"]), row["source_frame_1based"], row["time_sec"], row["segment_id"], seed
            )
            if result.point is None:
                issues.append(f"frame {frame}: {result.reason}")
    return candidate, issues, dict(seed_origins=origins, roi=calibration)


def movie_frame(plane, row, result, sample, limits, issues):
    """Two aligned full-field panels; the right mask keeps its actual 10x resolution."""
    height, width = plane.shape
    raw = cv2.cvtColor(
        cv2.resize(gray_display(plane, limits), None, fx=10, fy=10, interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR
    )
    expanded = cv2.resize(plane, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
    mask = cv2.cvtColor((expanded > sample["threshold"]).astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    roi = ROI(*(sample.get("roi") or [0, 0, width, height]))
    panel = mask[roi.y * 10 : (roi.y + roi.height) * 10, roi.x * 10 : (roi.x + roi.width) * 10]
    panel[result.foreground] = 255
    panel[result.mask] = [210, 210, 0]
    for image in (raw, mask):
        if sample.get("roi") is not None:
            cv2.rectangle(
                image,
                (roi.x * 10, roi.y * 10),
                ((roi.x + roi.width) * 10 - 1, (roi.y + roi.height) * 10 - 1),
                (40, 220, 40),
                2,
            )
        seed = sample["seeds"].get(str(row["source_frame_1based"]))
        if seed is not None:
            cv2.drawMarker(
                image,
                tuple(np.rint((np.asarray(seed) + 0.5) * 10 - 0.5).astype(int)),
                (220, 120, 220),
                cv2.MARKER_CROSS,
                16,
                2,
            )
        if result.point is not None:
            cv2.drawMarker(
                image, tuple(np.rint((result.point + 0.5) * 10 - 0.5).astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2
            )
    canvas = np.full((height * 10 + 100, width * 20, 3), 18, np.uint8)
    canvas[100:, : width * 10], canvas[100:, width * 10 :] = raw, mask
    stamp = "missing" if row["time_sec"] is None else f"{row['time_sec']:.6f}s"
    text(
        canvas,
        f"{sample['input_id']} | source {row['source_frame_1based']} | {stamp} | segment {row['segment_id']}",
        (8, 22),
        scale=0.55,
    )
    text(canvas, "Original TIFF + ROI / target / centroid", (8, 48), scale=0.5)
    text(canvas, "10x linear mask + same ROI / adopted region", (width * 10 + 8, 48), scale=0.5)
    reason = "; ".join(issues[:2]) if issues else result.reason
    text(canvas, f"{reason} | green: ROI; cyan: adopted; red: measured centroid; purple: seed", (8, 76), scale=0.5)
    if sample.get("roi") is None:
        text(canvas, "ROI UNAVAILABLE: full-field preview only; not an inferred ROI", (8, 96), scale=0.45)
    return canvas


def verify_artifact(path, sample=None, *, verify_images=True, settings_path=None):
    """Fail closed if a reviewed condition, source, implementation or artifact changed."""
    path = Path(path).resolve()
    saved = json.loads(path.read_text())
    if saved.get("status") != "complete" or saved.get("implementation") != implementation():
        raise ValueError("確認結果が未完了または実装が変わりました。再生成してください")
    if sample is not None and conditions(sample) != saved["conditions"]:
        raise ValueError("対象・閾値・ROIなどが確認動画から変わりました。再生成・承認してください")
    if (
        sample is not None
        and settings_path is not None
        and str(_resolve(settings_path, sample["tiff_directory"])) != saved["source_directory"]
    ):
        raise ValueError("元TIFFの保存先が変わりました。再生成してください")
    if any(_hash(path.parent / name) != digest for name, digest in saved["artifact_hashes"].items()):
        raise ValueError("確認動画または仮抽出結果が変わりました。再生成してください")
    if verify_images:
        paths = {str(p) for p in Path(saved["source_directory"]).glob("*.tif")}
        if paths != set(saved["input_hashes"]) or any(
            _hash(name) != digest for name, digest in saved["input_hashes"].items()
        ):
            raise ValueError("元TIFFが確認動画から変わりました。再生成・承認してください")
    return saved


def generate(sample, settings_path, folder, cancel, progress):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    rows = images.directory_rows(_resolve(settings_path, sample["tiff_directory"]))
    calibration = np.asarray([sample["px2um_x"], sample["px2um_y"]], dtype=float)
    if not np.isfinite(calibration).all() or (calibration <= 0).any():
        raise ValueError("positive source-pixel calibration required")
    initial_code = implementation()
    hashes = {}
    for row in rows:
        if cancel.is_set():
            raise ReviewCancelled()
        hashes[row["path"]] = _hash(row["path"])
    candidate, issues, origins = propose(sample, rows)
    first = images.load_image(rows[0]["path"])
    limits = display_limits(np.stack([images.load_image(r["path"]) for r in rows[:200]]))
    runtime = copy.deepcopy(candidate)
    if candidate.get("roi") is None:
        runtime.update(roi=[0, 0, first.shape[1], first.shape[0]], seeds={})
    records, mapping = [], []
    flags: list[dict[str, Any]] = []
    maximum = nominal_max_dt(rows)

    def frames():
        previous = None
        for index, (record, plane, result) in enumerate(_observations(runtime, rows, "tiff", settings_path)):
            if cancel.is_set():
                raise ReviewCancelled()
            row = rows[index]
            records.append(record)
            mapping.append(
                dict(
                    video_frame=index,
                    source_frame_1based=row["source_frame_1based"],
                    time_sec=row["time_sec"],
                    segment_id=row["segment_id"],
                )
            )
            gap = previous is not None and images.recording_boundary(previous, row, maximum)
            if result.reason != "accepted" or gap:
                reason = "recording_boundary" if gap else result.reason
                if flags and flags[-1]["reason"] == reason and flags[-1]["end_index"] == index - 1 and not gap:
                    flags[-1].update(end_index=index, source_end_frame=row["source_frame_1based"])
                else:
                    flags.append(
                        dict(
                            index=index,
                            end_index=index,
                            source_frame=row["source_frame_1based"],
                            source_end_frame=row["source_frame_1based"],
                            reason=reason,
                        )
                    )
            if index % 50 == 0:
                progress(index, len(rows))
            previous = row
            yield movie_frame(plane, row, result, candidate, limits, issues)

    encode_video(folder / "review.mp4", frames(), fps=50)
    pd.DataFrame(records).to_csv(folder / "centroids.csv", index=False, float_format="%.17g")
    _write(folder / "presentation.json", dict(fps=50, frames=mapping, flags=flags))
    if cancel.is_set():
        raise ReviewCancelled()
    if initial_code != implementation() or any(_hash(name) != digest for name, digest in hashes.items()):
        raise ValueError("入力または実装が処理中に変わりました。再生成してください")
    saved = dict(
        status="complete",
        source_directory=str(_resolve(settings_path, sample["tiff_directory"])),
        sample_no=sample["sample_no"],
        input_id=sample["input_id"],
        conditions=conditions(candidate),
        base_conditions=sample.get("_review_base", conditions(sample)),
        eligible=not issues,
        issues=issues,
        origins=origins,
        display_limits=[float(v) for v in limits],
        input_hashes=hashes,
        implementation=initial_code,
        artifact_hashes={name: _hash(folder / name) for name in ("centroids.csv", "presentation.json", "review.mp4")},
        frame_count=len(rows),
        flags_count=len(flags),
    )
    _write(folder / "candidate.json", saved)
    progress(len(rows), len(rows))
    return saved


class ReviewManager:
    """One serial worker; completed immutable movie revisions survive editor restarts."""

    def __init__(self, settings_path, numbers=None, root=None, record_runs=False):
        self.path = Path(settings_path).resolve()
        self.numbers = numbers
        self.record_runs = record_runs
        settings = _load(self.path)
        day = Path(settings["day"])
        if day.is_absolute() or ".." in day.parts:
            raise ValueError("invalid review day")
        self.root = (Path(root) if root is not None else Path.cwd() / "outputs/preprocessing_review" / day).resolve()
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.state: dict[str, Any] = dict(status="idle", entries={})
        latest = self.root / "latest.json"
        if latest.is_file():
            run = self.root / json.loads(latest.read_text())["review_id"]
            if run.parent != self.root:
                raise ValueError("invalid local review index")
            self.state = json.loads((run / "review.json").read_text())
            if self.state["status"] in {"running", "queued", "finalizing"}:
                self.state["status"] = "interrupted"
                for entry in self.state["entries"].values():
                    if entry["status"] in {"running", "queued"}:
                        entry["status"] = "interrupted"
                self._persist()

    def _persist(self):
        folder = self.root / self.state["review_id"]
        _write(folder / "review.json", self.state)
        self._index(folder)

    def _index(self, folder):
        parts = [
            '<!doctype html><meta charset="utf-8"><title>TIFF確認動画</title><h1>保存済み確認動画</h1><p>仮処理の記録です。承認操作は設定画面で行ってください。</p>'
        ]
        for number, entry in self.state["entries"].items():
            parts.append(f'<h2>No.{html.escape(number)} — {html.escape(entry["status"])}</h2>')
            if entry.get("artifact"):
                relative = Path(entry["artifact"]).parent.relative_to(self.root).as_posix()
                url = "../" + relative + "/review.mp4"
                parts.append(
                    f'<video controls preload="metadata" style="max-width:100%" src="{html.escape(url, quote=True)}"></video>'
                )
        (folder / "index.html").write_text("\n".join(parts), encoding="utf-8")

    def status(self):
        with self.lock:
            return copy.deepcopy(self.state)

    def start(self, numbers=None, overrides=None):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("前処理中です。完了または中断を待ってください")
            settings = _load(self.path)
            if self.record_runs:
                require_clean_git(REPOSITORY)
            samples = copy.deepcopy(selected_samples(settings, self.numbers if numbers is None else numbers))
            allowed = {s["sample_no"] for s in selected_samples(settings, self.numbers)}
            if not {s["sample_no"] for s in samples} <= allowed:
                raise ValueError("unknown or unselected sample number")
            overrides = overrides or {}
            previous = copy.deepcopy(self.state.get("entries", {}))
            for sample in samples:
                base = conditions(sample)
                old = previous.get(str(sample["sample_no"]), {})
                if old.get("base_conditions") == base and old.get("request"):
                    sample.update(old["request"])
                change = overrides.get(str(sample["sample_no"]), {})
                if set(change) - {"threshold", "roi", "seeds"}:
                    raise ValueError("only target, threshold and ROI may be edited here")
                sample.update(change)
                sample["_review_base"] = base
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            self.state = dict(
                review_id=run_id, status="running", settings_revision=_hash(self.path), entries=copy.deepcopy(previous)
            )
            for s in samples:
                self.state["entries"][str(s["sample_no"])] = dict(
                    status="queued", input_id=s["input_id"], request=conditions(s), base_conditions=s["_review_base"]
                )
            self.cancel.clear()
            _write(self.root / run_id / "settings_snapshot.json", settings)
            self._persist()
            _write(self.root / "latest.json", dict(review_id=run_id))
            self.thread = threading.Thread(target=self._work, args=(samples, previous, overrides), daemon=True)
            self.thread.start()
            return self.status()

    def _work(self, samples, previous, overrides):
        try:
            if self.record_runs:
                with analysis_run(
                    parameters=dict(operation=METHOD, samples=[s["sample_no"] for s in samples]),
                    inputs=[self.path] + [_resolve(self.path, s["tiff_directory"]) for s in samples],
                    outputs=[self.root / self.state["review_id"]],
                ):
                    self._process(samples, previous, overrides)
                    if self.cancel.is_set():
                        raise ReviewCancelled("preprocessing interrupted")
            else:
                self._process(samples, previous, overrides)
            with self.lock:
                self.state["status"] = "interrupted" if self.cancel.is_set() else "complete"
                self._persist()
        except ReviewCancelled:
            pass
        except Exception as error:
            with self.lock:
                self.state["status"] = "failed"
                self.state["error"] = str(error)
                for entry in self.state["entries"].values():
                    if entry["status"] in {"running", "queued"}:
                        entry.update(status="failed", issues=[str(error)])
                self._persist()

    def _process(self, samples, previous, overrides):
        for sample in samples:
            number = str(sample["sample_no"])
            if self.cancel.is_set():
                break
            with self.lock:
                self.state["entries"][number]["status"] = "running"
                self._persist()
            try:
                old = previous.get(number, {})
                saved = None
                artifact = old.get("artifact")
                if number not in overrides and artifact:
                    try:
                        checked = verify_artifact(artifact)
                        if conditions(sample) in (checked["base_conditions"], checked["conditions"]) and checked[
                            "source_directory"
                        ] == str(_resolve(self.path, sample["tiff_directory"])):
                            saved = checked
                    except (ValueError, OSError, KeyError):
                        pass
                if saved is None:
                    folder = self.root / self.state["review_id"] / f"no{int(number):02}"
                    artifact = str(folder / "candidate.json")
                    saved = generate(sample, self.path, folder, self.cancel, lambda i, n: self._progress(number, i, n))
                with self.lock:
                    self.state["entries"][number] = dict(
                        status="complete",
                        artifact=artifact,
                        eligible=saved["eligible"],
                        issues=saved["issues"],
                        frame_count=saved["frame_count"],
                        flags_count=saved["flags_count"],
                        input_id=saved["input_id"],
                    )
                    self._persist()
            except ReviewCancelled:
                break
            except Exception as error:
                with self.lock:
                    self.state["entries"][number].update(status="failed", issues=[str(error)])
                    self._persist()
        with self.lock:
            if self.cancel.is_set():
                for entry in self.state["entries"].values():
                    if entry["status"] in {"running", "queued"}:
                        entry["status"] = "interrupted"
            self.state["status"] = "interrupted" if self.cancel.is_set() else "finalizing"
            self._persist()

    def _progress(self, number, index, count):
        with self.lock:
            self.state["entries"][number].update(done=index, total=count)

    def close(self):
        self.cancel.set()
        if self.thread is not None:
            self.thread.join()

    def interrupt(self):
        self.cancel.set()
        return dict(status="cancelling")

    def artifact(self, number):
        with self.lock:
            entry = self.state["entries"].get(str(number), {})
            if entry.get("status") != "complete" or not entry.get("artifact"):
                raise ValueError("このNo.の確認動画は未完了です")
            path = Path(entry["artifact"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("invalid review artifact path")
            return path

    def details(self, number):
        path = self.artifact(number)
        saved = json.loads(path.read_text())
        presentation = json.loads((path.parent / "presentation.json").read_text())
        return dict(candidate=saved, presentation=presentation)

    def approve(self, numbers, revision, order_confirmed=False):
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                raise ValueError("前処理の完了または中断を待ってください")
            if self.state["status"] not in {"complete", "interrupted"}:
                raise ValueError("前処理が未完了です。再生成してください")
            if revision != _hash(self.path) or revision != self.state.get("settings_revision"):
                raise ValueError("設定が別操作で変更されました。再読込して再生成してください")
            settings = _load(self.path)
            selected_samples(settings, numbers)
            if not settings["order_confirmed"] and not order_confirmed:
                raise ValueError("撮影順を確認してください")
            for sample in selected_samples(settings, numbers):
                artifact = self.artifact(sample["sample_no"])
                saved = verify_artifact(artifact)
                if saved["source_directory"] != str(_resolve(self.path, sample["tiff_directory"])):
                    raise ValueError("元TIFFの保存先が変わりました。再生成してください")
                if not saved["eligible"]:
                    raise ValueError("対象またはROIが未確定のNo.は承認できません")
                if conditions(sample) not in (saved["base_conditions"], saved["conditions"]):
                    raise ValueError("保存条件が候補から変わりました。再生成してください")
                sample.update(saved["conditions"], target_confirmed=True, review_artifact=str(artifact))
            if revision != _hash(self.path):
                raise ValueError("承認中に設定が変更されました。再読込してください")
            settings["order_confirmed"] = True
            settings["review"] = dict(
                method=METHOD,
                review_id=self.state["review_id"],
                accepted_samples=numbers,
                held_samples=[s["sample_no"] for s in settings["samples"] if s["sample_no"] not in numbers],
            )
            _write(self.path, settings)
            _write(self.root / self.state["review_id"] / ("approval_" + uuid.uuid4().hex[:8] + ".json"), settings)
            self.state["settings_revision"] = _hash(self.path)
            self._persist()
            return settings
