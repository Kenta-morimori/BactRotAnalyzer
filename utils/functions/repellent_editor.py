"""Loopback-only HTML editor for explicit per-recording TIFF target choices."""

import base64
import io as buffers
import json
import secrets
import shlex
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from utils.functions import tiff_roi_io as images
from utils.functions.centroid_review import display_limits, gray_display
from utils.functions.repellent_input import (
    TIFF_INITIAL_THRESHOLD,
    _hash,
    _load,
    _resolve,
    _write,
    initial_threshold,
    invalidate_review,
    missing_tiff_setup,
    selected_samples,
)
from utils.functions.repellent_review import ReviewManager
from utils.functions.tiff_roi import (
    ROI,
    Tracker,
    binary_regions,
    infer_roi,
    nominal_max_dt,
)


def _png(array):
    output = buffers.BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


class Editor:
    def __init__(self, path, numbers=None, confirm_order=False, review_root=None, record_runs=False):
        self.path = Path(path).resolve()
        self.numbers = numbers
        self.rows = {}
        self.lock = threading.RLock()
        self.confirm_order = confirm_order
        self.review = ReviewManager(self.path, numbers, root=review_root, record_runs=record_runs)

    def settings(self):
        settings = _load(self.path)
        review_state = self.review.status()
        return dict(
            day=settings["day"],
            source=settings["source"],
            order_confirmed=settings["order_confirmed"] or self.confirm_order,
            samples=selected_samples(settings, self.numbers),
            all_sample_numbers=[s["sample_no"] for s in settings["samples"]],
            revision=_hash(self.path),
            defaults=dict(threshold=TIFF_INITIAL_THRESHOLD),
            settings_path=str(self.path),
            readiness=saved_readiness(settings, review_state),
            review=review_state,
        )

    def recording(self, number):
        if type(number) is not int:
            raise ValueError("sample number must be an integer")
        sample = next((s for s in self.settings()["samples"] if s["sample_no"] == number), None)
        if sample is None:
            raise ValueError("unknown or unselected sample number")
        folder = _resolve(self.path, sample["tiff_directory"])
        key = (number, str(folder))
        if key not in self.rows:
            self.rows[key] = images.directory_rows(folder)
        rows = self.rows[key]
        maximum = nominal_max_dt(rows)
        required = [
            r["source_frame_1based"]
            for i, r in enumerate(rows)
            if r["time_sec"] is not None and (i == 0 or images.recording_boundary(rows[i - 1], r, maximum))
        ]
        return sample, rows, required

    def preview(self, request):
        sample, rows, required = self.recording(request["sample_no"])
        index = request.get("index", 0)
        if type(index) is not int or not 0 <= index < len(rows):
            raise ValueError("frame slider index is outside this recording")
        row = rows[index]
        plane = images.load_image(row["path"])
        threshold = float(initial_threshold(request) if "threshold" in request else initial_threshold(sample))
        roi_values = request.get("roi") or sample.get("roi") or [0, 0, plane.shape[1], plane.shape[0]]
        roi = ROI(*roi_values)
        roi.validate(plane.shape)
        mask, _, _ = binary_regions(plane, roi, threshold)
        canvas = np.zeros((plane.shape[0] * 10, plane.shape[1] * 10, 3), np.uint8)
        panel = canvas[roi.y * 10 : (roi.y + roi.height) * 10, roi.x * 10 : (roi.x + roi.width) * 10]
        panel[mask] = 255
        seeds = request.get("seeds", sample.get("seeds", {}))
        seed = seeds.get(str(row["source_frame_1based"]))
        reason = "このフレームに再指定点なし：追跡結果の確定表示ではありません"
        if seed is not None and row["time_sec"] is not None:
            found = Tracker(roi, threshold).step(
                plane, row["source_frame_1based"], row["time_sec"], row["segment_id"], seed
            )
            panel[found.mask] = [0, 210, 210]
            reason = found.reason
        return dict(
            image=_png(gray_display(plane, display_limits(plane))),
            mask=_png(canvas),
            width=plane.shape[1],
            height=plane.shape[0],
            count=len(rows),
            index=index,
            source_frame=row["source_frame_1based"],
            time_sec=row["time_sec"],
            timestamp=row["source_timestamp"],
            segment=row["segment_id"],
            required=required,
            threshold_max=(
                int(np.iinfo(plane.dtype).max) if np.issubdtype(plane.dtype, np.integer) else int(plane.max())
            ),
            reason=reason,
        )

    def auto_roi(self, request):
        sample, rows, required = self.recording(request["sample_no"])
        if not required:
            raise ValueError("有効な開始時刻がありません")
        start = next(i for i, row in enumerate(rows) if row["source_frame_1based"] == required[0])
        rows = rows[start : start + 200]
        seeds = request.get("seeds", sample.get("seeds", {}))
        seed = seeds.get(str(rows[0]["source_frame_1based"]))
        if seed is None:
            raise ValueError(f"元フレーム{required[0]}で対象点を指定してください")
        roi, diagnostics = infer_roi(
            [images.load_image(r["path"]) for r in rows[:200]],
            rows[:200],
            seed,
            float(request["threshold"]),
        )
        if roi is None:
            raise ValueError("自動ROIの支持不足です。原画像上でROIをドラッグしてください")
        return dict(roi=[roi.x, roi.y, roi.width, roi.height], diagnostics=diagnostics)

    def locate(self, request):
        _, rows, _ = self.recording(request["sample_no"])
        for index, row in enumerate(rows):
            if row["source_frame_1based"] == request["source_frame"]:
                return dict(index=index)
        raise ValueError("元フレームが存在しません")

    def refresh(self):
        self.rows.clear()
        return self.settings()

    def save(self, request):
        if request.get("revision") != _hash(self.path):
            raise ValueError("別の画面または操作で設定が変わりました。再読込してください")
        settings = _load(self.path)
        _, rows, required = self.recording(request["sample_no"])
        sample = next(s for s in settings["samples"] if s["sample_no"] == request["sample_no"])
        threshold = float(request["threshold"])
        roi = ROI(*request["roi"])
        roi.validate(images.load_image(rows[0]["path"]).shape)
        seeds = request["seeds"]
        images.validate_sample(dict(frames=rows, seeds=seeds, threshold=threshold))
        confirmed = request.get("confirmed") is True
        if confirmed and (not required or not seeds or any(str(frame) not in seeds for frame in required)):
            raise ValueError("記録開始・ギャップ後の未指定フレームがあります。下の一覧から対象を指定してください")
        by_frame = {r["source_frame_1based"]: r for r in rows}
        for frame, seed in seeds.items():
            row = by_frame[int(frame)]
            if row["time_sec"] is None:
                raise ValueError("時刻欠測の行を対象再指定に使用できません")
            found = Tracker(roi, threshold).step(
                images.load_image(row["path"]), int(frame), row["time_sec"], row["segment_id"], seed
            )
            if found.point is None:
                raise ValueError(f"元フレーム{frame}の対象点を確認してください: {found.reason}")
        order = settings["order_confirmed"] or request.get("order_confirmed") is True
        if confirmed and not order:
            raise ValueError("撮影順を確認してください")
        sample.update(threshold=threshold, roi=request["roi"], seeds=seeds, target_confirmed=confirmed)
        invalidate_review(settings, sample)
        settings["order_confirmed"] = order
        if request["revision"] != _hash(self.path):
            raise ValueError("保存直前に設定が変更されました。再読込してください")
        _write(self.path, settings)
        return self.settings()


def editor_server(path, numbers=None, confirm_order=False, review_root=None, record_runs=False):
    editor = Editor(path, numbers, confirm_order, review_root, record_runs)
    if editor.settings()["source"] != "tiff":
        raise ValueError("HTML対象設定はTIFF入力用です")
    token = secrets.token_urlsafe(32)
    closing = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value, kind="application/json"):
            data = (
                value.encode()
                if isinstance(value, str)
                else json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            )
            self.send_response(status)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self):
            return self.headers.get("Host") == f"127.0.0.1:{server.server_port}"

        def do_GET(self):
            if not self.allowed():
                self.reply(403, {"error": "local host required"})
                return
            parsed = urlparse(self.path)
            if parsed.path == "/review/media":
                query = parse_qs(parsed.query)
                if query.get("token") != [token]:
                    self.reply(403, {"error": "local editor token required"})
                    return
                try:
                    artifact = editor.review.artifact(int(query["sample_no"][0]))
                    self.media(artifact.parent / "review.mp4")
                except (ValueError, KeyError, OSError) as error:
                    self.reply(400, {"error": str(error)})
                return
            if parsed.path != "/":
                self.reply(404, {"error": "not found"})
                return
            page = Path(__file__).with_name("repellent_editor.html").read_text()
            self.reply(200, page.replace("EDITOR_TOKEN", token), "text/html")

        def media(self, path):
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                try:
                    unit, bounds = requested.split("=", 1)
                    left, right = bounds.split("-", 1)
                    if unit != "bytes" or "," in bounds:
                        raise ValueError()
                    start, end = (
                        (int(left), min(int(right), size - 1) if right else size - 1)
                        if left
                        else (max(0, size - int(right)), size - 1)
                    )
                    if not 0 <= start <= end < size:
                        raise ValueError()
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as stream:
                    stream.seek(start)
                    remaining = end - start + 1
                    while remaining:
                        block = stream.read(min(256 * 1024, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (
                not self.allowed()
                or self.headers.get("X-Editor-Token") != token
                or (origin is not None and origin != f"http://127.0.0.1:{server.server_port}")
            ):
                self.reply(403, {"error": "local editor token required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1_000_000:
                    raise ValueError("invalid request size")
                request = json.loads(self.rfile.read(length))
                if urlparse(self.path).path == "/exit":
                    with editor.lock:
                        if closing.is_set():
                            self.reply(409, {"error": "設定画面は終了中です"})
                            return
                        closing.set()
                        editor.review.cancel.set()
                        settings = editor.settings()
                        try:
                            self.reply(
                                200,
                                dict(command=analysis_command(settings["day"]), **settings["readiness"]),
                            )
                        finally:
                            threading.Thread(target=server.shutdown, daemon=True).start()
                    return
                actions = {
                    "/settings": editor.refresh,
                    "/preview": lambda: editor.preview(request),
                    "/auto-roi": lambda: editor.auto_roi(request),
                    "/save": lambda: editor.save(request),
                    "/locate": lambda: editor.locate(request),
                    "/review/status": editor.review.status,
                    "/review/start": lambda: editor.review.start(request.get("samples"), request.get("overrides")),
                    "/review/details": lambda: editor.review.details(request["sample_no"]),
                    "/review/cancel": editor.review.interrupt,
                    "/review/approve": lambda: editor.review.approve(
                        request["samples"], request["revision"], request.get("order_confirmed") is True
                    )
                    and editor.settings(),
                }
                action = actions.get(urlparse(self.path).path)
                if action is None:
                    raise ValueError("unknown editor action")
                with editor.lock:
                    if closing.is_set():
                        self.reply(409, {"error": "設定画面は終了中です"})
                        return
                    self.reply(200, action())
            except (ValueError, KeyError, TypeError, OSError) as error:
                self.reply(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    setattr(server, "review", editor.review)
    return server


def analysis_command(day):
    return "python scripts/04_repellent_response_analysis.py --day " + shlex.quote(day)


def saved_readiness(settings, review_state=None):
    """Report persisted choices for all samples, including those hidden in the editor."""
    batch = settings.get("review")
    missing = []
    for sample in selected_samples(settings):
        reasons = missing_tiff_setup(sample)
        if batch is not None and sample["sample_no"] not in batch["accepted_samples"]:
            reasons = ["一括承認対象外（保留）"] + reasons
        if reasons:
            missing.append(dict(sample_no=sample["sample_no"], missing=reasons))
    blocked = {s["sample_no"] for s in missing}
    ready = [s["sample_no"] for s in settings["samples"] if s["sample_no"] not in blocked]
    ordered = settings.get("order_confirmed") is True
    command = analysis_command(settings["day"])
    executable = None
    if ordered and ready:
        executable = (
            command if batch is not None or not missing else command + " --input-samples " + " ".join(map(str, ready))
        )
    lines = ["保存済みの解析可能No.: " + (", ".join(map(str, ready)) if ordered and ready else "なし")]
    pending = [
        s["sample_no"]
        for s in settings["samples"]
        if s["sample_no"] in ready
        and (review_state or {}).get("entries", {}).get(str(s["sample_no"]), {}).get("artifact")
        and s.get("review_artifact") != review_state["entries"][str(s["sample_no"])]["artifact"]
    ]
    if pending:
        lines.append(
            "未承認の新しい候補No.: " + ", ".join(map(str, pending)) + "。下のコマンドは前回承認済みの条件を使います。"
        )
    if not ordered:
        lines.append("不足: 撮影順の確認・保存")
    lines.extend(f"No.{s['sample_no']} 不足: {'・'.join(s['missing'])}" for s in missing)
    if executable:
        lines.extend(["今すぐ解析できるコマンド:", executable])
    else:
        lines.append("対象指定・確認済み保存が完了するまで解析できません。終了操作は保存しません。")
    if batch is None and (missing or not ordered):
        lines.extend(["全サンプルを確認済み保存した後のコマンド（現在は全件実行不可）:", command])
    return dict(
        sample_count=len(settings["samples"]),
        ready_samples=ready if ordered else [],
        missing_samples=missing,
        executable_command=executable,
        summary="\n".join(lines),
        pending_review_samples=pending,
    )


def serve_editor(path, numbers=None, confirm_order=False):
    server = editor_server(path, numbers, confirm_order, None, True)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(
        f"サンプル設定画面: {url}\n自動設定・動画確認後、まとめて承認して保存します。終了は画面の終了ボタン（Ctrl-Cも可）。",
        flush=True,
    )
    try:
        webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(server, "review"):
            server.review.close()
        server.server_close()
        state = server.review.status() if hasattr(server, "review") else None
        print("設定画面を終了しました。\n" + saved_readiness(_load(path), state)["summary"], flush=True)
