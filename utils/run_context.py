"""Local execution records and a mandatory clean-Git gate for analysis CLIs."""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
from importlib.metadata import distributions
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

REPOSITORY = Path(__file__).resolve().parents[1]


def _git(repository, *arguments):
    return subprocess.check_output(
        ["git", "-C", str(repository), *arguments], stderr=subprocess.PIPE, text=True
    ).strip()


def require_clean_git(repository):
    """Fail closed for dirty, unversioned, or incorrectly located source trees."""
    repository = Path(repository).resolve()
    try:
        if Path(_git(repository, "rev-parse", "--show-toplevel")).resolve() != repository:
            raise RuntimeError("source directory is not the Git repository root")
        commit = _git(repository, "rev-parse", "--verify", "HEAD")
        branch = _git(repository, "rev-parse", "--abbrev-ref", "HEAD")
        status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise RuntimeError(
            "Git状態を確認できないため実行を中止。commit済みのGit作業ツリーで実行してください。"
        ) from exc
    if status:
        preview = "\n".join(status.splitlines()[:20])
        raise RuntimeError(
            "未コミット変更があるため実行を中止。変更を確認し、必要なものだけcommitまたはstashしてください。\n"
            + preview
        )
    return {"commit": commit, "branch": branch, "is_clean": True, "detached_head": branch == "HEAD"}


def _now():
    return datetime.now(ZoneInfo("Asia/Tokyo"))


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fingerprint(path):
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {"path": str(path), "size_bytes": after.st_size, "sha256": digest.hexdigest()}


def input_inventory(paths):
    """Inventory the declared input scope, including absent optional caches.

    Directories are conservative candidate inputs, not a file-access trace.
    Follow directory symlinks while visiting each resolved directory only once.
    """
    files = set()
    missing = set()
    directories = set()
    for item in paths:
        if item is None:
            continue
        pending = [Path(item)]
        while pending:
            path = pending.pop().resolve()
            if not path.exists():
                missing.add(path)
            elif path.is_file():
                files.add(path)
            elif path.is_dir() and path not in directories:
                directories.add(path)
                pending.extend(path.iterdir())
    return {
        "scope": "declared candidate inputs before execution; not an access trace",
        "files": [_fingerprint(path) for path in sorted(files)],
        "missing": [str(path) for path in sorted(missing)],
    }


class _Tee:
    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

    def write(self, text):
        count = self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return count

    def flush(self):
        self.console.flush()
        self.logfile.flush()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def analysis_run(*, parameters, inputs=(), outputs=(), repository=None):
    """Record one CLI run without changing its scientific output layout.

    Developer tests and library functions remain usable on uncommitted code.
    CLI entry points must enter this context before creating analysis outputs.
    """
    repository = Path(repository or REPOSITORY).resolve()
    git = require_clean_git(repository)
    started = _now()
    run_id = started.strftime("%H%M%S_%f") + "_" + uuid4().hex[:8]
    root = repository / "outputs" / "_runs" / started.strftime("%Y-%m-%d") / run_id
    if not root.resolve().is_relative_to(repository / "outputs"):
        raise ValueError("Run logs must stay inside the repository's private outputs directory")
    root.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": started.isoformat(),
        "timezone": "Asia/Tokyo",
        "git": git,
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "parameters": parameters,
        "declared_outputs": [str(Path(path).resolve()) for path in outputs if path is not None],
        "log": str(root / "run.log"),
    }
    manifest = root / "manifest.json"
    _write_json(manifest, record)
    clock = time.monotonic()
    with (root / "run.log").open("a", encoding="utf-8", buffering=1) as logfile:
        handler = logging.StreamHandler(logfile)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger = logging.getLogger()
        console_handler = None
        if not logger.handlers:
            # Preserve the console behavior of logging.warning/basicConfig when
            # this context is the first code to configure the root logger.
            console_handler = logging.StreamHandler(sys.stderr)
            logger.addHandler(console_handler)
        logger.addHandler(handler)
        try:
            with redirect_stdout(_Tee(sys.stdout, logfile)), redirect_stderr(_Tee(sys.stderr, logfile)):
                try:
                    print(f"Run record: {manifest}", file=sys.stderr, flush=True)
                    print(f"git.branch={git['branch']} git.commit={git['commit']}", file=sys.stderr, flush=True)
                    record["environment"] = {
                        "python": sys.version,
                        "executable": sys.executable,
                        "platform": platform.platform(),
                        "packages": sorted(
                            [{"name": d.metadata["Name"], "version": d.version} for d in distributions()],
                            key=lambda d: (d["name"] or "").lower(),
                        ),
                    }
                    record["inputs"] = input_inventory(inputs)
                    # Hashing may take time; check again before executing analysis code.
                    if require_clean_git(repository) != git:
                        raise RuntimeError("Git revision changed during input inventory")
                    _write_json(manifest, record)
                    yield root
                    if require_clean_git(repository) != git:
                        raise RuntimeError("Git revision changed during execution")
                    record["status"] = "complete"
                except BaseException as exc:
                    record["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
                    if isinstance(exc, SystemExit) and exc.code in (None, 0):
                        record["status"] = "complete"
                    record["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    traceback.print_exc(file=logfile)
                    raise
                finally:
                    record["finished_at"] = _now().isoformat()
                    record["elapsed_seconds"] = time.monotonic() - clock
                    _write_json(manifest, record)
                    print(f"Run {record['status']}: {manifest}", file=sys.stderr, flush=True)
        finally:
            logger.removeHandler(handler)
            handler.close()
            if console_handler is not None:
                logger.removeHandler(console_handler)
                console_handler.close()
