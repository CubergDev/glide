"""execute() and undo(): the only code that changes the disk. Every path is re-validated at run time, nothing is
ever overwritten, every move is checked from a fresh look, and an undo manifest exists before the first move."""

from __future__ import annotations

import contextlib
import errno
import hmac
import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from .model import ActionResult, Approval, Plan, Report, Stamp
from .safety import Refused, check_name, check_root, unique_name

StopCheck = Callable[[], bool]


def _look(path: str) -> Stamp | None:
    """A fresh look at `path` itself (symlinks not followed): its stamp, or None when nothing can be seen there."""
    try:
        info = os.lstat(path)
    except OSError:
        return None
    return (info.st_ino, info.st_size, info.st_mtime_ns)


def _device(path: str) -> int:
    return os.lstat(path).st_dev


def _split(root: str, destination: str) -> tuple[str, str]:
    """(folder, name) of a destination that must be exactly <root>/<folder>/<name>, each one safe component."""
    if not destination.startswith(root + os.sep):
        raise Refused("a destination is outside the folder being organized")
    parts = destination[len(root) + 1 :].split(os.sep)
    if len(parts) != 2:
        raise Refused("a destination must be exactly one folder below the root")
    return check_name(parts[0], "folder name"), check_name(parts[1], "file name")


def _check_source(root: str, source: str) -> None:
    if os.path.dirname(source) != root:
        raise Refused("a source is not a direct child of the folder being organized")
    check_name(os.path.basename(source), "file name")


def _status(results: list[ActionResult], neutral: int = 0) -> str:
    done = sum(r.status == "done" for r in results)
    bad = len(results) - done - neutral
    if bad <= 0:
        return "ok"
    return "partial" if done else "failed"


def _move(src: str, dst: str, stamp: Stamp) -> tuple[str, str]:
    """rename src to dst, then decide from a fresh look: done, failed (source intact) or uncertain. Never retried."""
    refused = None
    try:
        os.rename(src, dst)
    except OSError as error:
        refused = error
    at_dst, at_src = _look(dst), _look(src)
    if at_dst == stamp and at_src is None:
        return "done", ""
    if refused is not None and at_src == stamp and at_dst is None:
        if refused.errno == errno.EXDEV:
            return "failed", "different volume: nothing was copied"
        return "failed", f"could not move ({errno.errorcode.get(refused.errno or 0, 'error')})"
    return "uncertain", "the result of the move could not be confirmed: reconcile by hand"


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def execute(
    plan: Plan,
    approval: Approval,
    *,
    manifest_dir: Path | str,
    home: Path | str | None = None,
    should_stop: StopCheck | None = None,
) -> Report:
    """Run `plan`, only if `approval` carries exactly its hash. Refuses the whole plan (nothing touched) when it is
    altered or any path is unsafe; individual files that changed since planning fail on their own."""
    if not isinstance(approval, Approval) or not isinstance(approval.plan_hash, str):
        raise Refused("an approval for this plan is required")
    if not hmac.compare_digest(approval.plan_hash, plan.plan_hash):
        raise Refused("the approval is for a different plan")
    if plan.compute_hash() != plan.plan_hash:
        raise Refused("the plan does not match its hash: it was changed after it was made")
    root = plan.root
    if check_root(root, home) != root:
        raise Refused("the folder is no longer where the plan was made (a link or move changed it)")
    taken: dict[str, set[str]] = {}
    targets: list[tuple[str, str, str]] = []  # source, folder, resolved destination
    for move in plan.moves:
        _check_source(root, move.source)
        folder, name = _split(root, move.destination)
        name = unique_name(os.path.join(root, folder), name, taken.setdefault(folder, set()))
        targets.append((move.source, folder, os.path.join(root, folder, name)))
    if not targets:
        return Report("ok", ())

    runs = Path(manifest_dir)
    entries = [
        {"source": s, "destination": d, "stamp": list(m.stamp), "status": "planned"}
        for (s, _, d), m in zip(targets, plan.moves, strict=True)
    ]
    data = {"root": root, "intent": plan.intent, "plan_hash": plan.plan_hash, "entries": entries, "folders_created": []}
    try:
        runs.mkdir(parents=True, exist_ok=True)
        manifest = runs / f"{time.strftime('%Y%m%dT%H%M%S')}-{plan.plan_hash[:8]}.json"
        n = 1
        while manifest.exists():
            n += 1
            manifest = runs / f"{time.strftime('%Y%m%dT%H%M%S')}-{plan.plan_hash[:8]}-{n}.json"
        _write(manifest, data)
    except OSError as error:
        raise Refused("the undo manifest could not be written, so nothing was moved") from error

    results: list[ActionResult] = []
    created: list[str] = []
    halted = ""
    for i, ((source, folder, destination), move) in enumerate(zip(targets, plan.moves, strict=True)):

        def finish(status: str, reason: str = "", i=i, source=source, destination=destination) -> None:
            nonlocal halted
            results.append(ActionResult(source, destination, status, reason))
            entries[i]["status"] = status
            try:
                _write(manifest, data)
            except OSError:
                halted = halted or "the undo manifest could not be updated"
            if status == "uncertain":
                halted = "an earlier action is uncertain: reconcile before continuing"

        if halted:
            results.append(ActionResult(source, destination, "skipped", halted))
            entries[i]["status"] = "skipped"
            continue
        if should_stop is not None and should_stop():
            halted = "stopped"
            results.append(ActionResult(source, destination, "skipped", "stopped"))
            entries[i]["status"] = "skipped"
            continue
        stamp = _look(source)
        if stamp is None:
            finish("failed", "the file is gone")
            continue
        if os.path.islink(source):
            finish("failed", "the source is a symlink")
            continue
        if stamp != tuple(move.stamp):
            finish("failed", "the file changed since it was planned")
            continue
        parent = os.path.join(root, folder)
        if os.path.islink(parent):
            finish("failed", "the destination folder is a symlink")
            continue
        if os.path.lexists(parent) and not os.path.isdir(parent):
            finish("failed", "the destination folder is not a folder")
            continue
        try:
            same = _device(source) == _device(parent if os.path.lexists(parent) else root)
        except OSError:
            same = False
        if not same:
            finish("failed", "different volume: nothing was copied")
            continue
        made = False
        if not os.path.lexists(parent):
            try:
                os.mkdir(parent)
                made = True
            except FileExistsError:
                pass
            except OSError:
                finish("failed", "could not create the destination folder")
                continue
            if made:
                created.append(folder)
                data["folders_created"] = list(created)
                try:
                    _write(manifest, data)
                except OSError:
                    halted = "the undo manifest could not be updated"
        if os.path.lexists(destination):
            finish("failed", "the destination name exists")
        else:
            status, reason = _move(source, destination, stamp)
            finish(status, reason)
        if made and results[-1].status == "failed" and not os.listdir(parent):
            os.rmdir(parent)
            created.remove(folder)
            data["folders_created"] = list(created)
            with contextlib.suppress(OSError):
                _write(manifest, data)
    return Report(_status(results), tuple(results), manifest, tuple(created))


def _load(manifest: Path | str, home: Path | str | None) -> tuple[Path, dict, str]:
    path = Path(manifest)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        root = data["root"]
        entries = data["entries"]
        folders = data["folders_created"]
        if not (isinstance(root, str) and isinstance(entries, list) and isinstance(folders, list)):
            raise TypeError
        for e in entries:
            if not all(isinstance(e[k], str) for k in ("source", "destination", "status")) or len(e["stamp"]) != 3:
                raise TypeError
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise Refused("that is not a readable undo manifest") from error
    if check_root(root, home) != root:
        raise Refused("the manifest's folder is not a safe real folder")
    for e in entries:
        _check_source(root, e["source"])
        _split(root, e["destination"])
    for folder in folders:
        check_name(folder, "folder name")
    return path, data, root


def undo(manifest: Path | str, *, home: Path | str | None = None, should_stop: StopCheck | None = None) -> Report:
    """Reverse a run, last move first. A destination that is gone or a symlink, or an original path that exists again,
    fails that entry alone; only empty folders the plan created are removed."""
    path, data, root = _load(manifest, home)
    entries = data["entries"]
    results: list[ActionResult] = []
    neutral = 0
    halted = ""

    def save() -> None:
        with contextlib.suppress(OSError):
            _write(path, data)

    for entry in reversed(entries):
        source, destination, state = entry["source"], entry["destination"], entry["status"]
        if state != "done":
            reason = {
                "undone": "already undone",
                "uncertain": "outcome unknown: reconcile by hand",
                "undo_uncertain": "outcome unknown: reconcile by hand",
            }.get(state, "that move did not happen")
            results.append(ActionResult(destination, source, "skipped", reason))
            neutral += "reconcile" not in reason
            continue
        if halted:
            results.append(ActionResult(destination, source, "skipped", halted))
            continue
        if should_stop is not None and should_stop():
            halted = "stopped"
            results.append(ActionResult(destination, source, "skipped", "stopped"))
            continue
        folder = os.path.dirname(destination)
        stamp = _look(destination)
        problem = ""
        if stamp is None:
            problem = "the moved file is gone"
        elif os.path.islink(destination) or os.path.islink(folder):
            problem = "the moved file or its folder is a symlink"
        elif os.path.lexists(source):
            problem = "the original path exists again"
        else:
            try:
                if _device(destination) != _device(root):
                    problem = "different volume: nothing was copied"
            except OSError:
                problem = "the volume could not be checked"
        if problem:
            results.append(ActionResult(destination, source, "failed", problem))
            continue
        status, reason = _move(destination, source, stamp)
        results.append(ActionResult(destination, source, status, reason))
        if status == "done":
            entry["status"] = "undone"
        elif status == "uncertain":
            entry["status"] = "undo_uncertain"
            halted = "an earlier action is uncertain: reconcile before continuing"
        save()
    removed: list[str] = []
    for folder in data["folders_created"]:
        target = os.path.join(root, folder)
        if os.path.isdir(target) and not os.path.islink(target) and not os.listdir(target):
            try:
                os.rmdir(target)
                removed.append(folder)
            except OSError:
                pass
    data["folders_created"] = [f for f in data["folders_created"] if f not in removed]
    save()
    return Report(_status(results, neutral), tuple(results), path, folders_removed=tuple(removed))
