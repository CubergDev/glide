"""The REFUSALS table in conftest.py, proved row by row with synthetic modules.

Most rows name code that does not exist in the tree yet (`sounddevice`, `uvicorn`, `mcp` stdio, `keyring`, the pet, the
webhook listener, `execution.dom`) or is not installed here. Each row is exercised against a stand-in module that has the
named attribute, with a "real" implementation that records any call that gets through. A row passes when calling the
attribute, or touching anything on it, raises the guard's error and nothing recorded.
"""

from __future__ import annotations

import sys
import types

import conftest
import pytest
from conftest import REFUSALS, _Refused

PAIRS = [(spec, path) for spec, paths in REFUSALS for path in paths]
PROBE = "glide._guard_probe"


def stand_in(spec: str, paths: list[str], reached: list[str]) -> tuple[str, types.ModuleType]:
    """A module under the name the row expects, whose every named attribute is a 'real' function that records a call."""
    name = f"{PROBE}.{spec.removeprefix('glide:')}" if spec.startswith("glide:") else spec
    module = types.ModuleType(name)
    for path in paths:
        *chain, leaf = path.split(".")
        owner: object = module
        for part in chain:
            if not hasattr(owner, part):
                setattr(owner, part, type(part, (), {}))
            owner = getattr(owner, part)
        setattr(owner, leaf, lambda *args, _path=path, **kwargs: reached.append(_path))
    return name, module


def resolve(module: types.ModuleType, path: str) -> object:
    target: object = module
    for part in path.split("."):
        target = getattr(target, part)
    return target


@pytest.mark.parametrize(("spec", "path"), PAIRS, ids=[f"{spec}:{path}" for spec, path in PAIRS])
def test_every_row_refuses_a_call_that_would_reach_the_machine(spec, path, monkeypatch):
    reached: list[str] = []
    name, module = stand_in(spec, [path], reached)
    monkeypatch.setitem(sys.modules, name, module)
    conftest.apply_refusals(monkeypatch)
    target = resolve(module, path)
    assert isinstance(target, _Refused), f"{spec}:{path} was not refused"
    with pytest.raises(RuntimeError, match="real machine"):
        target()
    with pytest.raises(RuntimeError, match="real machine"):
        _ = target.anything_at_all
    assert reached == []


@pytest.mark.parametrize(("spec", "path"), PAIRS[:1] + PAIRS[-1:], ids=["first-row", "last-row"])
def test_the_refusal_is_undone_with_the_test(spec, path):
    reached: list[str] = []
    name, module = stand_in(spec, [path], reached)
    original = resolve(module, path)
    sys.modules[name] = module
    mp = pytest.MonkeyPatch()
    try:
        conftest.apply_refusals(mp)
        assert isinstance(resolve(module, path), _Refused)
    finally:
        mp.undo()
        del sys.modules[name]
    assert resolve(module, path) is original


def test_a_row_whose_module_or_attribute_is_absent_is_skipped_without_failure(monkeypatch):
    # A module with none of the named attributes, and one that is not installed at all.
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    guard = conftest.apply_refusals(monkeypatch)
    assert guard.ban(types.ModuleType("empty"), "nothing", "x") is False
    assert conftest._import_quiet("glide_guard_no_such_module_anywhere") is None


def test_a_strict_refusal_fails_loudly_when_its_target_is_renamed_away(monkeypatch):
    guard = conftest.Guard(monkeypatch)
    with pytest.raises(AttributeError, match="no longer exists"):
        guard.strict(types.ModuleType("m"), "gone", "m.gone")


def test_a_stand_in_module_that_refuses_every_attribute_is_skipped_not_crashed(monkeypatch):
    """Off Windows `win32api` is a stand-in whose every unknown attribute raises RuntimeError."""
    module = types.ModuleType("win32api")

    def refuse_all(attr: str) -> object:
        raise RuntimeError(f"win32api.{attr} is unavailable on this OS")

    module.__getattr__ = refuse_all
    monkeypatch.setitem(sys.modules, "win32api", module)
    conftest.apply_refusals(monkeypatch)  # must not raise
    with pytest.raises(RuntimeError, match="unavailable on this OS"):
        _ = module.mouse_event


def test_a_copy_taken_at_import_time_is_refused_too(monkeypatch):
    """`from glide.x import y` copies y at import time; refusing x.y alone leaves the copy live. This is how
    `dom._get_json` got past the `cdp._get_json` refusal."""
    reached: list[str] = []
    name, module = stand_in("sounddevice", ["play"], reached)
    copy_module = types.ModuleType(f"{PROBE}.consumer")
    copy_module.play = module.play  # what `from sounddevice import play` does
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, copy_module.__name__, copy_module)
    conftest.apply_refusals(monkeypatch)
    assert isinstance(copy_module.play, _Refused)
    with pytest.raises(RuntimeError, match="real machine"):
        copy_module.play()
    assert reached == []


def test_every_row_that_is_installed_here_is_refused_in_a_normal_test():
    """With real modules (the `all` extra, or this tree's own adapters) the fixture has already applied the row."""
    checked = 0
    for spec, paths in REFUSALS:
        for module in conftest._modules(spec):
            if module.__name__.startswith(PROBE):
                continue
            for path in paths:
                *chain, leaf = path.split(".")
                owner: object = module
                for part in chain:
                    owner = conftest._probe(owner, part)
                if owner is conftest._MISSING or conftest._probe(owner, leaf) is conftest._MISSING:
                    continue
                assert isinstance(getattr(owner, leaf), _Refused), f"{module.__name__}.{path} is live in a test"
                checked += 1
    assert checked, "no row matched a real module; the tree's macOS adapter alone should match several"
