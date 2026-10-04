"""The REFUSALS table in conftest.py, proved row by row with synthetic modules, and kept honest against the real ones.

Most rows name code that is not installed here (`sounddevice`, `uvicorn`, `mcp` stdio, `keyring`) or not in the tree yet.
Each row is exercised against a stand-in module that has the named attribute, with a "real" implementation that records any
call that gets through. A row passes when calling the attribute, or touching anything on it, raises the guard's error and
nothing recorded. Separately, a row whose module IS importable must resolve every name it lists, so a rename cannot leave a
row dead.
"""

from __future__ import annotations

import importlib.machinery
import os
import socket
import subprocess
import sys
import types

import conftest
import pytest
from conftest import REFUSALS, UNPORTED, _Refused

pytestmark = pytest.mark.expect_refusals  # every test here fires the guard on purpose

PROBE = "glide._guard_probe"


def names_in(path: str) -> list[str]:
    """The attribute names a row path stands for: `prefix*` is one sample member of the family, `a|b` both."""
    *chain, last = path.lstrip("?").split(".")
    leaves = [last[:-1] + "Probe"] if last.endswith("*") else last.split("|")
    return [".".join([*chain, leaf]) for leaf in leaves]


PAIRS = [(spec, name) for spec, paths in REFUSALS for path in paths for name in names_in(path)]


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
    previous = sys.modules.get(name)  # a row may name a real module (`_posixsubprocess`): put it back, never delete it
    sys.modules[name] = module
    mp = pytest.MonkeyPatch()
    try:
        conftest.apply_refusals(mp)
        assert isinstance(resolve(module, path), _Refused)
    finally:
        mp.undo()
        if previous is None:
            del sys.modules[name]
        else:
            sys.modules[name] = previous
    assert resolve(module, path) is original


# ------------------------------------------------------------------ a missing name is loud, a missing module is not


def installed_module(name: str, **attributes) -> types.ModuleType:
    """A module that looks installed: it has a spec whose origin is outside the tests directory."""
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, None, origin=os.__file__)
    vars(module).update(attributes)
    return module


def test_a_missing_attribute_of_an_installed_module_fails_loudly():
    module = installed_module("fake_framework", present=lambda: None)
    with pytest.raises(AttributeError, match="does not have"):
        conftest.resolve_row("fake_framework", module, "renamed_away")
    with pytest.raises(AttributeError, match="does not have"):
        conftest.resolve_row("fake_framework", module, "Missing.method")
    with pytest.raises(AttributeError, match="does not have"):
        conftest.resolve_row("fake_framework", module, "NoSuchFamily*")
    assert [name for _, name, _ in conftest.resolve_row("fake_framework", module, "present")] == ["present"]


def test_an_alternative_a_wildcard_and_an_optional_name_are_the_only_ways_to_be_absent():
    module = installed_module("fake_framework", old_name=lambda: None, FamilyA=lambda: None, FamilyB=lambda: None, other=1)
    resolved = lambda path: [name for _, name, _ in conftest.resolve_row("fake_framework", module, path)]  # noqa: E731
    assert resolved("old_name|new_name") == ["old_name"]
    assert resolved("Family*") == ["FamilyA", "FamilyB"]
    assert resolved("?never_existed") == []
    with pytest.raises(AttributeError):
        conftest.resolve_row("fake_framework", module, "old_name2|new_name2")
    with pytest.raises(AttributeError):
        resolved("other*")  # a constant is not a callable to refuse, so the family is empty


def test_a_row_whose_module_is_absent_or_not_real_is_skipped_without_failure(monkeypatch):
    # A hand-made module with none of the named attributes, and one that is not installed at all.
    monkeypatch.setitem(sys.modules, "sounddevice", types.ModuleType("sounddevice"))
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    guard = conftest.apply_refusals(monkeypatch)
    assert guard.ban(types.ModuleType("empty"), "nothing", "x") is False
    assert conftest._import_quiet("glide_guard_no_such_module_anywhere") is None


def test_a_namespace_directory_with_the_modules_name_is_not_the_installed_package():
    """With the MCP SDK absent `import mcp` finds `tests/mcp`, a namespace package: it must not make the rows fail."""
    namespace = types.ModuleType("mcp")
    namespace.__spec__ = importlib.machinery.ModuleSpec("mcp", None, origin=None, is_package=True)
    namespace.__spec__.submodule_search_locations = [str(conftest.TESTS_DIR / "mcp")]
    assert not conftest._is_real(namespace)
    assert conftest.resolve_row("mcp", namespace, "stdio_client") == []


def test_the_import_only_stand_ins_are_never_installed():
    assert conftest._STUBBED  # off macOS and Windows several exist; on macOS at least the Windows ones do
    for name in conftest._STUBBED:
        assert conftest._import_quiet(name) is None


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


def test_a_strict_refusal_fails_loudly_when_its_target_is_renamed_away(monkeypatch):
    guard = conftest.Guard(monkeypatch)
    with pytest.raises(AttributeError, match="no longer exists"):
        guard.strict(types.ModuleType("m"), "gone", "m.gone")


# ------------------------------------------------------------------ every row resolves


def real_modules(spec: str) -> list[types.ModuleType]:
    return [module for module in conftest._modules(spec) if conftest._is_real(module)]


@pytest.mark.parametrize(
    "spec",
    sorted({spec for spec, _ in REFUSALS}),
)
def test_every_row_resolves_in_every_installed_module_it_names(spec):
    """The names are real: `resolve_row` raises on a name an installed module lacks, so a dead row fails here by name."""
    paths = [path for row_spec, paths in REFUSALS if row_spec == spec for path in paths]
    for module in real_modules(spec):
        for path in paths:
            conftest.resolve_row(spec, module, path)


def test_a_glide_row_matches_a_module_or_is_listed_as_not_ported_yet():
    for spec in {spec for spec, _ in REFUSALS if spec.startswith("glide:")}:
        tail = spec.removeprefix("glide:")
        modules = conftest._modules(spec)
        if tail in UNPORTED:
            assert not modules, f"{spec} now matches {[m.__name__ for m in modules]}: remove {tail!r} from UNPORTED"
        else:
            assert modules, f"{spec} matches no glide module: its row is dead (fix the name, or list it in UNPORTED)"


def test_the_rows_that_name_the_installed_macos_frameworks_cover_real_calls():
    """Not vacuous: on a machine with the frameworks, rows resolved and refused real attributes."""
    quartz = real_modules("Quartz")
    if not quartz:
        pytest.skip("Quartz is not installed here")
    assert isinstance(quartz[0].CGWindowListCopyWindowInfo, _Refused)
    assert isinstance(quartz[0].CGDisplayBounds, _Refused)


def test_every_row_that_is_installed_here_is_refused_in_a_normal_test():
    """With real modules (the `all` extra, or this tree's own adapters) the fixture has already applied the row."""
    checked = 0
    for spec, paths in REFUSALS:
        for module in real_modules(spec):
            for path in paths:
                for owner, name, dotted in conftest.resolve_row(spec, module, path):
                    assert isinstance(getattr(owner, name), _Refused), f"{module.__name__}.{dotted} is live in a test"
                    checked += 1
    assert checked, "no row matched a real module; the tree's macOS adapter alone should match several"


# ------------------------------------------------------------------ a copy taken at import time

# Bound at import, before any guard exists, by modules that are not part of `glide` and not test modules: what
# `from subprocess import Popen` or `from socket import getaddrinfo` does in a library.
HOLDER = types.ModuleType("third_party_probe_holder")
HOLDER.Popen = subprocess.Popen
HOLDER.getaddrinfo = socket.getaddrinfo
sys.modules[HOLDER.__name__] = HOLDER
# A test module's own reference is deliberately left alone: tests keep real references and patch what the body calls.
THIS_FILES_COPY = socket.getaddrinfo


def test_a_copy_taken_at_import_time_by_any_module_is_refused_too():
    assert isinstance(HOLDER.Popen, _Refused)
    assert HOLDER.getaddrinfo is socket.getaddrinfo  # the guard's replacement, not the real function
    assert HOLDER.getaddrinfo is not THIS_FILES_COPY
    with pytest.raises(RuntimeError, match="real machine"):
        HOLDER.Popen(["true"])
    with pytest.raises(RuntimeError, match="real machine"):
        HOLDER.getaddrinfo("example.com", 80)


def test_a_test_modules_own_reference_is_not_rebound():
    assert not isinstance(THIS_FILES_COPY, _Refused)


def test_a_copy_in_a_module_imported_after_the_first_guard_is_refused_when_the_table_is_applied_again(monkeypatch):
    reached: list[str] = []
    name, module = stand_in("sounddevice", ["play"], reached)
    copy_module = types.ModuleType("third_party_late_consumer")
    copy_module.play = module.play  # what `from sounddevice import play` does
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, copy_module.__name__, copy_module)
    conftest.apply_refusals(monkeypatch)
    assert isinstance(copy_module.play, _Refused)
    with pytest.raises(RuntimeError, match="real machine"):
        copy_module.play()
    assert reached == []


def test_a_copy_inside_glide_is_still_refused(monkeypatch):
    reached: list[str] = []
    name, module = stand_in("sounddevice", ["play"], reached)
    copy_module = types.ModuleType(f"{PROBE}.consumer")
    copy_module.play = module.play
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, copy_module.__name__, copy_module)
    conftest.apply_refusals(monkeypatch)
    assert isinstance(copy_module.play, _Refused)


def test_the_module_that_defines_a_function_keeps_the_real_one_for_a_test_that_must_have_it():
    """`websocket.create_connection` is refused, and so is every importer's copy, but `websocket._core.create_connection`
    (the definition) stays reachable: that is where `test_stt` fetches the real function to talk to its loopback server."""
    import websocket

    assert isinstance(websocket.create_connection, _Refused)
    assert not isinstance(websocket._core.create_connection, _Refused)
    assert conftest._is_home(websocket._core, websocket._core.create_connection)
    assert not conftest._is_home(websocket, websocket._core.create_connection)
    assert not conftest._is_home(sys.modules["posix"], os.fork)  # a built-in that re-exports is not a home
