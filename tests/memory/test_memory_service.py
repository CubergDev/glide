"""MemoryService: off means nothing happens on disk; on means private directories and files."""

import os
import stat

import pytest

from glide.memory import MemoryService, MemorySettings, Policy, Scope, SettingsError

SCOPE = Scope("alice", "project", "session")
posix = pytest.mark.skipif(os.name != "posix", reason="file modes require POSIX")


def mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_disabled_returns_none_and_touches_no_disk(tmp_path):
    data = tmp_path / "data"
    assert MemoryService.open(MemorySettings(data_dir=data)) is None
    assert MemoryService.open(MemorySettings.from_mapping({"data_dir": str(data)}, {})) is None
    assert not data.exists()
    assert {p.name for p in tmp_path.iterdir()} <= {"guard-home"}  # only the guard's own fake home


@posix
def test_enabled_creates_private_directories_and_files(tmp_path):
    data = tmp_path / "data"
    with MemoryService.open(MemorySettings(enabled=True, data_dir=data)) as service:
        service.store.remember(SCOPE, "style", "concise")
        assert mode(data) == 0o700
        assert mode(data / "memory") == 0o700
        assert mode(data / "memory" / "catalog") == 0o700
        assert service.database == data / "memory" / "memory.sqlite"
        for suffix in ("", "-wal", "-shm"):
            path = data / "memory" / f"memory.sqlite{suffix}"
            if path.exists():
                assert mode(path) == 0o600, path.name


@posix
def test_existing_memory_dir_is_tightened_but_the_parent_data_dir_is_left_alone(tmp_path):
    data = tmp_path / "shared"
    (data / "memory").mkdir(parents=True)
    data.chmod(0o755)
    (data / "memory").chmod(0o755)
    with MemoryService.open(MemorySettings(enabled=True, data_dir=data)):
        pass
    assert mode(data) == 0o755
    assert mode(data / "memory") == 0o700


def test_a_symlinked_or_file_data_location_is_refused(tmp_path):
    target = tmp_path / "elsewhere"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(SettingsError, match="must be a directory"):
        MemoryService.open(MemorySettings(enabled=True, data_dir=link))
    blocker = tmp_path / "blocker"
    blocker.write_text("")
    with pytest.raises(SettingsError, match="must be a directory"):
        MemoryService.open(MemorySettings(enabled=True, data_dir=blocker))


def test_memory_persists_across_reopen(tmp_path):
    settings = MemorySettings(enabled=True, data_dir=tmp_path / "data")
    with MemoryService.open(settings) as service:
        service.harness.command(SCOPE, "/remember style = short answers")
    with MemoryService.open(settings) as service:
        assert [m["text"] for m in service.store.memories(SCOPE)] == ["short answers"]


def test_auto_capture_follows_the_setting_only(tmp_path):
    sentence = "Remember that I prefer short answers."
    with MemoryService.open(MemorySettings(enabled=True, data_dir=tmp_path / "a")) as service:
        assert service.harness.policy.auto_memory is False
        assert service.harness.observe_user(SCOPE, sentence) == []
        assert service.store.memories(SCOPE) == []
    on = MemorySettings(enabled=True, auto_capture=True, data_dir=tmp_path / "b")
    with MemoryService.open(on, policy=Policy(auto_memory=False)) as service:  # the setting is the only switch
        assert len(service.harness.observe_user(SCOPE, sentence)) == 1
    off_with_policy = MemorySettings(enabled=True, auto_capture=False, data_dir=tmp_path / "c")
    with MemoryService.open(off_with_policy, policy=Policy(auto_memory=True)) as service:
        assert service.harness.observe_user(SCOPE, sentence) == []


def test_closing_closes_the_store(tmp_path):
    service = MemoryService.open(MemorySettings(enabled=True, data_dir=tmp_path / "data"))
    service.close()
    with pytest.raises(RuntimeError, match="closed"):
        service.store.memories(SCOPE)


from memory_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)
