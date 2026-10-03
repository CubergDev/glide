"""`glide memory` run in-process (D13): no subprocess, no os.fork. Memory is opt-in, so the off state is tested first."""

import io
import json

import pytest

from glide.memory import cli


@pytest.fixture
def world(tmp_path):
    data = tmp_path / "data"
    off = tmp_path / "off.toml"
    off.write_text("")
    on = tmp_path / "on.toml"
    on.write_text("[memory]\nenabled = true\n")
    return {"data": data, "off": off, "on": on, "home": tmp_path / "home", "env": {"GLIDE_DATA_DIR": str(data)}}


def run(world, config, *args, user=("--user", "user")):
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(
        ["--config", str(world[config]), *user, "--project", "project", *args],
        environ=world["env"],
        home=world["home"],
        out=out,
        err=err,
    )
    return code, out.getvalue(), err.getvalue()


def test_every_command_refuses_while_memory_is_off_and_creates_nothing(world):
    for command in (("init",), ("recall",), ("events",), ("remember", "k", "text"), ("forget", "id"), ("proposals",)):
        code, out, err = run(world, "off", *command)
        assert code == 2 and out == "" and "memory is off" in err
    assert not world["data"].exists()


def test_status_works_while_off_and_reports_without_creating(world):
    code, out, _ = run(world, "off", "status")
    result = json.loads(out)
    assert code == 0 and result["enabled"] is False and result["exists"] is False
    assert not world["data"].exists()


def test_environment_can_turn_memory_on_over_an_off_file(world):
    world["env"]["GLIDE_MEMORY"] = "1"
    code, out, _ = run(world, "off", "init")
    assert code == 0 and json.loads(out)["revision"] == 0 and world["data"].exists()


def test_persistence_recall_forget_round_trip(world):
    code, out, err = run(world, "on", "init")
    assert code == 0, err
    assert (world["data"] / "memory" / "memory.sqlite").is_file()
    code, out, err = run(world, "on", "remember", "style", "concise answers")
    assert code == 0, err
    identifier = json.loads(out)
    code, out, _ = run(world, "on", "recall")
    assert json.loads(out)[0]["text"] == "concise answers"
    assert run(world, "on", "events")[0] == 0
    assert json.loads(run(world, "on", "forget", identifier)[1]) == {"forgotten": True}
    assert json.loads(run(world, "on", "recall")[1]) == []
    status = json.loads(run(world, "on", "status")[1])
    assert status["enabled"] is True and status["exists"] is True


def test_invalid_scope_has_a_controlled_error(world):
    code, out, err = run(world, "on", "init", user=("--user", " "))
    assert code == 2 and out == ""
    assert "Traceback" not in err and "nonempty" in err


def test_credentials_are_refused_without_echoing_them(world):
    secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
    code, out, err = run(world, "on", "remember", "k", f"my key is {secret}")
    assert code == 2 and secret not in out + err and "credentials" in err


def test_a_bad_config_is_a_controlled_error(world, tmp_path):
    world["on"].write_text('[memory]\nenabled = "yes"\n')
    code, _, err = run(world, "on", "status")
    assert code == 2 and "enabled" in err


def test_propose_apply_rollback_and_plan(world, tmp_path):
    assert run(world, "on", "init")[0] == 0
    # evidence needs a recorded outcome; use the library through the same data directory
    from glide.memory import MemoryService, MemorySettings, Scope

    with MemoryService.open(MemorySettings(enabled=True, data_dir=world["data"])) as service:
        evidence = service.store.record_outcome(Scope("user", "project", "cli"), "run-1", True, "Use concise steps")
    code, out, err = run(world, "on", "propose", "prompt:lesson", "Use concise steps", "--evidence", evidence)
    assert code == 0, err
    proposal = json.loads(out)
    assert json.loads(run(world, "on", "proposals")[1])[0]["status"] == "draft"
    code, out, _ = run(world, "on", "apply", proposal, "--revision", "0")
    assert code == 0 and json.loads(out) == 1
    assert run(world, "on", "apply", proposal, "--revision", "1")[0] == 2  # already applied: refused cleanly
    assert json.loads(run(world, "on", "rollback", proposal, "--revision", "1")[1]) == 2
    catalog = tmp_path / "catalog"
    catalog.mkdir()
    code, out, err = run(world, "on", "plan", "draft a reply", "--catalog", str(catalog), "--model", "label", "--window", "32768")
    assert code == 0, err
    assert json.loads(out)["model"] == "label"


def test_argument_errors_exit_through_argparse(world):
    with pytest.raises(SystemExit) as caught:
        run(world, "on", "remember", "only-a-key")
    assert caught.value.code == 2
