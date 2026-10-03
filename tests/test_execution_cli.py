"""`glide-computer --engine structured`: how it is wired, and that every failed run exits nonzero. In-process."""

from types import SimpleNamespace

import pytest
from execution_world import Computer, Jev, Reasoner, response

from glide.computer import cli
from glide.computer.execution import engine
from glide.computer.execution.contracts import Milestone
from glide.computer.models import BrowserConnectionError


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr(cli.config, "load_dotenv", lambda _: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-key")
    monkeypatch.setattr(cli.config, "writer_vision", lambda: True)
    monkeypatch.setattr(cli, "desktop", SimpleNamespace(accessibility_trusted=lambda: True, abort_hint=lambda: "corner"))


def state(outcome, failure=""):
    return SimpleNamespace(outcome=outcome, failure=failure)


def chains(classifier="the chain classifier"):
    """The loaded configuration `glide` hands to `main`: only what the command reads from it."""
    return SimpleNamespace(classifier=lambda: classifier, scrub=lambda text: text)


@pytest.mark.parametrize(
    ("outcome", "failure", "code"),
    [
        ("aborted", "", 130),
        ("aborted (Ctrl-C)", "", 130),
        ("blocked", "A reason.", 1),
        ("blocked", "", 1),
        ("unsupported", "", 1),
        ("crashed", "", 1),
        ("provider failure", "", 1),
        ("step limit", "The execution budget ended.", 1),
        ("desktop unavailable", "No permission.", 1),
        ("done", "", None),
        ("dry run", "", None),
        ("stalled", "", None),
    ],
)
def test_every_failed_outcome_exits_nonzero_and_a_stop_exits_130(monkeypatch, tmp_path, offline, outcome, failure, code):
    monkeypatch.setattr(cli, "make_writer", lambda *a: None)
    monkeypatch.setattr(cli, "run", lambda *a, **kw: state(outcome, failure))
    assert cli.main(["a goal", "--out", str(tmp_path)], chains()) == (code or 0)


def test_the_structured_engine_takes_its_classifier_from_the_provider_chains(monkeypatch, tmp_path, offline):
    seen, loaded = {}, chains()
    monkeypatch.setattr(cli, "make_writer", lambda config=None: seen.setdefault("writer_config", config))

    def fake_run(cfg, ctx_factory, **kwargs):
        seen.update(cfg=cfg, **kwargs)
        return state("done")

    monkeypatch.setattr(cli, "run", fake_run)
    code = cli.main(["a goal", "--engine", "structured", "--readiness-timeout", "4", "--out", str(tmp_path)], loaded)
    assert code == 0 and seen["cfg"].engine == "structured" and seen["cfg"].readiness_timeout == 4.0
    assert seen["classifier_factory"] is loaded.classifier and seen["writer_config"] is loaded


def test_the_legacy_engine_takes_its_classifier_from_the_chains_too(monkeypatch, tmp_path, offline):
    # Changed from "keeps its own classifier and never loads the chains": `glide` loads glide.toml once for every
    # command and `main` always hands over the chain's classifier (D7), so no engine has a client that skips failover.
    seen, loaded = {}, chains()
    monkeypatch.setattr(cli, "make_writer", lambda *a: None)
    monkeypatch.setattr(cli, "run", lambda cfg, ctx_factory, **kw: seen.update(cfg=cfg, **kw) or state("done"))
    assert cli.main(["a goal", "--out", str(tmp_path)], loaded) == 0
    assert seen["cfg"].engine == "legacy" and seen["classifier_factory"] is loaded.classifier


def run_structured(monkeypatch, tmp_path, computer, writer, jev, *flags):
    """The command line, the real runner and the real engine, over fakes: only the backend and the models are replaced.
    Returns the exit code."""
    monkeypatch.setattr(cli, "make_writer", lambda config=None: writer)
    monkeypatch.setattr(cli, "provider", lambda _: "offline fixture")
    monkeypatch.setattr(engine, "make_backend", lambda _: computer)
    argv = ["Open the page", "--engine", "structured", "--act", "--readiness-timeout", "0", "--out", str(tmp_path), *flags]
    return cli.main(argv, chains(jev))


def test_a_structured_run_that_finishes_exits_zero(monkeypatch, tmp_path, offline):
    computer = Computer()
    step = Milestone("open", "Open the page", "url", value="https://example.net")
    assert run_structured(monkeypatch, tmp_path, computer, Reasoner([response(step)]), Jev("plan")) == 0
    assert [a.kind for a in computer.actions] == ["navigate"]


def test_a_structured_run_that_cannot_reach_the_browser_exits_one_with_zero_actions(monkeypatch, tmp_path, offline):
    computer = Computer()

    def refused(_):
        raise BrowserConnectionError("cdp", "http://127.0.0.1:9222", ConnectionRefusedError(61, "refused"))

    computer.on_inspect = refused
    assert run_structured(monkeypatch, tmp_path, computer, Reasoner([]), Jev("plan")) == 1
    assert not computer.actions


def test_a_structured_run_that_is_unsupported_exits_one(monkeypatch, tmp_path, offline):
    computer = Computer()
    assert run_structured(monkeypatch, tmp_path, computer, Reasoner([response(unsupported=["drag"])]), Jev("plan")) == 1
    assert not computer.actions
