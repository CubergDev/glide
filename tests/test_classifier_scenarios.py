"""Proof that the classifier adapters are drop-ins: the real step loop, run on the simulated computer.

tests/test_scenarios.py drives `runner.run` on `tests/world.py` with a policy standing in for the
classifier (`FakeTypeSafe`). Here the same scenarios, unchanged and with their own assertions, run
with an `LLMClassifier` injected through `runner.run(classifier_factory=...)` instead. Behind it is a
fake LLM that answers from the same policy: it reads the state and the questions back out of the
PROMPT and the options out of the JSON SCHEMA it was sent, asks the policy, and replies in the wire
format the adapter asks for (top options with probabilities). So a scenario that passes has had its
state, its criteria and its answers carried through the prompt, the schema, the parser and the
validation, and the loop made the same decisions as it does with TypeSafe.
"""

from __future__ import annotations

import inspect
import json
import re
from types import SimpleNamespace

import httpx2
import pytest
import test_browser_loop as browser_loop
import test_scenarios
import world
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer
from world import FakeWriter, Page, World, scripted

from glide.computer import runner
from glide.computer.actions import Context
from glide.computer.browser.decide import decide as browser_decide
from glide.computer.browser.decide import verify_typed as browser_verify
from glide.computer.browser.perceive import perceive
from glide.providers.base import ChatResult, ProviderSpec, Usage
from glide.providers.classifier import ChainedClassifier, LLMClassifier, build_client
from glide.providers.errors import AllProvidersFailed, ProviderError

MODEL = "policy-1"


# ---------------------------------------------------------------------------------------------
# A fake LLM that answers from a TypeSafe-shaped client, through the prompt and the schema
# ---------------------------------------------------------------------------------------------


def read_prompt(user: str, schema: dict) -> tuple[object, dict]:
    """The state and the questions as the adapter's prompt states them, rebuilt as SDK objects.

    Reads only what the prompt carries: nothing is passed on the side. Each Choice's keys come from
    the schema's enum (the same order as the option lines) and its descriptions from those lines.
    """
    head, *blocks = user.split("\n\n")
    assert head.startswith("STATE\n"), "the state comes first"
    state = json.loads(head.removeprefix("STATE\n"))
    questions: dict = {}
    for block in blocks:
        lines = block.split("\n")
        found = re.fullmatch(r"Q (.+): (choice, top (\d+)|yes/no)", lines[0])
        assert found, f"unreadable question header {lines[0]!r}"
        name, prop = found[1], schema["properties"][found[1]]
        if found[2] == "yes/no":
            sides = {"yes: ": "true", "no: ": "false"}
            criteria = {sides[p]: line[len(p) :] for line in lines[1:] for p in sides if line.startswith(p)}
            prose = [line for line in lines[1:] if not line.startswith(tuple(sides))]
            questions[name] = Noul(instructions=" ".join(prose) or None, criteria=criteria or None)
            assert prop == {"type": "number", "minimum": 0, "maximum": 1}
            continue
        body = lines[1:]
        split = body.index("options:")
        enum = prop["properties"]["top"]["items"]["properties"]["option"]["enum"]
        option_lines = body[split + 1 :]
        assert len(option_lines) == len(enum), "one line per option in the schema's enum"
        criteria = {}
        for key, line in zip(enum, option_lines, strict=True):
            assert line == key or line.startswith(f"{key}: "), f"option line {line!r} is not for {key!r}"
            criteria[key] = None if line == key else line[len(key) + 2 :]
        assert prop["properties"]["top"]["maxItems"] == int(found[3])
        questions[name] = Choice(instructions=" ".join(body[:split]) or None, criteria=criteria)
    assert set(questions) == set(schema["properties"]), "the prompt and the schema ask the same questions"
    return state, questions


def spread(answer, enum: list[str], k: int) -> list[tuple[str, float]]:
    """The top options a model would list for this answer: the policy's own spread, or its choice at its confidence.

    A single-entry distribution is a policy saying only how sure it is. The remainder is spread over
    the next options, and the chosen one never drops below a third, which is as unsure as three
    options can express (a policy that wants less is stopping the run either way).
    """
    given = dict(getattr(answer, "probabilities", None) or {})
    if len(given) >= 2:
        ranked = sorted(given.items(), key=lambda kv: -kv[1])[:k]
    else:
        others = [option for option in enum if option != answer.choice][: k - 1]
        top = max(answer.confidence, 1 / k + 0.01) if others else 1.0
        ranked = [(answer.choice, top)] + [(option, (1 - top) / len(others)) for option in others]
    for option in enum:  # a spread over fewer than k options is padded with options that get 0
        if len(ranked) < k and option not in dict(ranked):
            ranked.append((option, 0.0))
    return ranked


class ClientBackedLLM:
    """`chat` answered by `client.system_one(state=, questions=)`, the way a classifier policy answers."""

    name, model = "policy", MODEL

    def __init__(self, client, usage: Usage | None = None):
        self.client = client
        self.usage = usage or Usage(100, 20)
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens=512, temperature=0.0, schema=None, logprobs=False, timeout=None):
        assert schema is not None, "the adapter always sends its schema"
        self.calls.append({"messages": list(messages), "schema": schema, "max_tokens": max_tokens})
        prompt = next(m["content"] for m in messages if m["role"] == "user")  # a retry appends a user turn after it
        state, questions = read_prompt(prompt, schema)
        answers = self.client.system_one(state=state, questions=questions).answers
        reply = {}
        for name, prop in schema["properties"].items():
            if prop.get("type") == "number":
                reply[name] = answers[name].noul
                continue
            top = prop["properties"]["top"]
            enum = top["items"]["properties"]["option"]["enum"]
            reply[name] = {"top": [{"option": o, "p": p} for o, p in spread(answers[name], enum, top["maxItems"])]}
        return ChatResult(
            text=json.dumps(reply),
            usage=self.usage,
            provider=f"policy:{MODEL}",
            model=MODEL,
            latency_s=0.0,
            finish_reason="stop",
        )


ORIGINAL_FAKE = world.FakeTypeSafe  # taken before `through_llm` swaps it
KEY = "sk-test-0123456789abcdef"
SPEC = ProviderSpec("jev", "typesafe", "https://jev.example.test", "JEV_API_KEY")


def typesafe_server(client):
    """A /v1/systemone server, as an httpx2 transport, answering from a TypeSafe-shaped client on the wire."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        asked = {
            name: Noul(instructions=q.get("instructions"), criteria=q.get("criteria"))
            if q["type"] == "noul"
            else Choice(instructions=q.get("instructions"), criteria=q["criteria"])
            for name, q in body["questions"].items()
        }
        answers = client.system_one(state=body["state"], questions=asked).answers
        wire = {
            name: {"type": "noul", "noul": a.noul}
            if hasattr(a, "noul")
            else {"type": "choice", "choice": a.choice, "confidence": a.confidence, "probabilities": dict(a.probabilities)}
            for name, a in answers.items()
        }
        return httpx2.Response(
            200, json={"model": "jev-latest", "usage": {"input_tokens": 50, "output_tokens": 5}, "answers": wire}
        )

    return httpx2.MockTransport(handler)


class Backed:
    """What `drive` builds in place of `FakeTypeSafe`: the same policy, answering through a real adapter.

    `variant` picks the adapter: a bare LLMClassifier; a ChainedClassifier over one; or a
    ChainedClassifier over the TypeSafe client talking to a mock /v1/systemone server.
    """

    last: Backed | None = None
    variant = "llm"

    def __init__(self, policy, noul: float = 0.95):
        self.inner = ORIGINAL_FAKE(policy, noul)
        self.llm = ClientBackedLLM(self.inner)
        if self.variant == "llm":
            self.classifier = LLMClassifier(self.llm)
        elif self.variant == "chained":
            self.classifier = ChainedClassifier.from_clients([LLMClassifier(self.llm, name="llm:policy")])
        else:
            slot = build_client(SPEC, "jev-latest", KEY, transport=typesafe_server(self.inner))
            self.classifier = ChainedClassifier.from_clients([slot])
        self.answered = lambda: self.llm.calls if self.variant != "typesafe" else self.inner.states
        Backed.last = self

    def __getattr__(self, name: str):
        """Anything else a scenario reads off `world.fake` (`states`, `asked`, ...) is the policy's own."""
        if name == "inner":  # not set yet: no recursion while the object is being built
            raise AttributeError(name)
        return getattr(self.inner, name)


def refuse_hosted_client():
    raise AssertionError("the loop built the hosted TypeSafe client instead of using the injected classifier")


def inject(monkeypatch, variant: str) -> type[Backed]:
    """Make every `drive` in the test run the loop with a classifier adapter injected, and the hosted client forbidden."""
    Backed.last = None
    monkeypatch.setattr(Backed, "variant", variant)
    real_run = runner.run

    def run_injected(cfg, ctx_factory):
        monkeypatch.setattr(runner, "TypeSafeClient", refuse_hosted_client)  # `drive` set its own just before
        return real_run(cfg, ctx_factory, classifier_factory=lambda: Backed.last.classifier)

    monkeypatch.setattr(world, "FakeTypeSafe", Backed)
    monkeypatch.setattr(world, "run", run_injected)
    return Backed


@pytest.fixture
def through_llm(monkeypatch):
    return inject(monkeypatch, "llm")


@pytest.fixture(params=["chained", "typesafe"])
def through_adapter(request, monkeypatch):
    return inject(monkeypatch, request.param)


def scenarios() -> list:
    """Every plain scenario of test_scenarios.py, in file order. One that is xfail there is left to it."""
    found = []
    for name, fn in inspect.getmembers(test_scenarios, inspect.isfunction):
        if not name.startswith("test_l") or fn.__module__ != test_scenarios.__name__:
            continue
        if tuple(inspect.signature(fn).parameters) != ("monkeypatch", "tmp_path"):
            continue
        if any(mark.name in ("xfail", "parametrize") for mark in getattr(fn, "pytestmark", [])):
            continue
        found.append(fn)
    return sorted(found, key=lambda fn: fn.__code__.co_firstlineno)


@pytest.mark.parametrize("scenario", scenarios(), ids=lambda fn: fn.__name__.removeprefix("test_"))
def test_the_existing_scenario_has_the_same_outcome_through_an_llm_classifier(scenario, through_llm, monkeypatch, tmp_path):
    scenario(monkeypatch, tmp_path)
    assert Backed.last is not None and Backed.last.answered(), "the injected classifier answered"


# Varied difficulty: one click, a press through accessibility, waiting, typing with the Noul check, an off-screen
# control, a site by name, a stop on low confidence, a long mixed run, and the writer steering the classifier.
THROUGH_EVERY_ADAPTER = (
    "test_l2_one_click_reaches_the_target",
    "test_l4_a_slow_page_needs_waiting",
    "test_l6_a_form_is_filled_and_submitted",
    "test_l13_an_offscreen_control_is_pressed",
    "test_l15_use_browser_from_another_app_opens_a_catalog_site",
    "test_l17_low_confidence_stops_the_run",
    "test_l19_typing_that_fails_verification_gives_the_field_its_old_value_back",
    "test_l36_a_long_run_mixes_everything",
    "test_l42_a_stop_between_two_good_moves_is_settled_by_the_writer_and_the_run_goes_on",
)


@pytest.mark.parametrize("name", THROUGH_EVERY_ADAPTER)
def test_scenarios_also_hold_through_a_chain_and_through_the_typesafe_client_on_the_wire(
    name, through_adapter, monkeypatch, tmp_path
):
    getattr(test_scenarios, name)(monkeypatch, tmp_path)
    assert Backed.last is not None and Backed.last.answered()


def test_the_scenarios_are_actually_the_existing_ones():
    names = {fn.__name__ for fn in scenarios()}
    assert len(names) >= 50 and set(THROUGH_EVERY_ADAPTER) <= names
    assert "test_l40_a_legitimate_repeat_that_leaves_the_visible_text_unchanged_is_taken_for_a_stall" not in names  # xfail there


# ---------------------------------------------------------------------------------------------
# Confidence: a split distribution stops the loop as it does with TypeSafe
# ---------------------------------------------------------------------------------------------


class Spread:
    """A classifier policy that answers every Choice with the first three options at these weights."""

    def __init__(self, weights: tuple[float, float, float]):
        self.weights = weights
        self.states: list[dict] = []

    def system_one(self, state, questions):
        if any(isinstance(q, Noul) for q in questions.values()):
            return SimpleNamespace(answers={name: NoulAnswer(noul=0.95) for name in questions})
        self.states.append(state)
        answers = {}
        for name, q in questions.items():
            options = list(q.criteria)[:3]
            dist = dict(zip(options, self.weights, strict=False))
            answers[name] = ChoiceAnswer(choice=options[0], confidence=self.weights[0], probabilities=dist)
        return SimpleNamespace(answers=answers)


HOME = [Page(name="home", items=["Home", "Tickets", "About"], url="https://example.com/")]


def drive_spread(weights, steps, monkeypatch, tmp_path):
    spread_client = Spread(weights)
    classifier = LLMClassifier(ClientBackedLLM(spread_client))
    w = World([Page(name="home", items=["Home", "Tickets", "About"], url="https://example.com/")])
    return w, drive_with(w, classifier, steps=steps, monkeypatch=monkeypatch, tmp_path=tmp_path)


def test_a_split_distribution_falls_below_the_floor_and_the_loop_stops_without_acting(monkeypatch, tmp_path):
    w, state = drive_spread((0.35, 0.33, 0.32), 5, monkeypatch, tmp_path)

    assert state.outcome == "low confidence"
    assert state.history == [] and w.log == [] and w.mouse == []
    log = (tmp_path / "run" / "run.log").read_text()
    assert "confidence 0.35 below 0.4; stopping" in log
    answers = json.loads((tmp_path / "run" / "step-001-answers.json").read_text())
    assert answers["kind_confidence"] == pytest.approx(0.35)
    assert answers["kind_probabilities"]["click_item"] == pytest.approx(0.35)
    assert sum(answers["kind_probabilities"].values()) == pytest.approx(1.0)


def test_the_same_policy_with_a_clear_leader_acts(monkeypatch, tmp_path):
    w, state = drive_spread((0.6, 0.3, 0.1), 1, monkeypatch, tmp_path)

    assert state.outcome == "step limit"  # it acted, and stopped only because the run was given one step
    assert len(state.history) == 1 and w.log == ["click:Home"]


def test_typesafe_stops_on_the_same_split_and_so_does_the_llm_classifier(monkeypatch, tmp_path):
    """The same decision, the same stop, with the policy answering directly and through the adapter."""
    direct = World(HOME)
    direct_state = world.drive(
        direct, scripted(("click_item", "Tickets", 0.3)), monkeypatch=monkeypatch, tmp_path=tmp_path / "direct"
    )
    assert direct_state.outcome == "low confidence"

    split = World(HOME)
    via_llm = drive_with(
        split,
        LLMClassifier(ClientBackedLLM(world.FakeTypeSafe(scripted(("click_item", "Tickets", 0.3))))),
        monkeypatch=monkeypatch,
        tmp_path=tmp_path / "llm",
    )
    assert (via_llm.outcome, via_llm.history, split.log) == (direct_state.outcome, direct_state.history, direct.log)


# ---------------------------------------------------------------------------------------------
# A Noul question, and what a bad answer does to a run
# ---------------------------------------------------------------------------------------------


def test_the_check_after_typing_is_a_noul_question_of_its_own_and_its_answer_decides(monkeypatch, tmp_path, through_llm):
    def submitted(w: World) -> str | None:
        return "results" if w.typed.get("Search") == "tour" else None

    pages = [
        Page(name="search", items=["Search", "Popular"], url="https://example.com/", field="Search", on={"enter": submitted}),
        Page(name="results", items=["First result"], url="https://example.com/results"),
    ]
    for noul, kept in ((0.95, True), (0.2, False)):
        w = World(pages)
        world.drive(
            w,
            scripted(("type_text", None), ("done", None)),
            monkeypatch=monkeypatch,
            tmp_path=tmp_path / str(noul),
            writer=FakeWriter(text="tour"),
            noul=noul,
        )
        verification = [c for c in Backed.last.llm.calls if "ok" in c["schema"]["properties"]]
        assert len(verification) == 1 and list(verification[0]["schema"]["properties"]) == ["ok"]
        assert verification[0]["schema"]["properties"]["ok"] == {"type": "number", "minimum": 0, "maximum": 1}
        assert "Did the typing succeed" in verification[0]["messages"][-1]["content"]
        assert w.typed["Search"] == ("tour" if kept else "")  # a value the model doubts is taken back out of the field


def test_an_unknown_key_is_retried_once_and_the_run_goes_on(monkeypatch, tmp_path):
    inner = world.FakeTypeSafe(scripted(("click_item", "Tickets"), ("done", None)))
    good = ClientBackedLLM(inner)

    class FirstReplyHasAnInvalidKey:
        """Spoils the reply to the first question it is asked, and gives the policy's own answer when asked again."""

        name, model = "policy", MODEL
        spoiled = False

        def chat(self, messages, **kw):
            retrying = any(m["role"] == "assistant" for m in messages)
            if retrying:
                return self.honest
            self.honest = good.chat(messages, **kw)  # the policy answers each decision once
            if self.spoiled:
                return self.honest
            self.spoiled = True
            text = json.loads(self.honest.text)
            text["kind"]["top"][0]["option"] = "zzz-not-an-option"
            return ChatResult(json.dumps(text), self.honest.usage, self.honest.provider, self.honest.model, 0.0)

    w = World(
        [
            Page(name="home", items=["Home", "Tickets"], url="https://example.com/", on={"click:Tickets": "tickets"}),
            Page(name="tickets", items=["Buy"], url="https://example.com/tickets"),
        ]
    )
    state = drive_with(w, LLMClassifier(FirstReplyHasAnInvalidKey()), monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert state.outcome == "done" and state.history == ["clicked 'Tickets'"] and w.page.name == "tickets"
    assert state.calls.usage[MODEL].input_tokens == 100 * (2 + 1)  # the retry took tokens too, and the run counts them


def test_a_model_that_keeps_answering_with_an_unknown_key_ends_the_run_as_a_provider_failure(monkeypatch, tmp_path):
    class Nonsense:
        name, model = "policy", MODEL

        def chat(self, messages, **kw):
            return ChatResult('{"kind": {"top": [{"option": "zzz", "p": 1}]}}', Usage(5, 5), f"policy:{MODEL}", MODEL, 0.0)

    w = World(HOME)
    writer = FakeWriter()
    state = drive_with(w, LLMClassifier(Nonsense()), monkeypatch=monkeypatch, tmp_path=tmp_path, writer=writer)

    assert state.outcome == "provider failure" and w.log == [] and writer.requests == []
    log = (tmp_path / "run" / "run.log").read_text()
    assert f"policy:{MODEL} [content]" in log and "unusable classifier answer twice" in log


# ---------------------------------------------------------------------------------------------
# A provider failure is an outcome, not a crash
# ---------------------------------------------------------------------------------------------


class Flaky:
    """A classifier slot that answers from a policy for its first `good` calls, then fails with `error`."""

    def __init__(self, name: str, policy, good: int, error: ProviderError):
        self.name, self.model = name, "m"
        self.inner = world.FakeTypeSafe(policy)
        self.good, self.error = good, error
        self.calls = 0
        self.closed = False

    def system_one(self, state, questions, *, model=None, **kw):
        self.calls += 1
        if self.calls > self.good:
            raise self.error
        return self.inner.system_one(state, questions)

    def close(self):
        self.closed = True


TWO_STEPS = [
    Page(name="home", items=["Home", "Tickets"], url="https://example.com/", on={"click:Tickets": "tickets"}),
    Page(name="tickets", items=["Buy", "Terms"], url="https://example.com/tickets", on={"click:Buy": "checkout"}),
    Page(name="checkout", items=["Pay now"], url="https://example.com/checkout"),
]


def clicks():
    """A fresh policy per use: `scripted` counts the decisions it has answered."""
    return scripted(("click_item", "Tickets"), ("click_item", "Buy"), ("done", None))


def test_every_provider_failing_ends_the_run_with_an_outcome_a_log_line_and_a_run_json(monkeypatch, tmp_path):
    a = Flaky(
        "jev:jev-latest",
        clicks(),
        1,
        ProviderError("jev:jev-latest answered 429: slow down", kind="rate_limit", provider="jev:jev-latest"),
    )
    b = Flaky(
        "llm:policy",
        clicks(),
        0,
        ProviderError("llm:policy gave an unusable classifier answer twice", kind="content", provider="llm:policy"),
    )
    classifier = ChainedClassifier.from_clients([a, b])
    w, writer = World(TWO_STEPS), FakeWriter()

    state = drive_with(w, classifier, monkeypatch=monkeypatch, tmp_path=tmp_path, writer=writer)

    assert state.outcome == "provider failure"
    assert state.history == ["clicked 'Tickets'"] and w.page.name == "tickets"  # the work done before the failure stands
    assert writer.requests == []  # the writer is not asked to answer for a classifier that never stopped
    assert (a.closed, b.closed) == (True, True)  # and the classifier is closed on the way out
    log = (tmp_path / "run" / "run.log").read_text()
    assert "provider failure after 1 actions: every classifier provider failed:" in log
    assert "jev:jev-latest [rate_limit]" in log and "llm:policy [content]" in log
    summary = json.loads((tmp_path / "run" / "run.json").read_text())
    assert summary["outcome"] == "provider failure" and summary["steps_taken"] == 1
    assert summary["failure"] == state.failure and "slow down" in state.failure
    assert summary["calls"]["classifier"]["calls"] == 2  # one answered, one failed: a failed request still counts


def test_an_error_that_no_other_provider_could_fix_is_named_as_it_is(monkeypatch, tmp_path):
    only = Flaky(
        "jev:jev-latest", clicks(), 0, ProviderError("refused the request", kind="bad_request", provider="jev:jev-latest")
    )
    state = drive_with(World(TWO_STEPS), ChainedClassifier.from_clients([only]), monkeypatch=monkeypatch, tmp_path=tmp_path)
    assert state.outcome == "provider failure" and state.failure == "jev:jev-latest [bad_request] refused the request"
    log = (tmp_path / "run" / "run.log").read_text()
    assert "provider failure after 0 actions: jev:jev-latest [bad_request] refused the request" in log


def test_the_chain_fails_over_inside_the_loop_so_one_good_slot_is_enough(monkeypatch, tmp_path):
    down = Flaky("down", clicks(), 0, ProviderError("down", kind="server", provider="down"))
    up = Flaky("up", clicks(), 99, ProviderError("never", kind="server"))
    classifier = ChainedClassifier.from_clients([down, up])
    w = World(TWO_STEPS)

    state = drive_with(w, classifier, monkeypatch=monkeypatch, tmp_path=tmp_path)

    assert state.outcome == "done" and w.page.name == "checkout"
    assert classifier.chain.events[0].kind == "server" and classifier.last_slot == "up"


def test_a_bug_in_the_classifier_still_crashes_the_run_as_before(monkeypatch, tmp_path):
    class Broken:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def system_one(self, **request):
            raise RuntimeError("a bug, not a provider")

    with pytest.raises(RuntimeError, match="a bug"):
        drive_with(World(TWO_STEPS), Broken(), monkeypatch=monkeypatch, tmp_path=tmp_path)
    summary = json.loads((tmp_path / "run" / "run.json").read_text())
    assert summary["outcome"] == "crashed" and summary["failure"] is None


def test_without_a_factory_the_loop_uses_the_hosted_client_as_it_always_did(monkeypatch, tmp_path):
    used = []

    class Hosted(world.FakeTypeSafe):
        def __init__(self):
            super().__init__(scripted(("done", None)))
            used.append(self)

    w = World([Page(name="done", items=["Order 4821"], url="https://example.com/done")])
    w.install(monkeypatch)
    monkeypatch.setattr(runner, "TypeSafeClient", Hosted)
    cfg = runner.RunConfig(goal="g", out=tmp_path / "run", act=True, steps=3, delay=0)
    state = runner.run(
        cfg,
        lambda typesafe, history: Context(
            goal="g", browser="Google Chrome", email=None, typesafe=typesafe, writer=FakeWriter(), history=history
        ),
    )
    assert state.outcome == "done" and len(used) == 1


def drive_with(w: World, classifier, *, steps=20, monkeypatch, tmp_path, writer=None):
    """`world.drive`, but handing the classifier to `runner.run(classifier_factory=...)` as the caller would."""
    w.install(monkeypatch)
    monkeypatch.setattr(runner, "TypeSafeClient", refuse_hosted_client)
    cfg = runner.RunConfig(goal="do the thing", out=tmp_path / "run", act=True, steps=steps, delay=0)
    client = writer or FakeWriter()
    return runner.run(
        cfg,
        lambda typesafe, history: Context(
            goal="do the thing", browser="Google Chrome", email=None, typesafe=typesafe, writer=client, history=history
        ),
        classifier_factory=lambda: classifier,
    )


# ---------------------------------------------------------------------------------------------
# The browser backend takes any of them as its client
# ---------------------------------------------------------------------------------------------


def llm(inner) -> LLMClassifier:
    return LLMClassifier(ClientBackedLLM(inner))


def test_the_browser_loop_clicks_the_element_the_llm_classifier_chose(tmp_path):
    page = browser_loop.FakeBrowser(browser_loop.login_page())
    result, folder = browser_loop.run(page, llm(browser_loop.FakeTypeSafe(("click", "2"))), tmp_path, steps=1)

    assert result.steps[0].action == "click" and result.steps[0].confidence == pytest.approx(0.9)
    # `clicking` needs the element answer to be a real ChoiceAnswer; otherwise the step reads "click None -> element missing".
    assert result.steps[0].detail.startswith("click 2") and "missing" not in result.steps[0].detail
    answers = json.loads((folder.root / "step-01-answers.json").read_text())
    assert answers["kind"]["type"] == "choice" and answers["satisfied"]["type"] == "noul"  # serialize_answers knows both
    assert answers["element"]["choice"] == "2" and sum(answers["kind"]["probabilities"].values()) == pytest.approx(1.0)


def test_the_browser_loop_verifies_typing_with_a_noul_question_through_the_adapter(tmp_path):
    writer = browser_loop.FakeWriter({"fill": True, "text": "alice", "reason": "the goal"})

    page = browser_loop.FakeBrowser(browser_loop.login_page(), values={0: "alice"})
    result, _ = browser_loop.run(
        page, llm(browser_loop.FakeTypeSafe(("type_text", "0"))), tmp_path / "ok", writer=writer, steps=1
    )
    assert page.typed == ["alice"] and "verify 0.95" in result.steps[0].detail

    class Unsure(browser_loop.FakeTypeSafe):
        def system_one(self, *, state, questions, model=None):
            if set(questions) == {"ok"}:
                return SimpleNamespace(answers={"ok": NoulAnswer(noul=0.2)})
            return super().system_one(state=state, questions=questions, model=model)

    page = browser_loop.FakeBrowser(browser_loop.login_page(), values={0: "alice"})
    result, _ = browser_loop.run(page, llm(Unsure(("type_text", "0"))), tmp_path / "doubt", writer=writer, steps=1)
    assert "verify 0.20, cleared" in result.steps[0].detail


def test_the_browser_loop_ends_on_a_satisfied_noul_from_the_adapter(tmp_path):
    class Satisfied(browser_loop.FakeTypeSafe):
        def system_one(self, *, state, questions, model=None):
            reply = super().system_one(state=state, questions=questions, model=model)
            if "satisfied" in reply.answers:
                reply.answers["satisfied"] = NoulAnswer(noul=0.9)
            return reply

    page = browser_loop.FakeBrowser(browser_loop.listing_page())
    result, _ = browser_loop.run(page, llm(Satisfied(("wait", None))), tmp_path, steps=3)
    assert result.outcome == "done" and len(result.steps) == 1


def test_the_browser_decision_is_the_same_through_the_adapter_and_through_typesafe_directly():
    page = perceive(browser_loop.FakeBrowser(browser_loop.login_page()))
    direct = browser_decide(browser_loop.FakeTypeSafe(("click", "2")), "sign in", page, [])
    adapted = browser_decide(llm(browser_loop.FakeTypeSafe(("click", "2"))), "sign in", page, [])

    assert (adapted.kind.choice, adapted.element.choice, adapted.satisfied.noul) == (
        direct.kind.choice,
        direct.element.choice,
        0.0,
    )
    assert adapted.state == direct.state and adapted.clicking and adapted.chosen_element == 2
    assert browser_verify(llm(browser_loop.FakeTypeSafe()), "g", "Username", "alice", "alice") == pytest.approx(0.95)


def test_the_browser_loop_through_a_chain_of_the_llm_and_typesafe(tmp_path):
    """The first slot's model is useless (it answers nothing the schema allows), so the TypeSafe slot answers."""

    class Useless:
        name, model = "policy", MODEL

        def chat(self, messages, **kw):
            return ChatResult("no", Usage(), f"policy:{MODEL}", MODEL, 0.0)

    inner = browser_loop.FakeTypeSafe(("click", "2"))
    typesafe = build_client(SPEC, "jev-latest", KEY, transport=browser_server(inner))
    classifier = ChainedClassifier.from_clients([LLMClassifier(Useless(), name="llm:useless"), typesafe])

    page = browser_loop.FakeBrowser(browser_loop.login_page())
    with classifier:
        result, _ = browser_loop.run(page, classifier, tmp_path, steps=1)

    assert result.steps[0].action == "click" and classifier.last_slot == "jev:jev-latest"
    assert [(e.from_slot, e.kind) for e in classifier.chain.events] == [("llm:useless", "content")]


def browser_server(inner) -> httpx2.MockTransport:
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        asked = {
            name: Noul(instructions=q.get("instructions"), criteria=q.get("criteria"))
            if q["type"] == "noul"
            else Choice(instructions=q.get("instructions"), criteria=q["criteria"])
            for name, q in body["questions"].items()
        }
        answers = inner.system_one(state=body["state"], questions=asked, model=body.get("model")).answers
        wire = {
            name: {"type": "noul", "noul": a.noul}
            if isinstance(a, NoulAnswer)
            else {"type": "choice", "choice": a.choice, "confidence": a.confidence, "probabilities": dict(a.probabilities)}
            for name, a in answers.items()
        }
        return httpx2.Response(
            200, json={"model": "jev-latest", "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": wire}
        )

    return httpx2.MockTransport(handler)


def test_an_all_providers_failed_error_is_a_provider_error_the_runner_catches():
    assert issubclass(AllProvidersFailed, ProviderError)
