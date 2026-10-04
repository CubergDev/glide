"""A stateful fake adapter, scripted models and real production execution contracts."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import asdict, replace
from types import SimpleNamespace
from urllib.parse import urlencode

from glide.computer.actions import Context
from glide.computer.control import RunControl
from glide.computer.execution import providers, research
from glide.computer.execution.contracts import EFFECTS, Container, Element, Milestone, Observation, validate
from glide.computer.generation import GenerationResult
from glide.computer.runner import RunConfig, run


class Computer:
    passive_inspection = True

    def __init__(self):
        self.state = Observation(
            "Editor",
            "editor:1",
            "https://example.org",
            "initial",
            {"initial": "https://example.org"},
            {"name": Element("name", "Name", "field", "", True), "save": Element("save", "Save", "button")},
            {"feed": Container("feed", "Feed", 0, 5000), "sidebar": Container("sidebar", "Sidebar", 0, 1000)},
            "name",
            {"inspect", "click", "type", "key", "navigate", "tab_create", "tab_switch", "tab_close", "scroll"},
        )
        self.actions = []
        self.reads = 0
        self.closed = False
        self.on_execute = None
        self.on_inspect = None
        self.next_id = 0
        self.submissions = {}  # Observed form field -> (destination URL, query parameter).

    def inspect(self):
        self.reads += 1
        if self.on_inspect:
            self.on_inspect(self)
        return copy.deepcopy(self.state)

    def execute(self, action, observed):
        validate(action, self.state)
        self.actions.append(action)
        if self.on_execute:
            return self.on_execute(self, action)
        if action.kind == "navigate":
            self.state.url = action.value
            self.state.tabs[self.state.active_tab] = action.value
        elif action.kind == "tab_create":
            self.next_id += 1
            identity = str(self.next_id)
            self.state.tabs[identity] = action.value
            self.state.active_tab = identity
            self.state.url = action.value
            return identity
        elif action.kind == "tab_close":
            self.state.tabs.pop(action.target)
            if self.state.active_tab == action.target:
                self.state.active_tab = next(iter(self.state.tabs), "")
                self.state.url = self.state.tabs.get(self.state.active_tab, "")
        elif action.kind == "tab_switch":
            self.state.active_tab = action.target
            self.state.url = self.state.tabs[action.target]
        elif action.kind == "type":
            el = self.state.elements[action.target]
            self.state.elements[action.target] = replace(el, value=action.value)
            self.state.focus = action.target
        elif action.kind == "click":
            if action.target == "save":
                self.state.elements["saved"] = Element("saved", "Saved", "status")
            elif action.target == "dismiss":
                self.state.elements.pop("dialog", None)
            else:
                self.state.focus = action.target
        elif action.kind == "scroll":
            c = self.state.containers[action.target]
            self.state.containers[action.target] = replace(
                c, position=min(c.maximum, max(0, c.position + (100 if action.value == "down" else -100)))
            )
        elif action.kind == "key":
            if action.value == "return" and self.state.focus in self.submissions:
                destination, parameter = self.submissions[self.state.focus]
                field = self.state.elements[self.state.focus]
                self.state.url = destination + "?" + urlencode({parameter: field.value, "source": "offline"})
                self.state.tabs[self.state.active_tab] = self.state.url
            else:
                self.state.focus = "save" if self.state.focus == "name" else "name"
        return ""

    def close(self):
        self.closed = True


class Jev:
    def __init__(self, route="plan", selection=None, bindings=None):
        self.route = route
        self.selection = selection
        self.requests = []
        self.questions = []
        self.bindings = bindings or {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def system_one(self, state, questions):
        self.requests.append(state)
        self.questions.append(questions)
        criteria = questions["action"].criteria
        if state.get("task_scope"):
            choice = self.route if self.route in {"research", "reason", "clarify"} else "execute"
        elif "milestone" not in state:
            choice = self.route
            if choice not in criteria:
                choice = next(
                    k
                    for k, text in criteria.items()
                    if k not in {"plan", "clarify"} and json.loads(text)["action"]["kind"] == choice
                )
        elif self.selection:
            choice = self.selection(state, criteria)
        else:
            step = state["milestone"]
            kind = {
                "url": "navigate",
                "tab_created": "tab_create",
                "scroll": "scroll",
                "field_value": "type",
                "tab_closed": "tab_close",
                "tab_active": "tab_switch",
                "url_query": "key",
            }.get(step["effect"], "click")
            options = [(k, json.loads(v)) for k, v in criteria.items() if k != "replan"]
            wanted = (
                next(
                    (
                        c["id"]
                        for c in state["observation"]["containers"]
                        if c["id"] == step["target"] or c["label"].casefold() == step["target"].casefold()
                    ),
                    "",
                )
                if step["effect"] == "scroll"
                else "save"
            )
            choice = next(
                (k for k, a in options if a["kind"] == kind and (kind not in {"click", "scroll"} or a["target"] == wanted)),
                "replan",
            )
        answers = {"action": SimpleNamespace(choice=choice, confidence=0.99)}
        defaults = {
            "scope": "simple",
            "quantity": "1",
            "modifiers": "[]",
            "key": "return" if state.get("milestone", {}).get("effect") == "url_query" else "tab",
        }
        for name in questions:
            if name != "action":
                answers[name] = SimpleNamespace(choice=self.bindings.get(name, defaults[name]), confidence=0.99)
        return SimpleNamespace(model="jev-offline", answers=answers)


class Reasoner:
    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def generate(self, request, cancel=None):
        self.requests.append(request)
        reply = self.replies[min(len(self.requests) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, GenerationResult):
            return reply
        return GenerationResult(json.dumps(reply), "reasoner-offline")


class FakeSupervisor:
    """A scripted stand-in for `research.Supervisor`, which `drive(..., supervisor=...)` puts in its place.

    It has what the engine relies on from the real one and nothing else: `advance` returns the next browser batch (or
    sets `answer`), and the attributes the recovery path reads. `script` is a list of `("browse", [Milestone, ...])`,
    `("answer", Answer)` or `("raise", exception)`. Calling it is the engine constructing the supervisor; the
    arguments it was given are kept in `built`.
    """

    def __init__(self, script=()):
        self.script = list(script)
        self.goal = self.route = None
        self.built = {}
        self.replies, self.questions, self.sources = [], 0, []
        self.batch, self.batch_ids, self.browser_goal = 0, set(), ""
        self.answer = None
        self.advances = []

    def __call__(self, goal, route, budget, *, tools, search_url=""):
        self.goal, self.route = goal, route
        self.built = {"budget": budget, "tools": tools, "search_url": search_url}
        return self

    def advance(self, ctx, backend, observed, steps, ledger, measured, readiness_timeout):
        self.advances.append({"backend": backend, "owner": observed.owner, "progress": ledger.summary()})
        kind, payload = self.script.pop(0)
        if kind == "raise":
            raise payload
        if kind == "answer":
            self.answer = payload
            return []
        self.batch += 1
        batch = [replace(step, id=f"research{self.batch}_{step.id}") for step in payload]
        self.browser_goal, self.batch_ids = "browse", {step.id for step in batch}
        return batch

    def planning_context(self, observed):
        return {"allowed_effects": sorted(EFFECTS), "clarifications": self.replies}

    def ask(self, ctx, question):
        raise AssertionError("the fake supervisor never asks")

    def validate_plan(self, steps, observed):
        pass

    def validate_action(self, action, observed):
        pass

    def summary(self):
        return {"route": self.route, "model_calls": len(self.advances)}


class FakeClock:
    """Time that moves only when something sleeps, patched over `progress.time` (the bounded waits' one clock)."""

    def __init__(self, monkeypatch):
        from types import SimpleNamespace

        from glide.computer.execution import progress

        self.now = 0.0
        self.sleeps = []
        monkeypatch.setattr(progress, "time", SimpleNamespace(monotonic=lambda: self.now, sleep=self.sleep))

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def response(*steps, question="", unsupported=()):
    return {"question": question, "steps": [asdict(s) for s in steps], "unsupported": list(unsupported)}


def named(goal, reasoner):
    """`goal`, and every address the scripted plans open, as a request that names them would. A plan with no
    supervisor may only open an address its goal, the page or the settings offer, and most tests care about the
    plan's other properties, not about where it goes."""
    addresses = dict.fromkeys(re.findall(r"https?://[^\s\"\\]+", json.dumps(getattr(reasoner, "replies", []), default=str)))
    return " ".join([goal, *addresses])


def default_goal(reasoner):
    return named("Task", reasoner)


def give_backend(monkeypatch, computer, calls=None):
    """The engine's one way to a backend is `providers.make_backend`: hand it `computer`. The calls it gets are
    appended to `calls` (a new list when none is given), which is returned."""
    calls = [] if calls is None else calls

    def make_backend(browser="", *, act=False, on_switch=None):
        calls.append({"browser": browser, "act": act, "on_switch": on_switch})
        return computer

    monkeypatch.setattr(providers, "make_backend", make_backend)
    return calls


def drive(monkeypatch, tmp_path, computer, reasoner, jev=None, control=None, supervisor=None, **kwargs):
    """Run the structured engine on the fake computer, with Jev as the run's classifier (D7: `classifier_factory`)
    and the scripted reasoner as the writer. Only the backend door is replaced; the supervisor is the caller's, or
    the real one."""
    jev = jev or Jev()
    give_backend(monkeypatch, computer, kwargs.pop("calls", None))
    if supervisor is not None:
        monkeypatch.setattr(research, "Supervisor", supervisor)
    ask = kwargs.pop("ask", None)
    readiness = kwargs.pop("readiness_timeout", 0)  # "default" leaves it to RunConfig's own default
    cfg = RunConfig(
        kwargs.pop("goal", None) or default_goal(reasoner),
        tmp_path,
        act=kwargs.pop("act", True),
        engine="structured",
        execution_browser="Brave Browser",
        **({} if readiness == "default" else {"readiness_timeout": readiness}),
        **kwargs,
    )
    return run(
        cfg,
        lambda client, history: Context(cfg.goal, "Brave Browser", None, client, reasoner, history, ask=ask),
        classifier_factory=lambda: jev,
        control=control or RunControl("offline"),
    )


SCENARIOS = {
    "open ten Google tabs": [
        Milestone("tabs", "Create ten new Google tabs", "tab_created", value="https://google.com", quantity=10)
    ],
    "navigate to X and scroll": [
        Milestone("nav", "Reach X", "url", value="https://x.com"),
        Milestone("scroll", "Scroll the feed", "scroll", target="Feed", value="down"),
    ],
    "edit and save a desktop form": [
        Milestone("name", "Set the name", "field_value", target="Name", value="A different document"),
        Milestone("save", "Save and verify", "element_present", target="Saved"),
    ],
    "open four unrelated documentation tabs": [
        Milestone("docs", "Create documentation tabs", "tab_created", value="https://docs.python.org/3/", quantity=4)
    ],
    "scroll a feed three times": [Milestone("feed", "Reveal more items", "scroll", target="Feed", value="down", quantity=3)],
    "navigate to unrelated documentation": [Milestone("nav", "Reach documentation", "url", value="https://docs.python.org/3/")],
}
