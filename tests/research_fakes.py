"""Offline stand-ins for everything the research supervisor talks to: the model, the browser and the engine's loop.

`Run.execute` plays the engine's part and nothing more. It calls `Supervisor.advance` with a scripted model, a fake
browser that serves canned pages, and a fake planner, applies each planned step as "verified", and repeats. The
executor, the effect ledger and Jev are not here (the engine's own tests own them); what is here is exactly what
the supervisor decides, shows and refuses.
"""

import json
import re
from types import SimpleNamespace

from glide.computer.execution.contracts import Element, InvalidAction, Observation
from glide.computer.execution.research import Supervisor, Tools
from glide.computer.generation import GenerationResult
from glide.computer.models import Abort
from glide.computer.writer import WriterError
from glide.providers.base import ChatResult, Usage

SEARCH = "https://search.example.test"
A = "https://reviews.example.test/place"
B = "https://guide.example.test/place"
QUOTE_A = "Customers praise the food but report long waits."
QUOTE_B = "Service is friendly and the menu has vegetarian options."
URL = re.compile(r"https?://[^\s\"'<>]+")


def decision(kind, **kw):
    return {"kind": kind, "goal": "", "question": "", "reason": "", "claims": [], "limitations": "", **kw}


def claim(statement, source="s1", quote=QUOTE_A):
    return {"text": statement, "citations": [{"source_id": source, "quote": quote}]}


def approved():
    return {"supported": True, "complete": True, "feedback": ""}


class Reasoner:
    """A scripted model provider. Each call consumes the next reply; an exception in the script is raised."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []

    def generate(self, request, cancel=None):
        self.requests.append(request)
        assert self.script, f"unscripted model call for role {request.role}"
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return GenerationResult(json.dumps(item), "fixture")

    def packets(self, role=None):
        return [json.loads(r.text) for r in self.requests if role is None or r.role == role]

    def roles(self):
        return [r.role for r in self.requests]


class Facade:
    """An LLM facade of glide.toml (fast, smart, planner or research) that answers from a script and counts its calls."""

    def __init__(self, script=()):
        self.script, self.calls = list(script), []

    def chat(self, messages, *, max_tokens=512, temperature=0.0, schema=None, timeout=None):
        self.calls.append({"messages": list(messages), "schema": schema, "timeout": timeout})
        assert self.script, "this facade was not expected to be called"
        return ChatResult(
            text=json.dumps(self.script.pop(0)),
            usage=Usage(),
            provider="slot:model",
            model="model",
            latency_s=0.0,
            finish_reason="stop",
        )


class Pages:
    """A browser with a few pages. `read_page` answers for the current one, in the shape the real adapters return."""

    transport = "fixture"

    def __init__(self, pages=None, url=""):
        self.pages = pages if pages is not None else {}
        self.url = url
        self.read_log = []

    def page(self, url, text, links=(), title="Observed page"):
        self.pages[url] = {"title": title, "text": text, "links": [{"url": u, "title": t} for u, t in links]}
        return self

    def read_page(self):
        self.read_log.append(self.url)
        page = self.pages.get(self.url, {"title": "Empty", "text": "", "links": []})
        return {"url": self.url, "truncated": False, **page}

    def inspect(self, elements=None):
        if not self.url:
            return Observation("browser", "", capabilities={"inspect"})
        return Observation(
            "browser", f"tab1:{self.url}", self.url, "tab1", {"tab1": self.url}, elements or {}, capabilities={"inspect"}
        )


class Planner:
    """The browser planner: each call returns the next scripted (steps, question) and remembers what it was told."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def __call__(self, writer, goal, observed, *, reply="", context=None):
        self.calls.append({"goal": goal, "reply": reply, "context": context})
        assert self.results, "unscripted planner call"
        result = self.results.pop(0)
        return result if isinstance(result, tuple) else (result, "")


def literal_urls(text):
    return [url.rstrip(".,;!?") for url in URL.findall(text)]


class Run:
    """The engine's loop around one Supervisor."""

    def __init__(
        self,
        goal,
        writer,
        pages=None,
        *,
        route="research",
        planner=None,
        ask=None,
        budget=24,
        ready=None,
        search_action=lambda action, observed: False,
        search_url=SEARCH,
    ):
        self.pages = pages if pages is not None else Pages()
        self.writer, self.ask, self.ready = writer, ask, ready
        self.waited, self.stages, self.steps, self.applied = [], [], [], []
        self.planner = planner or Planner()
        self.tools = Tools(self.planner, self._wait_ready, literal_urls, search_action)
        self.supervisor = Supervisor(goal, route, budget, tools=self.tools, search_url=search_url)
        self.route, self.error = route, None

    def _wait_ready(self, backend, observed, timeout):
        self.waited.append((backend, observed.url, timeout))
        if self.ready:
            self.ready()

    def measured(self, name, function, *args, **kwargs):
        self.stages.append(name)
        return function(*args, **kwargs)

    def apply(self, step):
        """What the executor does once it has verified a step."""
        self.applied.append(step)
        if step.effect in {"url", "tab_created"}:
            self.pages.url = step.value

    def execute(self, limit=40, timeout=10.0):
        """Run to an answer or a stop. Returns the answer, or None with `error` (and `aborted`) set."""
        supervisor, reason = self.supervisor, self.route == "reason"
        ctx = SimpleNamespace(writer=self.writer, ask=self.ask)
        backend = None if reason else self.pages
        self.aborted = False
        try:
            for _ in range(limit):
                observed = Observation("", "", capabilities=set(), browser_front=False) if reason else self.pages.inspect()
                ledger = SimpleNamespace(summary=lambda: {"verified": len(self.applied)})
                batch = supervisor.advance(ctx, backend, observed, self.steps, ledger, self.measured, timeout)
                if supervisor.answer:
                    return supervisor.answer
                for step in batch:
                    self.apply(step)
                self.steps.extend(batch)
            self.error = "step limit"
        except Abort as stop:
            self.aborted, self.error = True, str(stop)
        except (InvalidAction, WriterError) as stop:
            self.error = str(stop)
        return None


def element(id_, label, **kw):
    return Element(id_, label, "button", **kw)
