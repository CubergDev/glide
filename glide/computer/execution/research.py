"""The research supervisor: the frontier model plans, the browser operates, and the answer cites what was read.

Three routes of work meet here (HANDOFF section 5):

- research: the model chooses one bounded browser outcome at a time (`browse`), the page is then read (`read`),
  and the final answer cites quotes from pages that were actually read in this run. A second call checks the
  answer against that evidence before it is spoken.
- reason: no browser at all. One response from the same research role, validated in code.
- every browser action is still selected and effect-verified by the executor, never by this module.

Trust. Page text, links and titles are untrusted data. They reach the model only inside the `evidence` field of its
packet (marked `origin: page`), never in its instructions, and nothing in them can widen what the supervisor
allows: a destination must come from the user's own words, the configured search page, or a page that was read; an
effect must be one of READ_EFFECTS; a citation must quote a source that was read in this run.

The executor's helpers (planning, readiness, URL extraction, search-action check) are passed in as `Tools`, so this
module depends only on the contracts and the writer. The research model is the `research` role of glide.toml: no
model id appears here.
"""

import re
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace

from .. import diagnostics
from ..control import checkpoint, current_control
from ..writer import Answer, compose_research, review_research
from .contracts import InvalidAction, safe_url
from .reading import page_record

MAX_SOURCES = 8  # pages read per run, besides the user's own request
MAX_MODEL_CALLS = 32  # the most any budget may allow
MAX_QUESTIONS = 3
READ_EFFECTS = {"url", "tab_created", "tab_active", "scroll", "query_submitted", "disclosure_expanded"}
READ_ACTIONS = {"inspect", "navigate", "tab_create", "tab_switch", "scroll"}
STAGES = {"browse", "read", "answer", "clarify", "blocked"}
ADDRESS = re.compile(r"https?://[^\s\"'<>)\]]+", re.IGNORECASE)

REQUEST = "request"  # the id of the user's own words, the one source that is not a page


@dataclass(frozen=True)
class Tools:
    """The executor's helpers, injected so research does not import the engine's modules.

    plan(writer, goal, observed, *, reply, context) -> (steps, question)   a bounded browser plan
    wait_ready(backend, observed, timeout)                                 block until the new document is readable
    urls_in(text) -> list[str]                                             literal URLs the user wrote
    is_search_action(action, observed) -> bool                             the code-owned search-form binding
    """

    plan: Callable
    wait_ready: Callable
    urls_in: Callable
    is_search_action: Callable


def _text(value, limit, *, required=False):
    if not isinstance(value, str) or len(value) > limit or "\x00" in value or (required and not value.strip()):
        raise InvalidAction("The research model returned an invalid response field")
    return value.strip()


def _flat(value):
    return " ".join(value.split())


class Supervisor:
    def __init__(self, goal, route, budget, *, tools, search_url=""):
        self.goal, self.route, self.tools, self.search_url = goal, route, tools, search_url
        self.budget = min(budget, MAX_MODEL_CALLS)
        self.calls = 0
        self.sources = [{"id": REQUEST, "origin": "request", "title": "User request", "url": "", "text": goal, "links": []}]
        self.completed = []
        self.replies = []
        self.feedback = ""
        self.browser_goal = ""
        self.batch_ids = set()
        self.batch = 0
        self.answer = None
        self.questions = 0

    # -- the engine's loop -------------------------------------------------------------------------------------

    def advance(self, ctx, backend, observed, steps, ledger, measured, readiness_timeout):
        """The next stage once every planned step is verified: read a page, ask, stop, answer, or a new browser batch."""
        self.mark_completed(steps)
        transport = getattr(backend, "transport", "unknown") if backend else "none"
        packet = self.packet(observed, ledger.summary(), transport)
        decision = measured("research_supervisor", self.next, ctx.writer, packet)
        kind = decision["kind"]
        if self.route == "reason" and kind in {"read", "browse"}:
            raise InvalidAction("This reasoning task needs external evidence; start a research task to collect it.")
        if kind == "read":
            measured("readiness", self.tools.wait_ready, backend, observed, readiness_timeout)
            measured("page_reading", self.collect, backend)
        elif kind == "clarify":
            self.ask(ctx, decision["question"])
        elif kind == "blocked":
            raise InvalidAction(decision["reason"] or "I couldn't collect enough evidence to complete this task.")
        elif kind == "answer":
            stage = "answer_validation" if self.route == "reason" else "research_verification"
            measured(stage, self.finish, ctx.writer, decision, packet)
        else:
            return self.browse(ctx, observed, decision["goal"], measured)
        return []

    def browse(self, ctx, observed, goal, measured):
        reply = ""
        while True:
            batch, question = measured(
                "planning",
                self.tools.plan,
                ctx.writer,
                goal,
                observed,
                reply=reply,
                context=self.planning_context(observed),
            )
            if not question:
                break
            reply = self.ask(ctx, question)
        self.validate_plan(batch, observed)
        self.batch += 1
        batch = [replace(step, id=f"research{self.batch}_{step.id}") for step in batch]
        self.browser_goal, self.batch_ids = goal, {s.id for s in batch}
        return batch

    def ask(self, ctx, question):
        if not ctx.ask or self.questions >= MAX_QUESTIONS:
            raise InvalidAction(question)
        self.questions += 1
        if control := current_control():
            control.event("question", question, question_id=str(uuid.uuid4()))
        reply = ctx.ask(question)
        checkpoint()
        reply = _text(reply, 2000, required=True)
        self.replies.append(question + "\nUser: " + reply)
        return reply

    # -- what the model is shown -------------------------------------------------------------------------------

    def planning_context(self, observed):
        """Data for the browser planner: grounded destinations only (never page text, titles or quotes)."""
        return {
            "original_goal": self.goal,
            "configured_search_url": self.search_url,
            "observed_urls": sorted(self.allowed_urls(observed)),
            "allowed_effects": sorted(READ_EFFECTS),
            "clarifications": list(self.replies),
        }

    def packet(self, observed, progress, transport):
        return {
            "original_goal": self.goal,
            "route": self.route,
            "browser_session": {"provider": transport, "owner": observed.owner, "active_tab": observed.active_tab},
            "observation": observed.packet(),
            "verified_progress": progress,
            "completed_browser_tasks": self.completed,
            "evidence": self.sources,
            "clarifications": list(self.replies),
            "feedback": self.feedback,
            "configured_search_url": self.search_url,
            "model_calls_remaining": self.budget - self.calls,
            "page_sources_remaining": MAX_SOURCES - (len(self.sources) - 1),
        }

    # -- the model's decision ----------------------------------------------------------------------------------

    def generate(self, writer, function, packet):
        if self.calls >= self.budget:
            raise InvalidAction("Research stopped at its reasoning budget before a supported answer was ready.")
        if writer is None:
            raise InvalidAction("Research needs a configured reasoning model.")
        checkpoint()
        self.calls += 1
        return function(writer, packet)

    def next(self, writer, packet):
        # Correct a rejected structured decision before any operation is dispatched.
        # Transport errors and already-executed browser effects are never replayed.
        for attempt in range(2):
            data = self.generate(writer, compose_research, packet)
            try:
                data = self.validate_decision(data)
            except InvalidAction as error:
                diagnostics.event("research_decision_rejected", attempt=attempt + 1, reason=str(error), retrying=attempt == 0)
                if attempt:
                    raise
                packet = {**packet, "decision_correction": {"validation_error": str(error), "rejected_decision": data}}
            else:
                break
        diagnostics.event(
            "research_stage",
            kind=data["kind"],
            model_call=self.calls,
            **({"reason": data["reason"]} if diagnostics.record_content() else {}),
        )
        self.feedback = ""
        return data

    def validate_decision(self, data):
        kind = data.get("kind")
        if kind not in STAGES:
            raise InvalidAction("The research model returned an unknown stage")
        for key, limit in (("goal", 2000), ("question", 500), ("reason", 500), ("limitations", 1500)):
            data[key] = _text(data.get(key), limit)
        if not isinstance(data.get("claims"), list) or len(data["claims"]) > 12:
            raise InvalidAction("The research answer exceeded its claim limit")
        if kind != "clarify" and data["question"]:
            raise InvalidAction("The research model mixed incompatible stages")
        if kind != "answer" and data["claims"]:
            raise InvalidAction("Only a final research answer may contain claims")
        # A description of what to read in goal does not introduce a second action. The discriminator alone chooses
        # the operation: ignore inactive narrative fields, but never accept a simultaneous clarification or claims.
        for field, active_kind in (("goal", "browse"), ("limitations", "answer")):
            if kind != active_kind and data[field]:
                diagnostics.event("research_field_ignored", kind=kind, field=field)
                data[field] = ""
        if kind == "browse":
            _text(data["goal"], 2000, required=True)
        if kind == "clarify":
            _text(data["question"], 500, required=True)
        return data

    # -- evidence ----------------------------------------------------------------------------------------------

    def collect(self, backend):
        """Read the current page into the evidence list. The only way a page becomes a source."""
        if not callable(getattr(backend, "read_page", None)):
            raise InvalidAction("This browser provider cannot read page text. Select CDP, Obscura or Playwright CLI.")
        if len(self.sources) - 1 >= MAX_SOURCES:
            self.feedback = "Page source limit reached; answer from collected evidence or explain what remains unavailable."
            return
        page = page_record(backend.read_page())
        if not page["text"].strip():
            self.feedback = "The current page returned no readable text. Open another relevant source or report the access limit."
            return
        if any(s["url"] == page["url"] and s["text"] == page["text"] for s in self.sources):
            self.feedback = "This exact page content has already been collected; use it or visit a different source."
            return
        page["id"], page["origin"] = f"s{len(self.sources)}", "page"
        self.sources.append(page)
        diagnostics.event(
            "research_evidence",
            source_id=page["id"],
            url=page["url"],
            characters=len(page["text"]),
            truncated=page["truncated"],
            evidence=page,
        )

    def read_urls(self):
        """Addresses of the pages this run really read."""
        return {s["url"] for s in self.sources if s["origin"] == "page"}

    def allowed_urls(self, observed):
        """Every destination research may open: the user's own words, the search page, what is open, what was read."""
        candidates = {
            *self.tools.urls_in(self.goal),
            *(u for reply in self.replies for u in self.tools.urls_in(reply)),
            self.search_url,
            observed.url,
            *observed.tabs.values(),
            *(s["url"] for s in self.sources),
            *(link["url"] for s in self.sources for link in s["links"]),
        }
        return {url for url in candidates if safe_url(url)}

    # -- what the model may make the browser do ----------------------------------------------------------------

    def validate_plan(self, steps, observed):
        allowed = self.allowed_urls(observed)
        for step in steps:
            if step.effect not in READ_EFFECTS:
                raise InvalidAction("The research plan requested an operation outside browser reading and navigation.")
            if step.effect in {"url", "tab_created"} and step.value not in allowed:
                raise InvalidAction("The research plan used a destination absent from the request and observed sources.")

    def validate_action(self, action, observed):
        controller = observed.elements.get(action.target)
        disclosure = (
            action.kind == "click"
            and action.parameter_source == "disclosure:" + action.target
            and controller is not None
            and controller.controls
            and controller.expanded is False
            and controller.enabled
            and not controller.secret
        )
        if action.kind not in READ_ACTIONS and not self.tools.is_search_action(action, observed) and not disclosure:
            raise InvalidAction("Research cannot send, edit or delete content; the proposed browser operation was stopped.")
        # Query destinations come from the currently observed form, bound by the existing executor.
        if (
            action.kind in {"navigate", "tab_create"}
            and action.value not in self.allowed_urls(observed)
            and not action.parameter_source.startswith("form:")
        ):
            raise InvalidAction("Research navigation lacked an observed destination.")

    # -- the answer --------------------------------------------------------------------------------------------

    def draft(self, data):
        """The spoken and written answer, or InvalidAction when any support is not what it claims to be.

        Code checks every citation against the pages this run read: the source must exist, the quote must be in its
        text, and the address shown for it comes from the adapter. For research, at least one real page must back the
        answer and no address may appear that was not read or seen as a link on a page that was.
        """
        if not data["claims"]:
            raise InvalidAction("The research model returned an empty answer.")
        research = self.route != "reason"
        sources = {s["id"]: s for s in self.sources}
        used, rendered, spoken = [], [], []
        for claim in data["claims"]:
            if not isinstance(claim, dict) or set(claim) != {"text", "citations"}:
                raise InvalidAction("The research model returned an invalid claim.")
            statement = _text(claim["text"], 1500 if research else 20000, required=True)
            cites = claim["citations"]
            if not isinstance(cites, list) or not (1 if research else 0) <= len(cites) <= 4:
                raise InvalidAction("A research claim had no supporting source.")
            refs = []
            for cite in cites:
                if not isinstance(cite, dict) or set(cite) != {"source_id", "quote"}:
                    raise InvalidAction("The research model returned an invalid citation.")
                identity = _text(cite["source_id"], 32, required=True)
                quote = _flat(_text(cite["quote"], 2000, required=True))
                source = sources.get(identity)
                if source is None or quote not in _flat(source["text"]):
                    raise InvalidAction("A research citation did not match any collected evidence.")
                if len(quote) < min(12, len(_flat(source["text"]))):
                    raise InvalidAction("A research citation was too short to establish support.")
                if identity not in used:
                    used.append(identity)
                refs.append(str(used.index(identity) + 1))
            rendered.append(statement + (" [" + ", ".join(dict.fromkeys(refs)) + "]" if refs else ""))
            spoken.append(statement)
        if research and not any(sources[k]["origin"] == "page" for k in used):
            raise InvalidAction("This research task needs evidence from a page that was actually read.")
        if data["limitations"]:
            rendered.append("Limitations: " + data["limitations"])
            spoken.append("Limitations: " + data["limitations"])
        if research:
            self.check_addresses(spoken)
        for i, identity in enumerate(used, 1):
            source = sources[identity]
            # Citation addresses always come from the adapter, never generated answer text.
            location = source["url"] if source["origin"] == "page" else "User-provided text"
            rendered.append(f"[{i}] {_flat(source['title'])} — {location}")
        return Answer("\n\n".join(rendered), True, spoken_text=" ".join(spoken))

    def check_addresses(self, passages):
        """A research answer may name only addresses of pages that were read, or links seen on them."""
        known = self.read_urls() | {link["url"] for s in self.sources for link in s["links"]}
        # A prefix is enough: a pattern that stops at a closing bracket cuts such an address short.
        named = (address.rstrip(".,;:!?") for passage in passages for address in ADDRESS.findall(passage))
        if any(not any(url.startswith(address) for url in known) for address in named):
            raise InvalidAction("The research answer named an address that was not read.")

    def finish(self, writer, data, packet):
        answer = self.draft(data)
        if self.route == "reason":
            # This route uses only supplied text and authored reasoning. A second call to the same model is not
            # independent evidence of correctness.
            self.answer = answer
            diagnostics.event("reasoning_answer_ready", validation="structured_response")
            return answer
        review = self.generate(writer, review_research, {**packet, "proposed_answer": data})
        if type(review.get("supported")) is not bool or type(review.get("complete")) is not bool:
            raise InvalidAction("The answer checker did not return a valid decision.")
        feedback = _text(review.get("feedback"), 1500)
        diagnostics.event(
            "research_answer_checked",
            supported=review["supported"],
            complete=review["complete"],
            **({"feedback": feedback} if diagnostics.record_content() else {}),
        )
        if review["supported"] and review["complete"]:
            self.answer = answer
            return answer
        self.feedback = (
            feedback or "The answer is not supported or does not cover the complete request; collect missing evidence."
        )
        return None

    # -- the run's record --------------------------------------------------------------------------------------

    def summary(self):
        """Counts and flags for the run report; the model that answered is in the provider switch events."""
        return {
            "route": self.route,
            "model_call_limit": self.budget,
            "model_calls": self.calls,
            "source_count": len(self.sources) - 1,
            "browser_tasks_completed": len(self.completed),
            "answer_verified": self.answer is not None and self.route == "research",
            "answer_ready": self.answer is not None,
            "answer_validation": "source_review" if self.route == "research" else "structured_response",
        }

    def mark_completed(self, steps):
        if self.browser_goal:
            self.completed.append(
                {"goal": self.browser_goal, "verified_milestones": [asdict(s) for s in steps if s.id in self.batch_ids]}
            )
            self.browser_goal = ""
