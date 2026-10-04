"""Plan validation and bounded, same-provider reasoning."""

import re
from dataclasses import asdict, replace
from urllib.parse import urlsplit

from .. import browser_settings, diagnostics
from ..config import SITES
from ..control import checkpoint
from ..writer import compose_plan
from .contracts import (
    EFFECTS,
    MAX_PLAN_STEPS,
    MAX_REPETITIONS,
    InvalidAction,
    Milestone,
    UnsupportedCapability,
    canonical_url,
    safe_url,
)
from .grounding import extract

NOT_GROUNDED = (
    "A destination in the plan is not an address the user gave, a link observed on the page, an open tab or the "
    "configured search address. Use one of those, or return a question asking the user which address to use."
)


def _origin(url):
    """The canonical host and port of an address, for comparing where two addresses lead."""
    parts = urlsplit(canonical_url(url))
    return parts.netloc


def grounded_origins(goal, reply, observed, steps, search_url):
    """The places a plan with no supervisor may send the browser, as canonical hosts: the user's own words (an address,
    or a host named in them, or a site the code's catalog names that they named), what the page offers (its address,
    the open tabs, the links and forms on it), the configured search address and what was already registered. Model
    text and page text are never an address of their own: a page that says "go to ..." adds nothing."""
    addresses = [url.rstrip(".,;:!?)]}") for url in (*extract(goal).urls, *extract(reply).urls)]
    words = set(re.findall(r"\w+", f"{goal} {reply}".casefold()))
    addresses += [url for name, url in SITES.items() if set(name.casefold().split("_")) <= words]
    addresses += [observed.url, search_url, *observed.tabs.values(), *(e.href for e in observed.elements.values())]
    addresses += [form.action for form in observed.forms.values()]
    addresses += [s.value for s in steps if s.effect in {"url", "tab_created"}]
    return {_origin(a) for a in addresses if a and safe_url(a)}


def plan(writer, goal, observed, steps=(), progress=(), reason="", reply="", context=None):
    if writer is None:
        raise InvalidAction("This goal needs planning, but the configured model provider is unavailable")
    packet = {
        "goal": goal,
        "observation": observed.packet(),
        "available_effects": sorted(EFFECTS & set(context.get("allowed_effects", EFFECTS)) if context else EFFECTS),
        "configured_search_url": browser_settings.current().search_url,
        "original_milestones": [asdict(s) for s in steps],
        "verified_progress": progress,
        "contradiction": reason,
        "user_reply": reply,
        **({"task_context": context} if context else {}),
    }
    grounded = None if context else grounded_origins(goal, reply, observed, steps, packet["configured_search_url"])
    # Only invalid, parsed plans get one correction. No operations are dispatched here;
    # transport failures, cancellation and unsupported capabilities are never replayed.
    for attempt in range(2):
        checkpoint()
        data = compose_plan(writer, packet)
        try:
            result = validate_plan(
                data, observed, steps, allowed_effects=context.get("allowed_effects") if context else None, grounded=grounded
            )
        except UnsupportedCapability:
            raise
        except InvalidAction as error:
            diagnostics.event("plan_validation_failed", attempt=attempt + 1, reason=str(error), retrying=attempt == 0)
            if attempt:
                raise
            packet["plan_correction"] = {"validation_error": str(error), "rejected_plan": data}
        else:
            if attempt:
                diagnostics.event("plan_correction_completed", attempts=attempt + 1)
            return result


def validate_plan(data, observed, steps=(), *, allowed_effects=None, grounded=None):
    """Validate the entire batch before registering or dispatching any of it. With `grounded` (a set of canonical hosts),
    every address the plan would open must lead to one of them; the research supervisor does its own check instead."""
    if not isinstance(data, dict) or set(data) not in ({"question", "steps"}, {"question", "steps", "unsupported"}):
        raise InvalidAction("Planner returned an invalid contract")
    question, raw = data["question"], data["steps"]
    unsupported = data.get("unsupported", [])
    if (
        not isinstance(unsupported, list)
        or len(unsupported) > MAX_PLAN_STEPS
        or any(not isinstance(v, str) or not v.strip() or len(v) > 128 or "\x00" in v for v in unsupported)
    ):
        raise InvalidAction("Invalid unsupported capability report")
    if not isinstance(question, str) or len(question) > 500 or not isinstance(raw, list) or len(raw) > MAX_PLAN_STEPS:
        raise InvalidAction("Invalid plan size or question")
    if unsupported:
        if set(unsupported) & observed.capabilities:
            raise InvalidAction("Planner reported an available capability as unsupported")
        # Never execute an offered supported fragment when another requirement is unavailable.
        raise UnsupportedCapability(unsupported, observed.capabilities)
    if bool(question) == bool(raw):
        raise InvalidAction("Planner must return a question or milestones")
    planned = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"id", "goal", "effect", "target", "value", "quantity"}:
            raise InvalidAction("Invalid milestone fields")
        if any(not isinstance(item[k], str) or len(item[k]) > 8192 for k in ("id", "goal", "effect", "target", "value")):
            raise InvalidAction("Invalid milestone parameter")
        planned.append(Milestone(**item))
    if allowed_effects is not None and any(step.effect not in allowed_effects for step in planned):
        raise InvalidAction(
            "This browser subtask only supports effects: "
            + ", ".join(sorted(allowed_effects))
            + ". Plan the bounded navigation/search goal; the supervisor reads and interprets its results."
        )
    # These effects require a dedicated operation AND its observable identity/offset.
    # A keyboard workaround cannot verify missing backend capabilities.
    required = {
        "query_submitted": "query_form",
        "tab_created": "tab_create",
        "tab_active": "tab_switch",
        "tab_closed": "tab_close",
        "scroll": "scroll",
        "media_playing": "media_state",
        "disclosure_expanded": "click",
    }
    capabilities = set(observed.capabilities)
    known_tab_urls = set(observed.tabs.values())
    original = {step.id: step for step in steps}
    has_page = bool(observed.active_tab) and safe_url(observed.url)
    for milestone in planned:
        capability = required.get(milestone.effect)
        if capability and capability not in capabilities:
            raise UnsupportedCapability([capability], capabilities)
        if milestone.effect == "disclosure_expanded":
            previous = original.get(milestone.id)
            retained = previous is not None and previous.contract == milestone.contract
            matches = [
                element
                for element in observed.elements.values()
                if (element.id == milestone.target or element.label.casefold() == milestone.target.casefold())
                and element.controls
                and type(element.expanded) is bool
                and element.enabled
                and not element.secret
            ]
            if not retained and (len(matches) != 1 or milestone.value):
                raise InvalidAction(
                    "Disclosure target must exactly match one observed controller ID or label, with empty value. "
                    "Do not append descriptions or use the controlled panel ID. Navigate or scroll and observe "
                    "the controller before planning expansion."
                )
        # url_query carries its address in target (policy.candidates navigates to it)
        where = milestone.target if milestone.effect == "url_query" else milestone.value
        if grounded is not None and milestone.effect in {"url", "url_query", "tab_created"} and _origin(where) not in grounded:
            raise InvalidAction(NOT_GROUNDED)
        if milestone.effect in {"url", "tab_created"}:
            has_page = True
            known_tab_urls.add(milestone.value)
            capabilities.update(observed.available_after_navigation)
        elif milestone.effect in {"tab_active", "tab_closed"}:
            previous = original.get(milestone.id)
            retained = previous is not None and previous.contract == milestone.contract
            by_id = milestone.target in observed.tabs and (
                not milestone.value or observed.tabs[milestone.target] == milestone.value
            )
            by_url = not milestone.target and bool(milestone.value) and milestone.value in known_tab_urls
            if not retained and not (by_id or by_url):
                raise InvalidAction(
                    "Tab activation/closure requires an exact observed tab ID in target, or an empty target and "
                    "the exact tab URL in value. A descriptive target is not executable. For a future tab, first "
                    "create it and use that same URL; do not invent its tab ID."
                )
            if milestone.effect == "tab_active":
                has_page = safe_url(observed.tabs.get(milestone.target, milestone.value))
                if has_page:
                    capabilities.update(observed.available_after_navigation)
        elif milestone.effect == "query_submitted" and not has_page:
            raise InvalidAction(
                "Query submission requires a selected web page. Add a tab_created or url milestone "
                "for the requested or configured search destination before the query."
            )
    if len({s.id for s in planned}) != len(planned):
        raise InvalidAction("Duplicate milestone IDs")
    if steps and not question:
        by_id = {s.id: s for s in planned}
        if any(s.id not in by_id or by_id[s.id].contract != s.contract for s in steps):
            raise InvalidAction("Replanning discarded or changed an original requirement")
    if not steps:
        planned = coalesce_repetitions(planned)
    return planned, question


def coalesce_repetitions(planned):
    """Compact adjacent identical repeatable effects before their IDs become durable."""
    compact = []
    for step in planned:
        previous = compact[-1] if compact else None
        if (
            previous
            and step.effect in {"tab_created", "scroll"}
            and (previous.effect, previous.target, previous.value) == (step.effect, step.target, step.value)
        ):
            quantity = previous.quantity + step.quantity
            if quantity > MAX_REPETITIONS:
                raise InvalidAction("Repeated effect quantity exceeds the repetition safety limit")
            compact[-1] = replace(previous, quantity=quantity)
        else:
            compact.append(step)
    return compact
