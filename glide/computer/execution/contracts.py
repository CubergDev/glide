"""Validated, observation-bound primitives and desired effects. No generated code."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import parse_qs, urlsplit, urlunsplit

from ..writer import looks_credential

KINDS = {"inspect", "click", "type", "key", "navigate", "tab_create", "tab_switch", "tab_close", "scroll"}
EFFECTS = {
    "url",
    "url_query",
    "query_submitted",
    "media_playing",
    "disclosure_expanded",
    "tab_created",
    "tab_active",
    "tab_closed",
    "field_value",
    "scroll",
    "element_present",
    "element_absent",
    "focus",
    "app",
}
KEYS = {"return", "escape", "tab", "delete", "left", "right", "up", "down", "home", "end", "space", *"abcdefghijklmnopqrstuvwxyz"}
MODIFIERS = {"command", "control", "alt", "shift"}
SCROLL_DIRECTIONS = {"up", "down"}
MAX_REPETITIONS = 100
MAX_ACTION_CHOICES = 250  # Reserve five workflow choices below the provider's 255-choice ceiling.
MAX_PLAN_STEPS = 16


class InvalidAction(ValueError):
    pass


class UnsupportedCapability(InvalidAction):
    def __init__(self, missing, available=()):
        self.missing = tuple(sorted(set(missing)))
        super().__init__(
            "Unsupported capability required: "
            + ", ".join(self.missing)
            + (". Available capabilities: " + ", ".join(sorted(available)) if available else "")
        )


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_url(value: str) -> bool:
    try:
        p = urlsplit(value)
        return (
            p.scheme in {"https", "http"}
            and bool(p.hostname)
            and p.username is None
            and p.password is None
            and (p.port is None or 0 < p.port <= 65535)
            and not any(c.isspace() or ord(c) < 32 for c in value)
        )
    except ValueError:
        return False


def canonical_url(value: str) -> str:
    p = urlsplit(value)
    # Ordinary root redirects commonly add www. This is destination verification,
    # never a credential or origin authorization rule.
    host = (p.hostname or "").lower().removeprefix("www.")
    port = f":{p.port}" if p.port and p.port != {"http": 80, "https": 443}.get(p.scheme) else ""
    return urlunsplit((p.scheme, host + port, p.path.rstrip("/"), p.query, p.fragment))


def observed_link_destination(requested: str, committed: str, declared_canonical: str = "") -> bool:
    """Query normalization needs the page's canonical identity in both selected and committed URLs."""
    if not safe_url(requested) or not safe_url(committed):
        return False

    def origin(url):
        parsed = urlsplit(url)
        return parsed.scheme, parsed.hostname, parsed.port or {"http": 80, "https": 443}[parsed.scheme]

    if origin(requested) != origin(committed):
        return False
    expected, actual = urlsplit(canonical_url(requested)), urlsplit(canonical_url(committed))
    if expected == actual:
        return True
    if expected[:3] != actual[:3] or expected.fragment != actual.fragment:
        return False
    if not safe_url(declared_canonical):
        return False
    if origin(declared_canonical) != origin(committed):
        return False
    canonical = urlsplit(canonical_url(declared_canonical))
    if canonical[:3] != actual[:3] or canonical.fragment != actual.fragment:
        return False
    required, retained = parse_qs(expected.query, keep_blank_values=True), parse_qs(actual.query, keep_blank_values=True)
    identity = parse_qs(canonical.query, keep_blank_values=True)
    return (
        bool(identity)
        and all(required.get(key) == values for key, values in retained.items())
        and all(required.get(key) == values == retained.get(key) for key, values in identity.items())
    )


@dataclass(frozen=True)
class Element:
    id: str
    label: str
    role: str = ""
    value: str | None = None
    typeable: bool = False
    secret: bool = False
    enabled: bool = True
    ref: object = field(default=None, repr=False, compare=False)
    point: tuple[float, float] | None = None
    shortcuts: tuple[str, ...] = ()
    search: bool = False  # Explicit search semantics observed on the page, never inferred from a hostname.
    href: str = ""
    expanded: bool | None = None
    controls: tuple[str, ...] = ()

    def public(self):
        return {
            "id": self.id,
            "label": self.label,
            "role": self.role,
            "value": None if self.secret or (self.value is not None and len(self.value) > 2048) else self.value,
            "value_length": None if self.secret or self.value is None else len(self.value),
            "value_digest": digest(self.value) if not self.secret and self.value is not None and len(self.value) > 2048 else None,
            "typeable": self.typeable,
            "secret": self.secret,
            "enabled": self.enabled,
            "shortcuts": self.shortcuts,
            "search": self.search,
            "href": self.href,
            "expanded": self.expanded,
            "controls": self.controls,
        }


@dataclass(frozen=True)
class Container:
    id: str
    label: str
    position: float
    maximum: float
    ref: object = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Media:
    id: str
    label: str
    paused: bool
    ended: bool
    ready_state: int
    current_time: float


@dataclass(frozen=True)
class Navigation:
    requested_url: str
    tab: str
    frame: str
    loader: str
    owner: str = ""


@dataclass
class Observation:
    app: str
    owner: str
    url: str = ""
    active_tab: str = ""
    tabs: dict[str, str] = field(default_factory=dict)
    elements: dict[str, Element] = field(default_factory=dict)
    containers: dict[str, Container] = field(default_factory=dict)
    focus: str = ""
    capabilities: set[str] = field(default_factory=lambda: {"inspect"})
    ready: bool = True
    browser_front: bool = True
    forms: dict = field(default_factory=dict)
    media: dict[str, Media] = field(default_factory=dict)
    available_after_navigation: set[str] = field(default_factory=set)
    navigation: Navigation | None = None
    canonical_url: str = ""

    def packet(self):
        return {
            "app": self.app,
            "owner": self.owner,
            "url": self.url,
            "active_tab": self.active_tab,
            "tabs": self.tabs,
            "elements": [e.public() for e in self.elements.values()],
            "containers": [
                {"id": c.id, "label": c.label, "position": c.position, "maximum": c.maximum} for c in self.containers.values()
            ],
            "query_forms": [
                {"id": f.id, "field": f.field, "action": f.action, "parameter": f.parameter, "values": f.values}
                for f in self.forms.values()
            ],
            "media": [vars(item) for item in self.media.values()],
            "focus": self.focus,
            "capabilities": sorted(self.capabilities),
            "available_after_navigation": sorted(self.available_after_navigation),
            "navigation": vars(self.navigation) if self.navigation else None,
            "canonical_url": self.canonical_url,
            "ready": self.ready,
            "browser_front": self.browser_front,
            "parameter_domains": {
                "key": sorted(KEYS) if "key" in self.capabilities else [],
                "modifiers": sorted(MODIFIERS) if "key" in self.capabilities else [],
                "repetitions": {"minimum": 1, "maximum": MAX_REPETITIONS},
                "scroll": {
                    "directions": sorted(SCROLL_DIRECTIONS) if "scroll" in self.capabilities else [],
                    "quantity_unit": "verified scroll operations",
                    "exact_distance_parameters": False,
                },
            },
        }

    @property
    def identity(self):
        return digest(self.packet())


@dataclass(frozen=True)
class Action:
    kind: str
    observation: str
    target: str = ""
    value: str = ""
    modifiers: tuple[str, ...] = ()
    parameter_source: str = ""
    target_contract: str = ""  # Code-attached Jev binding of a description to an observed handle.

    @property
    def identity(self):
        return digest([self.kind, self.target, self.value, self.modifiers, self.parameter_source, self.target_contract])


def validate(action: Action, obs: Observation):
    if not isinstance(action.value, str) or len(action.value) > 8192 or "\x00" in action.value:
        raise InvalidAction("Invalid action value")
    if not isinstance(action.target_contract, str) or (
        action.target_contract
        and (
            action.kind not in {"type", "scroll"}
            or len(action.target_contract) != 64
            or any(c not in "0123456789abcdef" for c in action.target_contract)
        )
    ):
        raise InvalidAction("Invalid target binding")
    if not isinstance(action.parameter_source, str) or len(action.parameter_source) > 8192:
        raise InvalidAction("Invalid parameter source")
    if action.parameter_source:
        if action.parameter_source.startswith("disclosure:"):
            source = action.parameter_source.removeprefix("disclosure:")
            controller = obs.elements.get(source)
            if (
                action.kind != "click"
                or action.target != source
                or not controller
                or not controller.controls
                or controller.expanded is not False
                or controller.secret
            ):
                raise InvalidAction("Disclosure control is unavailable or already expanded")
        elif action.parameter_source.startswith("query:"):
            from .query import is_search_action

            if not is_search_action(action, obs):
                raise InvalidAction("Search control changed or is unavailable")
        elif action.parameter_source.startswith("form:"):
            form = obs.forms.get(action.parameter_source.removeprefix("form:"))
            if action.kind != "navigate" or not form:
                raise InvalidAction("Query source is no longer observed")
            values = parse_qs(urlsplit(action.value).query).get(form.parameter, [])
            if len(values) != 1 or form.destination(obs, values[0]) != action.value:
                raise InvalidAction("Navigation differs from the observed query form")
        elif action.kind != "key" or (
            action.parameter_source != "request" and not action.parameter_source.startswith("observed:")
        ):
            raise InvalidAction("Invalid parameter provenance")
        if (
            action.parameter_source.startswith("observed:")
            and action.parameter_source.removeprefix("observed:") not in obs.elements
        ):
            raise InvalidAction("Shortcut source is no longer observed")
    if not isinstance(action.target, str) or len(action.target) > 8192 or "\x00" in action.target:
        raise InvalidAction("Invalid action target")
    if action.kind not in KINDS or action.kind not in obs.capabilities:
        raise UnsupportedCapability([action.kind], obs.capabilities)
    if action.observation != obs.identity:
        raise InvalidAction("Observation changed before dispatch")
    if not obs.ready and action.kind in {"click", "type", "key", "scroll"}:
        raise InvalidAction("Target document is not ready")
    if action.kind in {"inspect", "click", "tab_switch", "tab_close"} and action.value:
        raise InvalidAction("Unexpected value for this action")
    if action.kind in {"inspect", "key", "tab_create"} and action.target:
        raise InvalidAction("Unexpected target for this action")
    if action.kind in {"click", "type"}:
        el = obs.elements.get(action.target)
        if not el or not el.enabled:
            raise InvalidAction("Element is absent or unavailable")
        if action.kind == "click" and el.point is not None and not all(math.isfinite(v) for v in el.point):
            raise InvalidAction("Invalid click geometry")
        if action.kind == "type" and (not el.typeable or el.secret or looks_credential(el.label)):
            raise InvalidAction("Credential or non-text field refused")
    elif action.kind == "scroll":
        container = obs.containers.get(action.target)
        if not container or action.value not in SCROLL_DIRECTIONS:
            raise InvalidAction("Invalid scroll container or direction")
        if not all(math.isfinite(v) for v in (container.position, container.maximum)):
            raise InvalidAction("Invalid scroll geometry")
        if container.maximum <= 0:
            raise InvalidAction("Container cannot scroll")
    elif action.kind in {"navigate", "tab_create"}:
        if not safe_url(action.value):
            raise InvalidAction("Invalid navigation URL")
        if action.kind == "navigate" and action.target and action.target not in obs.tabs:
            raise InvalidAction("Navigation tab is absent")
    elif action.kind in {"tab_switch", "tab_close"} and action.target not in obs.tabs:
        raise InvalidAction("Tab is absent")
    elif action.kind == "key":
        if (
            action.value not in KEYS
            or not set(action.modifiers) <= MODIFIERS
            or len(set(action.modifiers)) != len(action.modifiers)
        ):
            raise InvalidAction("Invalid key or shortcut")
        focused = obs.elements.get(obs.focus)
        if focused and (focused.secret or looks_credential(focused.label)):
            raise InvalidAction("Key input into a credential field refused")
    if action.kind != "key" and action.modifiers:
        raise InvalidAction("Modifiers are valid only for keys")


def rebind(action: Action, before: Observation, fresh: Observation) -> Action:
    """Accept irrelevant UI changes, never a changed target or document owner."""
    from dataclasses import replace

    if action.kind in {"click", "type"}:
        a, b = before.elements.get(action.target), fresh.elements.get(action.target)
        if (
            before.owner != fresh.owner
            or not a
            or not b
            or a.public() != b.public()
            or (a.point != b.point and before.app != "browser")
            or a.ref != b.ref
        ):
            raise InvalidAction("The selected element changed before execution")
    elif action.kind == "scroll":
        a, b = before.containers.get(action.target), fresh.containers.get(action.target)
        if before.owner != fresh.owner or not a or not b or a.label != b.label or a.maximum != b.maximum or a.ref != b.ref:
            raise InvalidAction("The selected scroll container changed")
    elif action.kind == "key":
        if before.owner != fresh.owner or before.focus != fresh.focus:
            raise InvalidAction("Keyboard focus changed before execution")
        if action.parameter_source.startswith("query:"):
            source = action.parameter_source.removeprefix("query:")
            a, b = before.elements.get(source), fresh.elements.get(source)
            if not a or not b or a.public() != b.public() or a.ref != b.ref:
                raise InvalidAction("The search field changed before submission")
        if action.parameter_source.startswith("observed:"):
            source = action.parameter_source.removeprefix("observed:")
            a, b = before.elements.get(source), fresh.elements.get(source)
            if not a or not b or a.public() != b.public() or a.ref != b.ref:
                raise InvalidAction("The observed shortcut parameters changed before execution")
    elif action.kind == "navigate" and action.parameter_source.startswith("form:"):
        form_id = action.parameter_source.removeprefix("form:")
        if before.owner != fresh.owner or before.forms.get(form_id) != fresh.forms.get(form_id):
            raise InvalidAction("The selected form changed before execution")
        old_form = before.forms.get(form_id)
        if not old_form:
            raise InvalidAction("The query source was not observed")
        a, b = before.elements.get(old_form.field), fresh.elements.get(old_form.field)
        if not a or not b or a.public() != b.public():
            raise InvalidAction("The query input changed before execution")
    elif action.kind in {"tab_close", "tab_switch", "navigate"} and action.target:
        if action.target not in fresh.tabs or before.tabs.get(action.target) != fresh.tabs[action.target]:
            raise InvalidAction("The selected tab changed before execution")
        if action.kind == "navigate" and before.active_tab != fresh.active_tab:
            raise InvalidAction("Active navigation tab changed")
    bound = replace(action, observation=fresh.identity)
    validate(bound, fresh)
    return bound


class Backend(Protocol):
    def inspect(self) -> Observation: ...
    def execute(self, action: Action, observed: Observation) -> str: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class Milestone:
    id: str
    goal: str
    effect: str
    target: str = ""
    value: str = ""
    quantity: int = 1

    def __post_init__(self):
        if any(
            not isinstance(v, str) or len(v) > 8192 or "\x00" in v
            for v in (self.id, self.goal, self.effect, self.target, self.value)
        ):
            raise InvalidAction("Invalid milestone parameter")
        if self.effect not in EFFECTS:
            raise UnsupportedCapability(["verification:" + self.effect], EFFECTS)
        if type(self.quantity) is not int or not 1 <= self.quantity <= MAX_REPETITIONS:
            raise InvalidAction("Invalid milestone effect or quantity")
        if not self.id or len(self.id) > 64 or not self.goal or len(self.goal) > 1000:
            raise InvalidAction("Invalid milestone identity or goal")
        if self.effect in {"url", "tab_created"} and not safe_url(self.value):
            raise InvalidAction("Invalid milestone URL")
        if self.effect == "url_query" and (
            not self.value
            or not safe_url(self.target)
            or not any(values == [self.value] for values in parse_qs(urlsplit(self.target).query).values())
        ):
            raise InvalidAction("Query verification needs an expected URL with a named parameter equal to the exact text")
        if self.effect == "query_submitted" and (not self.target or not self.value):
            raise InvalidAction("Query submission needs a target description and exact text")
        if self.effect == "scroll" and self.value not in SCROLL_DIRECTIONS:
            raise UnsupportedCapability(["scroll:" + self.value], SCROLL_DIRECTIONS)
        if (
            self.effect in {"field_value", "element_present", "element_absent", "focus", "app", "scroll", "disclosure_expanded"}
            and not self.target
        ):
            raise InvalidAction("Milestone needs an observed target description")
        if self.effect not in {"tab_created", "scroll"} and self.quantity != 1:
            raise InvalidAction("Quantity unsupported for this effect")

    @property
    def contract(self):
        return digest([self.effect, self.target, self.value, self.quantity])


def effect(m: Milestone, action: Action | None, before: Observation, after: Observation, receipt="") -> str:
    """A stable effect ID, or empty. Dispatch receipts alone never prove completion."""
    if not after.ready and m.effect not in {"url", "tab_created", "tab_active", "tab_closed"}:
        return ""
    if m.effect == "url" and after.browser_front and canonical_url(after.url) == canonical_url(m.value):
        return digest([after.active_tab, after.url])
    if (
        m.effect == "url"
        and action
        and action.kind == "navigate"
        and action.value == m.value
        and after.browser_front
        and safe_url(after.url)
        and after.navigation
        and after.navigation.requested_url == action.value
        and after.navigation.tab == after.active_tab
        and after.navigation.owner == after.owner
        and after.navigation.loader == receipt
        and bool(receipt)
        and bool(after.navigation.frame)
        and before.owner != after.owner
    ):
        return digest(["redirected", after.active_tab, after.url, after.navigation.loader])
    if m.effect == "url_query" and after.browser_front and safe_url(after.url):
        destination = urlsplit(canonical_url(after.url))
        expected = urlsplit(canonical_url(m.target))
        same_destination = (
            destination.scheme == expected.scheme and destination.netloc == expected.netloc and destination.path == expected.path
        )
        query, required = parse_qs(destination.query), parse_qs(expected.query)
        if same_destination and required and all(query.get(key) == values for key, values in required.items()):
            return digest(["query", after.active_tab, after.url, m.value])
    if m.effect == "query_submitted":
        from .query import submission_effect

        return submission_effect(m, action, before, after)
    if m.effect == "media_playing" and after.browser_front and action and action.kind in {"click", "key"}:
        selected = before.elements.get(action.target) if action.kind == "click" else None
        followed_result = (
            selected
            and safe_url(selected.href)
            and before.url != after.url
            and observed_link_destination(selected.href, after.url, after.canonical_url)
        )
        media = [
            item
            for item in after.media.values()
            if followed_result or (m.target and (item.id == m.target or item.label.casefold() == m.target.casefold()))
        ]
        if len(media) == 1:
            item = media[0]
            previous = before.media.get(item.id)
            resumed_selected = before.owner == after.owner and previous and previous.paused
            if (
                (followed_result or resumed_selected)
                and not item.paused
                and not item.ended
                and item.ready_state >= 2
                and math.isfinite(item.current_time)
                and item.current_time > 0
            ):
                return digest(["playing", after.owner, item.id])
    if (
        m.effect == "tab_created"
        and action
        and action.kind == "tab_create"
        and receipt
        and receipt not in before.tabs
        and receipt in after.tabs
        and (
            canonical_url(after.tabs[receipt]) == canonical_url(m.value)
            or (
                after.navigation
                and after.navigation.requested_url == m.value
                and action.value == m.value
                and after.navigation.tab == receipt == after.active_tab
                and after.navigation.owner == after.owner
                and after.navigation.frame
                and after.navigation.loader
                and after.tabs[receipt] == after.url
                and safe_url(after.url)
            )
        )
    ):
        return digest(["created", receipt])
    if (
        m.effect == "tab_active"
        and after.active_tab in after.tabs
        and (after.active_tab == m.target or after.tabs[after.active_tab] == m.value)
    ):
        return digest(["active", after.active_tab])
    if (
        m.effect == "tab_closed"
        and action
        and action.kind == "tab_close"
        and action.target in before.tabs
        and (action.target == m.target or before.tabs[action.target] == m.value)
        and action.target not in after.tabs
    ):
        return digest(["closed", action.target])
    if m.effect == "scroll" and action and action.kind == "scroll":
        a, b = before.containers.get(action.target), after.containers.get(action.target)
        correct_target = a and (
            a.id == m.target or a.label.casefold() == m.target.casefold() or action.target_contract == m.contract
        )
        if (
            a
            and b
            and correct_target
            and ((b.position > a.position and m.value == "down") or (b.position < a.position and m.value == "up"))
        ):
            return digest([after.owner, b.id, b.position])
    matches = [e for e in after.elements.values() if e.id == m.target or e.label.casefold() == m.target.casefold()]
    if m.effect == "disclosure_expanded":
        controllers = [e for e in matches if e.controls and type(e.expanded) is bool]
        if len(controllers) == 1 and controllers[0].expanded is True:
            return digest(["expanded", after.owner, controllers[0].id, controllers[0].controls])
    if m.effect == "field_value" and action and action.kind == "type" and action.target_contract == m.contract:
        original, selected = before.elements.get(action.target), after.elements.get(action.target)
        if before.owner == after.owner and original and selected and original.typeable and not original.secret:
            matches = [selected]
    if m.effect == "field_value" and len(matches) == 1 and not matches[0].secret and matches[0].value == m.value:
        return digest(["value", after.owner, m.target, m.value])
    if m.effect == "element_present" and matches:
        return digest(["present", after.owner, m.target])
    if m.effect == "element_absent" and not matches:
        old = [e for e in before.elements.values() if e.id == m.target or e.label.casefold() == m.target.casefold()]
        if old:
            return digest(["absent", before.owner, m.target])
    if m.effect == "focus" and any(e.id == after.focus for e in matches):
        return digest(["focus", after.owner, after.focus])
    if m.effect == "app" and after.app.casefold() == m.target.casefold():
        return digest(["app", after.app, after.owner])
    return ""


def primitive_effect(action: Action | None, before: Observation, after: Observation, receipt="") -> str:
    """Verify an intermediate operation without claiming the user's milestone is complete."""
    if not action:
        return ""
    if action.kind == "type":
        field = after.elements.get(action.target)
        if field and not field.secret and field.value == action.value:
            return digest(["typed", after.owner, field.id, field.value])
    if action.kind == "tab_create":
        return effect(Milestone("operation", "Create tab", "tab_created", value=action.value), action, before, after, receipt)
    if action.kind == "tab_close":
        return effect(Milestone("operation", "Close tab", "tab_closed", target=action.target), action, before, after)
    if action.kind == "tab_switch" and after.active_tab == action.target:
        return digest(["switched", action.target])
    if action.kind == "navigate" and canonical_url(after.url) == canonical_url(action.value) and after.browser_front:
        return digest(["navigated", after.active_tab, after.url])
    if action.kind in {"click", "key"} and after.ready:
        # Focus, changed document, changed selected control, or newly exposed controls.
        # A changing clock/status value is deliberately excluded.
        if before.owner != after.owner or before.focus != after.focus:
            return digest(["context", after.owner, after.focus])
        target = action.target if action.kind == "click" else before.focus
        old, new = before.elements.get(target), after.elements.get(target)
        if old and (not new or old.public() != new.public()):
            return digest(["control", target, new.public() if new else None])

        def controls(obs):
            return sorted(
                (e.id, e.label, e.role, "" if e.secret else e.value or "")
                for e in obs.elements.values()
                if e.role.casefold() not in {"status", "axstatictext", "text"}
            )

        if controls(before) != controls(after):
            return digest(["controls", after.owner, controls(after)])
    return ""
