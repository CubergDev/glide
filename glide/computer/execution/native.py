"""Accessibility-first observations and atomic native execution on the runner's thread."""

from ..control import checkpoint
from ..models import DesktopError
from ..platform_adapter import desktop
from ..writer import looks_credential
from .contracts import Container, Element, Observation, digest, validate

TAB_CAPABILITIES = {"tab_create", "tab_switch", "tab_close"}


class NativeBackend:
    transport = "native"
    passive_inspection = True

    def __init__(self, browser):
        self.browser = browser

    def check(self):
        """Nothing to probe without touching the desktop; permissions are the desktop preflight's job."""

    def inspect_for(self, step, action=None):
        """Read browser identities for browser effects; UI input still gets a fresh full tree.

        These observations deliberately contain no actionable UI controls. A navigation or
        tab operation needs stable browser state, not three accessibility tree walks and a
        screenshot fallback on every preflight and readiness poll.
        """
        browser_effect = step.effect in {"url", "tab_created", "tab_active", "tab_closed"} or (
            step.effect == "url_query" and action is not None and action.kind == "navigate"
        )
        if not browser_effect or (
            action is not None and action.kind not in {"navigate", "tab_create", "tab_switch", "tab_close"}
        ):
            return self.inspect()
        checkpoint()
        app, pid = desktop.frontmost_app_and_pid()
        checkpoint()
        state = self._tabs()
        checkpoint()
        capabilities = {"inspect", "navigate"} | (set() if state.get("unsupported") else TAB_CAPABILITIES)
        url = self._url(app, state)
        return Observation(
            app,
            self._owner(app, pid, state, url),
            url,
            state.get("active", ""),
            state.get("tabs", {}),
            capabilities=capabilities,
            ready=bool(state.get("ready", False)),
            browser_front=app == self.browser,
        )

    def _tabs(self):
        try:
            return desktop.execution_tabs(self.browser)
        except DesktopError:
            return {"tabs": {}, "active": "", "unsupported": True}

    def _url(self, app, state):
        """The active tab's URL. A URL-only adapter is asked only when this browser is in front: asking a
        browser that is not running could launch it."""
        url = state.get("tabs", {}).get(state.get("active", ""), "")
        if state.get("unsupported") and app == self.browser:
            url = desktop.browser_url(self.browser) or ""
            checkpoint()
        return url

    def _owner(self, app, pid, state, url):
        return f"{app}:{pid}:{state.get('active', '')}:{url}" if app == self.browser else f"{app}:{pid}"

    def inspect(self):
        checkpoint()
        app, pid = desktop.frontmost_app_and_pid()
        bounds = desktop.frontmost_window_bounds(pid)
        width, height = (max(1, bounds[0] + bounds[2]), max(1, bounds[1] + bounds[3])) if bounds else (10000, 10000)
        nodes, _, _ = desktop.actionable_elements(pid, width, height, focused_only=True)
        field = desktop.focused_field()
        elements = {}
        focus = ""
        for n in nodes[:180]:
            if n.covered_by or not n.ref:
                continue
            # Autocomplete may relabel, move or reorder a field while its AX handle stays the same.
            identity = digest([app, pid, n.role, reference_identity(n.ref)])[:24]
            secret = "secure" in n.role.lower() or looks_credential(n.label)
            typeable = n.role in {"AXTextField", "AXTextArea", "AXSearchField", "AXComboBox"} and not secret
            value = desktop.ax_value(n.ref) if typeable else None
            elements[identity] = Element(
                identity,
                n.label,
                n.role,
                value,
                typeable,
                secret,
                enabled=getattr(n, "enabled", True),
                ref=n.ref,
                point=(n.x + n.w / 2, n.y + n.h / 2),
            )
            if field and field.ref == n.ref:
                focus = identity
        # Some AX trees expose only the focused field.
        if field and field.ref and not focus:
            identity = digest([app, pid, field.role, reference_identity(field.ref)])[:24]
            secret = not field.is_text or looks_credential(field.label + " " + field.placeholder)
            elements[identity] = Element(
                identity,
                field.label or field.placeholder,
                field.role,
                None if secret else field.value,
                field.is_text and not secret,
                secret,
                ref=field.ref,
            )
            focus = identity
        for label in desktop.execution_labels(pid):
            identity = digest([app, pid, "label", label["id"]])[:24]
            elements[identity] = Element(identity, label["label"], "status", enabled=False)
        if not any(e.role != "status" for e in elements.values()):
            # Pixels are a fallback, never routine input for the planner.
            from ..perception import capture, perceive

            screen = capture(browser=self.browser)
            for item in perceive(screen, 100, "(structured observation)"):
                identity = digest([app, pid, item.text, item.center])[:24]
                elements[identity] = Element(
                    identity,
                    item.text,
                    item.role,
                    secret=looks_credential(item.text),
                    ref=screen.ax_refs.get(item.index),
                    point=(item.center[0] / screen.scale, item.center[1] / screen.scale),
                )
        state = self._tabs()
        checkpoint()
        url = self._url(app, state)
        containers = {}
        for c in desktop.execution_scrolls(pid):
            identity = digest([app, pid, c["id"], reference_identity(c["ref"])])[:24]
            containers[identity] = Container(identity, c["label"], c["position"], c["maximum"], c["ref"])
        capabilities = {"inspect", "click", "type", "key", "navigate", "scroll"} | (
            set() if state.get("unsupported") else TAB_CAPABILITIES
        )
        return Observation(
            app,
            self._owner(app, pid, state, url),
            url,
            state.get("active", ""),
            state.get("tabs", {}),
            elements,
            containers,
            focus,
            capabilities,
            ready=state.get("ready", True) if app == self.browser else True,
            browser_front=app == self.browser,
        )

    def execute(self, action, observed):
        validate(action, observed)
        desktop.check_abort()
        checkpoint()
        if action.kind == "inspect":
            return ""
        if action.kind in {"tab_create", "tab_switch", "tab_close", "navigate"}:
            if action.kind == "navigate" and "tab_create" not in observed.capabilities:
                if not desktop.open_url(self.browser, action.value):
                    raise DesktopError("Browser did not activate")
                return ""
            return desktop.execution_tab(self.browser, action.kind, action.target, action.value)
        if action.kind == "key":
            desktop.execution_shortcut(action.value, action.modifiers)
        elif action.kind == "scroll":
            desktop.execution_scroll(observed.containers[action.target].ref, action.value)
        elif action.kind == "click":
            el = observed.elements[action.target]
            if el.ref is None or not desktop.ax_press(el.ref):
                checkpoint()
                if el.point is None:
                    raise DesktopError("Element cannot be activated")
                desktop.click_at(el.point)
        elif action.kind == "type":
            el = observed.elements[action.target]
            if observed.focus != action.target:
                focused = desktop.ax_focus(el.ref)
                checkpoint()
                if not focused:
                    raise DesktopError("Text field cannot be focused")
            written = desktop.ax_set_value(el.ref, action.value)
            checkpoint()
            if not written:
                desktop.clear_field()
                checkpoint()
                desktop.type_text(action.value)
        return ""

    def close(self):
        pass


def reference_identity(ref):
    try:
        return hash(ref)
    except TypeError:
        return id(ref)  # Unstable handles fail fresh preflight rather than cross windows.
