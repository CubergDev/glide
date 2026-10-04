"""Attach to an explicitly selected existing CDP browser. Never starts or owns a browser process.

Reads (inspection, page reading) and writes (clicks, typing, navigation) are kept apart on purpose:

- A read may be retried a bounded number of times, because the same read twice cannot change the page. A read that
  fails because the document changed under it (a navigation, a destroyed execution context) is a page problem, not a
  lost connection: it is retried and, if it keeps failing, reported as "couldn't read the selected page".
- A write is never retried. The guard scripts that run before a write are read-only, but a failure there still
  surfaces to the caller, and the write itself is dispatched at most once per call.
- Only a closed or broken transport is a lost connection (`BrowserConnectionError`).
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import time
from dataclasses import replace

import websocket

from .. import browser_settings
from ..browser.cdp import CDPError, Session, _get_json, local_debugger_url
from ..control import checkpoint, controlled
from ..diagnostics import event
from ..models import BrowserConnectionError, DesktopError
from ..writer import looks_credential
from .contracts import Container, Element, Media, Navigation, Observation, safe_url, validate

BROWSER_CAPABILITIES = {
    "inspect",
    "click",
    "type",
    "key",
    "navigate",
    "tab_create",
    "tab_switch",
    "tab_close",
    "scroll",
    "query_form",
    "media_state",
}
READ_ATTEMPTS = 3  # one read, then at most two more, with READ_BACKOFF_S between
READ_BACKOFF_S = 0.05
# What the browser says when the document or its execution context went away under a script. These are page
# problems: the websocket is fine. (Timeouts of a single evaluation are the same kind of problem.)
DOCUMENT_CHANGED = (
    "Execution context was destroyed",
    "Cannot find context with specified id",
    "Cannot find default execution context",
    "Inspected target navigated or closed",
    "request deadline exceeded",
)
KEYS = {
    "return": ("Enter", 13),
    "escape": ("Escape", 27),
    "tab": ("Tab", 9),
    "delete": ("Backspace", 8),
    "left": ("ArrowLeft", 37),
    "right": ("ArrowRight", 39),
    "up": ("ArrowUp", 38),
    "down": ("ArrowDown", 40),
    "home": ("Home", 36),
    "end": ("End", 35),
    "space": (" ", 32),
}
MODIFIER_BITS = {"alt": 1, "control": 2, "command": 2 if sys.platform == "win32" else 4, "shift": 8}


class PageEvaluationError(CDPError):
    """A page/context script failure, distinct from losing the browser transport.

    The message is fixed on purpose: the page's own error text can contain page content. It is kept in `details`
    for a recorder that may keep content.
    """

    def __init__(self, details):
        self.details = details
        super().__init__("Page script evaluation failed; the page may have changed")


# The fixed scripts below are code plus JSON-encoded data, never model-authored JavaScript. The label an element is
# known by, and what counts as a credential field, are defined once and used by the observation and by every guard
# that checks an element is still what was observed.
LABEL = (
    "({0}.getAttribute('aria-label')||{0}.labels?.[0]?.innerText||{0}.getAttribute('placeholder')||{0}.innerText"
    "||{0}.getAttribute('title')||{0}.id||{0}.tagName).trim().slice(0,160)"
)
SECRET = "/password|one.?time|otp|cvv|card number|api[ _-]?key|secret|token/i"

SNAPSHOT = r"""(() => {
  if (!window.__glideNodes) window.__glideNodes = {ids:new WeakMap(),refs:new Map(),next:0};
  const s=window.__glideNodes;
  const id=el=>{if(!s.ids.has(el))s.ids.set(el,String(++s.next));const k=s.ids.get(el);s.refs.set(k,el);return k;};
  const visible=el=>{const r=el.getBoundingClientRect(),st=getComputedStyle(el);return r.width>0&&r.height>0&&st.display!=='none'&&st.visibility!=='hidden';};
  const label=el=>__LABEL__;
  const disclosure=el=>{
    if(el.tagName==='SUMMARY'&&el.parentElement?.tagName==='DETAILS')return {expanded:el.parentElement.open,controls:[id(el.parentElement)]};
    if(!['A','BUTTON'].includes(el.tagName)&&el.getAttribute('role')!=='button')return {};
    if(el.tagName==='BUTTON'&&el.form&&el.type!=='button')return {};
    const href=el.getAttribute('href');if(href&&!href.startsWith('#'))return {};
    const collapse=el.getAttribute('data-toggle')==='collapse'||el.getAttribute('data-bs-toggle')==='collapse';
    const expanded=el.getAttribute('aria-expanded');
    if(!collapse&&!['true','false'].includes(expanded))return {};
    let names=(el.getAttribute('aria-controls')||'').split(/\s+/).filter(Boolean);
    if(!names.length&&collapse){const target=el.getAttribute('data-bs-target')||el.getAttribute('data-target')||href||'';
      if(/^#[^\s#]+$/.test(target))names=[target.slice(1)];}
    const panels=names.map(name=>document.getElementById(name));
    if(!panels.length||panels.some(panel=>!panel||!panel.isConnected))return {};
    return {expanded:panels.every(visible),controls:panels.map(id)};
  };
  const items=[], containers=[];
  for(const el of document.querySelectorAll('button,a,input,textarea,select,summary,[role=button],[role=textbox],[role=searchbox],[role=combobox],[contenteditable=true],[role=status],[role=alert],[role=dialog]')) {
    if(items.length>=160) break; if(!visible(el))continue;
    const r=el.getBoundingClientRect(); if(r.top<0||r.bottom>innerHeight||r.left<0||r.right>innerWidth)continue;
    const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);if(hit!==el&&!el.contains(hit))continue;
    const secret=el.type==='password'||__SECRET__.test(label(el));
    items.push({id:id(el),label:label(el),role:el.getAttribute('role')||el.tagName.toLowerCase(),
      typeable:!el.readOnly&&el.matches('input:not([type=checkbox]):not([type=radio]):not([type=button]):not([type=submit]),textarea,[contenteditable=true]'),secret,
      enabled:!el.disabled&&el.getAttribute('aria-disabled')!=='true'&&!el.matches('[role=status],[role=alert],[role=dialog]'),
      value:secret?null:(el.value??(el.isContentEditable?el.innerText:null)),x:r.x+r.width/2,y:r.y+r.height/2,
      shortcuts:(el.getAttribute('aria-keyshortcuts')||'').split(/\s+/).filter(Boolean),
      search:el.type==='search'||el.getAttribute('role')==='searchbox'||el.getAttribute('enterkeyhint')==='search'||!!el.closest('[role=search],search'),
      href:el.closest('a[href]')?.href||'',...disclosure(el)});
  }
  const root=document.scrollingElement;
  if(root&&root.scrollHeight>root.clientHeight)containers.push({id:id(root),label:'page',position:root.scrollTop,maximum:root.scrollHeight-root.clientHeight});
  const walker=document.createTreeWalker(document.body||document.documentElement,NodeFilter.SHOW_ELEMENT);
  let el, count=0;while((el=walker.nextNode())&&count++<1500&&containers.length<32){
    if(el===root||!visible(el))continue;
    const style=getComputedStyle(el);if(!/auto|scroll/.test(style.overflowY)||el.scrollHeight<=el.clientHeight)continue;
    containers.push({id:id(el),label:label(el),position:el.scrollTop,maximum:el.scrollHeight-el.clientHeight});
  }
  const focused=document.activeElement,focus=focused?id(focused):'';
  if(focused&&!items.some(e=>e.id===focus)&&focused.matches('input,textarea,[contenteditable=true]')) {
    const secret=focused.type==='password'||__SECRET__.test(label(focused));
    items.push({id:focus,label:label(focused),role:'textbox',typeable:false,secret,enabled:false,
      value:secret?null:(focused.value??focused.innerText),x:0,y:0,shortcuts:[]});
  }
  const forms=[];
  for(const form of document.forms) {
    if(forms.length>=32)break;
    if(form.method.toLowerCase()!=='get')continue;
    const fields=[...form.elements];
    if(fields.some(e=>e.type==='password'||__SECRET__.test((e.name||'')+' '+label(e))))continue;
    const editable=fields.filter(e=>(e.tagName==='TEXTAREA'||(e.tagName==='INPUT'&&['text','search','email','url','tel','number'].includes(e.type)))&&e.name&&!e.disabled&&!e.readOnly);
    if(editable.length!==1||!items.some(i=>i.id===s.ids.get(editable[0])&&i.typeable&&i.enabled&&!i.secret))continue;
    const field=editable[0], values=[...new FormData(form)].filter(([k,v])=>k!==field.name&&typeof v==='string');
    if(values.length>32||values.some(([k,v])=>k.length>128||v.length>2048))continue;
    forms.push({id:id(form),field:id(field),action:form.action,parameter:field.name,values});
  }
  const media=[...document.querySelectorAll('audio,video')].slice(0,32).map(el=>({
    id:id(el),label:label(el),paused:el.paused,ended:el.ended,ready_state:el.readyState,current_time:el.currentTime}));
  const keep=new Set([...items.map(e=>e.id),...items.flatMap(e=>e.controls||[]),...containers.map(c=>c.id),...forms.map(f=>f.id),...media.map(m=>m.id),focus]);
  for(const [key,node] of s.refs)if(!node.isConnected||!keep.has(key))s.refs.delete(key);
  return {url:location.href,canonical_url:document.querySelector('link[rel~="canonical"][href]')?.href||'',document_id:String(performance.timeOrigin),items,containers,forms,media,focus,ready:document.readyState!=='loading'};
})()""".replace("__LABEL__", LABEL.format("el")).replace("__SECRET__", SECRET)
# After input, wait (bounded) for the page to paint its reaction before the next snapshot; a combobox gets longer.
SETTLE = (
    "(() => new Promise(resolve => {const e=window.__glideNodes?.refs.get(__FIELD__);"
    "const combo=e?.getAttribute('role')==='combobox';let frames=0,done=false;"
    "const finish=()=>{if(!done){done=true;resolve()}};setTimeout(finish,combo?200:50);"
    "const check=()=>{if(done)return;const roots=(e?.getAttribute('aria-controls')||'').split(/\\s+/).filter(Boolean)"
    ".map(id=>document.getElementById(id)).filter(Boolean);"
    "const options=(roots.length?roots:[document]).flatMap(r=>[...r.querySelectorAll('[role=option]')]);"
    "if(++frames>=2&&(!combo||options.some(o=>{const r=o.getBoundingClientRect();return r.width>0&&r.height>0&&r.top<innerHeight&&r.bottom>0})))finish();"
    "else requestAnimationFrame(check)};requestAnimationFrame(check)}))()"
)
DOCUMENT = "({url:location.href,document_id:String(performance.timeOrigin),ready:document.readyState!=='loading'})"
FORM_GUARD = (
    "(p=>{const f=window.__glideNodes?.refs.get(p.form),e=window.__glideNodes?.refs.get(p.field);"
    "if(!f?.isConnected||!e?.isConnected||e.form!==f||e.disabled||e.readOnly||f.method.toLowerCase()!=='get'||f.action!==p.action||e.name!==p.parameter)throw Error('changed form');"
    "const values=[...new FormData(f)].filter(([k,v])=>k!==e.name&&typeof v==='string');"
    "if(JSON.stringify(values)!==JSON.stringify(p.values))throw Error('changed values');})(__DETAILS__)"
)
CLICK_GEOMETRY = (
    "(() => {const e=window.__glideNodes?.refs.get(__TARGET__);"
    "if(!e?.isConnected||e.disabled||e.getAttribute('aria-disabled')==='true')throw Error('stale click');"
    "const label=" + LABEL.format("e") + ";"
    "if(label!==__EXPECTED__)throw Error('changed target');const r=e.getBoundingClientRect();"
    "return {x:r.x+r.width/2,y:r.y+r.height/2};})()"
)
CLICK_HIT = (
    "(() => {const e=window.__glideNodes?.refs.get(__TARGET__),h=document.elementFromPoint(__X__,__Y__);"
    "if(!e?.isConnected||e.disabled||e.getAttribute('aria-disabled')==='true'||(h!==e&&!e.contains(h)))throw Error('stale click');})()"
)
TYPE_GUARD = (
    "const label=" + LABEL.format("e") + ";"
    "if(label!==__EXPECTED__||e.disabled||e.type==='password'||e.getAttribute('aria-disabled')==='true'"
    "||" + SECRET + ".test(label))throw Error('unavailable');"
)


def fill(template: str, **values: str) -> str:
    """`template` with each __NAME__ replaced once by its value. Values (page labels, URLs) are never rescanned."""
    return re.sub(r"__([A-Z]+)__", lambda match: values[match.group(1)], template)


def on_element(target: str, body: str) -> str:
    """A script that runs `body` with `e` bound to the observed element, or fails when it is gone."""
    return (
        "(() => {const e=window.__glideNodes?.refs.get("
        + json.dumps(target)
        + ");if(!e?.isConnected)throw Error('stale target');"
        + body
        + "})()"
    )


class BrowserBackend:
    transport = "cdp"
    passive_inspection = True
    suppress_origin = False

    def __init__(self, endpoint, target=""):
        try:
            self.origin, self.port = browser_settings.loopback_origin(endpoint)
        except ValueError as error:
            raise DesktopError("CDP must be an explicitly configured loopback HTTP origin") from error
        self._initialize(target)

    def _initialize(self, target):
        """Common observed-page state, independent of browser/session transport."""
        self.owned_tab = target == "new"
        self.target = "" if self.owned_tab else target
        self.browser = self.page = None
        self.page_id = ""
        self.document = ""
        self.after_input = None
        self.navigation = None

    # -- connection ---------------------------------------------------------------------------------------------

    def targets(self):
        return {str(t["targetId"]): t for t in self.browser.call("Target.getTargets")["targetInfos"] if t.get("type") == "page"}

    def connect(self):
        if self.browser:
            return
        started = time.perf_counter()
        event("browser_connection_started", provider=self.transport, endpoint=self.origin)
        try:
            version = _get_json(self.origin + "/json/version", timeout=2)
            self.browser = Session(
                local_debugger_url(version["webSocketDebuggerUrl"], self.port),
                origin=self.origin,
                timeout=2,
                navigation_timeout=15,
                suppress_origin=self.suppress_origin,
            )
        except (OSError, ValueError, KeyError, TypeError, CDPError, websocket.WebSocketException) as error:
            event(
                "browser_connection_failed",
                provider=self.transport,
                endpoint=self.origin,
                elapsed_s=time.perf_counter() - started,
                exception=error,
            )
            raise BrowserConnectionError(self.transport, self.origin, error) from error
        event(
            "browser_connection_completed", provider=self.transport, endpoint=self.origin, elapsed_s=time.perf_counter() - started
        )

    def check(self):
        """Whether the provider can be reached: a passive connection, no tab opened and nothing sent to a page."""
        self.connect()

    def attach(self):
        if self.page_id != self.target:
            if self.page:
                self.page.close()
            attached = self.browser.call("Target.attachToTarget", {"targetId": self.target, "flatten": True})
            self.page = AttachedPage(self.browser, attached["sessionId"])
            self.page_id = self.target

    def connection_lost(self, error):
        """Whether `error` means the transport is gone, as opposed to a refused request or a page that changed."""
        return isinstance(error, OSError | websocket.WebSocketException) or str(error) == "websocket closed"

    # -- reads --------------------------------------------------------------------------------------------------

    def inspect_for(self, step, action=None):
        if step.effect not in {"url", "url_query", "tab_created", "tab_active", "tab_closed"}:
            return self.inspect()
        return self.inspect(controls=False)

    def read_page(self):
        from .reading import PAGE_SCRIPT, page_record

        checkpoint()
        observed = self.inspect()
        if not self.target or not observed.ready:
            raise DesktopError("The selected page is not ready to read")
        data = self.read_snapshot(PAGE_SCRIPT)
        checkpoint()
        if (
            not isinstance(data, dict)
            or data.get("url") != observed.url
            or self.target + ":" + str(data.get("document_id")) != observed.owner
        ):
            raise DesktopError("The page changed while reading; collect fresh evidence")
        return page_record(data)

    def read_snapshot(self, expression):
        """Evaluate a fixed read-only script, retrying when the document changed under it. Writes never come here."""
        for attempt in range(1, READ_ATTEMPTS + 1):
            checkpoint()
            try:
                return self.page.evaluate(expression)
            except PageEvaluationError as error:
                event(
                    "browser_snapshot_failed",
                    provider=self.transport,
                    attempt=attempt,
                    exception=error,
                    page_content={"exception_details": error.details},
                )
                if attempt == READ_ATTEMPTS:
                    raise DesktopError("Couldn't read the selected page. Check the page before retrying.") from error
                time.sleep(READ_BACKOFF_S)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def inspect(self, *, controls=True):
        checkpoint()
        try:
            self.connect()
            targets = self.targets()
            if not targets or (self.owned_tab and not self.target):
                self.target = ""
                return Observation(
                    "browser",
                    self.origin,
                    capabilities={"inspect", "tab_create", "navigate", "query_form"},
                    available_after_navigation=BROWSER_CAPABILITIES.copy(),
                )
            self._select_target(targets)
            self.attach()
            self._settle()
            data = self.read_snapshot(SNAPSHOT if controls else DOCUMENT)
            self.document = self.target + ":" + data["document_id"]
            return self._observation(data, targets)
        except (CDPError, OSError, websocket.WebSocketException) as error:
            if self.connection_lost(error):
                event("browser_inspection_failed", provider=self.transport, endpoint=self.origin, exception=error)
                raise BrowserConnectionError(self.transport, self.origin, error) from error
            # The browser answered, so the connection is not lost: it refused or could not complete this read.
            event("browser_inspection_rejected", provider=self.transport, endpoint=self.origin, exception=error)
            raise DesktopError("The browser could not read the selected page. Check the page before retrying.") from error

    def _select_target(self, targets):
        if not self.target:
            if len(targets) != 1:
                raise DesktopError(
                    "Select the browser tab explicitly (glide.toml [browser] target, or GLIDE_BROWSER_TARGET) "
                    "when the approved browser has multiple tabs"
                )
            self.target = next(iter(targets))
        if self.target not in targets:
            raise DesktopError("The selected browser tab disappeared; choose a fresh target")

    def _settle(self):
        """After an input, let the page react before reading it. Read-only; the fresh snapshot owns readiness."""
        if not self.after_input:
            return
        action, self.after_input = self.after_input, None
        try:
            self.page.evaluate(fill(SETTLE, FIELD=json.dumps(action.target)), await_promise=True)
        except PageEvaluationError as error:
            # The page can navigate during this check; the snapshot that follows is retried like any read.
            event(
                "browser_settle_interrupted",
                provider=self.transport,
                exception=error,
                page_content={"exception_details": error.details},
            )

    def _observation(self, data, targets):
        navigation = None
        if self.navigation and self.navigation.tab == self.target:
            try:
                frame = self.page.call("Page.getFrameTree").get("frameTree", {}).get("frame", {})
            except CDPError:
                frame = {}  # A provider without loader evidence retains exact-URL verification.
            if (
                frame.get("id") == self.navigation.frame
                and frame.get("loaderId") == self.navigation.loader
                and frame.get("url") == data["url"]
            ):
                navigation = replace(self.navigation, owner=self.document)
        elements = {
            e["id"]: Element(
                e["id"],
                e["label"],
                e["role"],
                e["value"],
                e["typeable"],
                e["secret"] or looks_credential(e["label"]),
                e["enabled"],
                point=(e["x"], e["y"]),
                shortcuts=tuple(e.get("shortcuts", ())),
                search=bool(e.get("search", False)),
                href=e.get("href", "") if safe_url(e.get("href", "")) else "",
                expanded=e.get("expanded"),
                controls=tuple(e.get("controls", ())),
            )
            for e in data.get("items", [])
        }
        forms = {}
        if data.get("forms"):
            from .query import QueryForm

            forms = {
                f["id"]: QueryForm(f["id"], f["field"], f["action"], f["parameter"], tuple(tuple(v) for v in f["values"]))
                for f in data["forms"]
            }
        return Observation(
            "browser",
            self.document,
            data["url"],
            self.target,
            {k: v.get("url", "") for k, v in targets.items()},
            elements,
            {c["id"]: Container(**c) for c in data.get("containers", [])},
            data.get("focus", ""),
            BROWSER_CAPABILITIES.copy(),
            data["ready"],
            forms=forms,
            media={m["id"]: Media(**m) for m in data.get("media", [])},
            navigation=navigation,
            canonical_url=data.get("canonical_url", "") if safe_url(data.get("canonical_url", "")) else "",
        )

    # -- writes: never retried ----------------------------------------------------------------------------------

    def paired(self, method, down, up):
        """A press and its release. The release is cleanup and still goes out after a stop."""
        checkpoint()
        try:
            self.page.call(method, down)
        finally:
            with controlled(None):
                self.page.call(method, up)

    def key(self, key, modifiers):
        name, code = KEYS[key] if key in KEYS else (key, ord(key.upper()))
        flags = sum({MODIFIER_BITS[m] for m in modifiers})
        common = {"key": name, "windowsVirtualKeyCode": code, "modifiers": flags}
        down = {**common, "type": "keyDown"}
        if key == "a" and ("command" in modifiers or "control" in modifiers):
            down["commands"] = ["selectAll"]
        self.paired("Input.dispatchKeyEvent", down, {**common, "type": "keyUp"})

    def select_all(self):
        self.key("a", ("command",))

    def navigate(self, url):
        try:
            result = self.page.call("Page.navigate", {"url": url})
        except CDPError as error:
            if str(error) == "Page.navigate: request deadline exceeded" or str(error).startswith("Playwright request timed out;"):
                event("browser_navigation_timed_out", provider=self.transport, exception=error)
                raise DesktopError("Timed out while opening the requested page. Check the browser before retrying.") from error
            raise
        if result.get("errorText"):
            raise DesktopError("Browser navigation failed")
        if result.get("loaderId") and result.get("frameId"):
            self.navigation = Navigation(url, self.target, str(result["frameId"]), str(result["loaderId"]))
            return self.navigation.loader
        return ""

    def execute(self, action, observed):
        validate(action, observed)
        checkpoint()
        if action.kind == "inspect":
            return ""
        self.navigation = None
        if action.kind == "tab_create" or (action.kind == "navigate" and not self.target):
            # Capture the requested navigation's loader even when the task owns a new tab.
            result = self.browser.call("Target.createTarget", {"url": "about:blank"})
            self.target = str(result["targetId"])
            checkpoint()  # Creation is already an effect; Stop must prevent the following navigation.
            self.attach()
            checkpoint()
            receipt = self.navigate(action.value)
            return self.target if action.kind == "tab_create" else receipt
        if action.kind == "tab_switch":
            self.browser.call("Target.activateTarget", {"targetId": action.target})
            self.target = action.target
            return self.target
        if action.kind == "tab_close":
            self.browser.call("Target.closeTarget", {"targetId": action.target})
            if self.target == action.target:
                self.target = next(iter(self.targets()), "")
            return action.target
        if action.kind == "navigate":
            return self._navigate_action(action, observed)
        if action.kind == "key":
            self.key(action.value, action.modifiers)
        elif action.kind == "click":
            self._click(action, observed)
        elif action.kind == "scroll":
            direction = 1 if action.value == "down" else -1
            self.page.evaluate(on_element(action.target, f"e.scrollBy(0,{direction}*Math.max(1,e.clientHeight*0.8));"))
        elif action.kind == "type":
            self._type(action, observed)
        if action.kind in {"click", "type", "key", "scroll"}:
            self.after_input = action
        return ""

    def _navigate_action(self, action, observed):
        if action.parameter_source.startswith("form:"):
            form_id = action.parameter_source.removeprefix("form:")
            form = observed.forms[form_id]
            details = json.dumps(
                {"form": form_id, "field": form.field, "action": form.action, "parameter": form.parameter, "values": form.values}
            )
            self.page.evaluate(fill(FORM_GUARD, DETAILS=details))
            checkpoint()
        if action.target and action.target != self.target:
            raise DesktopError("Navigation target differs from the selected browser session")
        return self.navigate(action.value)

    def _click(self, action, observed):
        target = json.dumps(action.target)
        geometry = self.page.evaluate(
            fill(CLICK_GEOMETRY, TARGET=target, EXPECTED=json.dumps(observed.elements[action.target].label))
        )
        x, y = geometry["x"], geometry["y"]
        self.page.evaluate(fill(CLICK_HIT, TARGET=target, X=str(x), Y=str(y)))
        checkpoint()
        common = {"x": x, "y": y, "button": "left", "clickCount": 1}
        self.paired("Input.dispatchMouseEvent", {**common, "type": "mousePressed"}, {**common, "type": "mouseReleased"})

    def _type(self, action, observed):
        guard = fill(TYPE_GUARD, EXPECTED=json.dumps(observed.elements[action.target].label))
        self.page.evaluate(on_element(action.target, guard + "e.focus();"))
        checkpoint()
        self.select_all()
        checkpoint()
        self.key("delete", ())
        checkpoint()
        self.page.evaluate(on_element(action.target, guard + "if(e!==document.activeElement)throw Error('focus changed');"))
        checkpoint()
        self.page.call("Input.insertText", {"text": action.value})

    def close(self):
        for session in (self.page, self.browser):
            if session:
                session.close()


class AttachedPage:
    """A flattened page session sharing the existing browser websocket."""

    def __init__(self, browser, session_id):
        self.browser, self.session_id = browser, session_id

    def call(self, method, params=None):
        return self.browser.call(method, params, session_id=self.session_id)

    def evaluate(self, expression, *, await_promise=False):
        try:
            result = self.call(
                "Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": await_promise}
            )
        except CDPError as error:
            if any(marker in str(error) for marker in DOCUMENT_CHANGED):
                raise PageEvaluationError(
                    {"text": "Execution context changed during evaluation", "protocol_error": str(error)}
                ) from error
            raise
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            event("browser_evaluation_failed", page_content={"exception_details": details})
            raise PageEvaluationError(details)
        return result.get("result", {}).get("value")

    def close(self):
        with contextlib.suppress(Exception), controlled(None):
            self.browser.call("Target.detachFromTarget", {"sessionId": self.session_id})
