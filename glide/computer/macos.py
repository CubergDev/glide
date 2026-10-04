"""macOS adapter: synthetic input, app control, screen capture, and the focused accessibility element.

This is the only module that touches Quartz, ApplicationServices, AppleScript, or Vision OCR.
windows.py provides the same functions for Windows; platform_adapter.py picks one. The bounded tree
walk itself lives in ax_walk.py, shared by both.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

import ApplicationServices as AS
import Quartz
from ocrmac import ocrmac
from PIL import Image

from .ax_walk import AX_PRESS, AxAttrs, Frame, walk_actionable
from .config import ABORT_CORNER_PX
from .models import Abort, AxNode, Box, DesktopError, DesktopPermissionError, Field
from .point_types import PointTarget, point_box

KEYCODES = {"return": 36, "tab": 48, "escape": 53, "a": 0, "delete": 51, "[": 33}
SHORTCUT_CODES = {
    **KEYCODES,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "home": 115,
    "end": 119,
    "space": 49,
    **dict(
        zip(
            "abcdefghijklmnopqrstuvwxyz",
            (0, 11, 8, 2, 14, 3, 5, 4, 34, 38, 40, 37, 46, 45, 31, 35, 12, 15, 1, 17, 32, 9, 13, 7, 16, 6),
            strict=True,
        )
    ),
}
MIN_WINDOW_SIDE_PT = 50.0  # anything smaller is a palette or a shadow, not the window being worked in

# ------------------------------------------------------------------ escape hatch


def mouse_location() -> tuple[float, float]:
    loc = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
    return loc.x, loc.y


def check_abort() -> None:
    from .control import checkpoint

    checkpoint()
    x, y = mouse_location()
    if x <= ABORT_CORNER_PX and y <= ABORT_CORNER_PX:
        raise Abort("mouse in top-left corner")


def abort_hint() -> str:
    return "Ctrl-C, or slam the mouse into the top-left corner"


def sleep_watching(seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        check_abort()
        time.sleep(0.1)


def accessibility_trusted() -> bool:
    return bool(AS.AXIsProcessTrusted())


def screen_capture_trusted() -> bool:
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def request_permissions(*, accessibility: bool = False, screen_capture: bool = False) -> None:
    if accessibility:
        AS.AXIsProcessTrustedWithOptions({AS.kAXTrustedCheckOptionPrompt: True})
    if screen_capture:
        Quartz.CGRequestScreenCaptureAccess()


# ------------------------------------------------------------------ input


def _post(event) -> None:
    Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
    time.sleep(0.04)


def _down_then_up(event: Callable[[bool], object]) -> None:
    """Post the down event, then the up event even when the down is interrupted, so nothing stays held."""
    try:
        _post(event(True))
    finally:
        _post(event(False))


def click_at(point: tuple[float, float]) -> None:
    # Check before moving: the synthetic move would otherwise take the pointer out of the abort corner.
    check_abort()
    _post(Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, point, Quartz.kCGMouseButtonLeft))
    check_abort()
    kinds = {True: Quartz.kCGEventLeftMouseDown, False: Quartz.kCGEventLeftMouseUp}
    _down_then_up(lambda down: Quartz.CGEventCreateMouseEvent(None, kinds[down], point, Quartz.kCGMouseButtonLeft))


def press(key: str, command: bool = False) -> None:
    check_abort()
    code = KEYCODES[key]

    def event(down: bool):
        e = Quartz.CGEventCreateKeyboardEvent(None, code, down)
        if command:
            Quartz.CGEventSetFlags(e, Quartz.kCGEventFlagMaskCommand)
        return e

    _down_then_up(event)


def _unicode_key(ch: str, down: bool):
    event = Quartz.CGEventCreateKeyboardEvent(None, 0, down)
    Quartz.CGEventKeyboardSetUnicodeString(event, len(ch), ch)
    return event


def type_text(text: str) -> None:
    """One character at a time, checking the abort corner before each."""
    for ch in text:
        check_abort()
        _down_then_up(partial(_unicode_key, ch))


def clear_field() -> None:
    press("a", command=True)
    press("delete")


def scroll(lines: int) -> None:
    """Scroll events go to the view under the cursor, so park it over the frontmost window first."""
    center = frontmost_window_center()
    check_abort()
    if center is not None:
        _post(Quartz.CGEventCreateMouseEvent(None, Quartz.kCGEventMouseMoved, center, Quartz.kCGMouseButtonLeft))
    check_abort()
    _post(Quartz.CGEventCreateScrollWheelEvent(None, Quartz.kCGScrollEventUnitLine, 1, lines))


# ------------------------------------------------------------------ apps and windows


def osascript(script: str) -> str:
    return subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True).stdout.strip()


def frontmost_app() -> str:
    return frontmost_app_and_pid()[0]


def frontmost_app_and_pid() -> tuple[str, int]:
    """Observe the foreground process without starting an AppleScript subprocess."""
    from AppKit import NSWorkspace

    app = NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        raise DesktopError("Frontmost application is unavailable")
    return str(app.localizedName() or ""), int(app.processIdentifier())


def frontmost_pid() -> int:
    return frontmost_app_and_pid()[1]


def activate(app: str, timeout: float = 3.0) -> bool:
    """Bring an app to the front and confirm it got there."""
    check_abort()
    osascript(f'tell application "{app}" to activate')
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        check_abort()
        if frontmost_app() == app:
            return True
        time.sleep(0.1)
    check_abort()
    osascript(f'tell application "System Events" to set frontmost of process "{app}" to true')
    time.sleep(0.3)
    return frontmost_app() == app


def open_url(browser: str, url: str) -> bool:
    check_abort()
    osascript(f'tell application "{browser}" to open location "{url}"')
    return activate(browser)


def browser_url(browser: str) -> str | None:
    try:
        return osascript(f'tell application "{browser}" to get URL of active tab of front window') or None
    except subprocess.CalledProcessError:
        return None


def open_path(path: Path, as_text: bool = False) -> None:
    """Show a file to the user; `as_text` opens it in the default text editor."""
    check_abort()
    subprocess.run(["open", *(["-t"] if as_text else []), str(path)], check=False)


def frontmost_window_bounds(pid: int | None = None) -> tuple[float, float, float, float] | None:
    """The frontmost app's topmost on-screen window as x, y, w, h in points. Pure Quartz, no AX needed.

    Pass the pid when the caller already has it; looking it up costs an AppleScript round trip.
    """
    pid = frontmost_pid() if pid is None else pid
    options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
    for window in Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []:
        if window.get("kCGWindowOwnerPID") == pid and window.get("kCGWindowLayer") == 0:
            b = window["kCGWindowBounds"]
            if b["Width"] > MIN_WINDOW_SIDE_PT and b["Height"] > MIN_WINDOW_SIDE_PT:
                return float(b["X"]), float(b["Y"]), float(b["Width"]), float(b["Height"])
    return None


def frontmost_window_center(pid: int | None = None) -> tuple[float, float] | None:
    """Center of the frontmost app's topmost on-screen window, in points."""
    bounds = frontmost_window_bounds(pid)
    if bounds is None:
        return None
    x, y, w, h = bounds
    return x + w / 2, y + h / 2


# ------------------------------------------------------------------ capture and accessibility


def screenshot() -> Image.Image:
    if not screen_capture_trusted():
        raise DesktopPermissionError(
            "Screen Recording access is missing for the application launching this session. "
            "Enable it in macOS Privacy & Security; a grant for Codex does not grant Terminal access."
        )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "screen.png"
        try:
            subprocess.run(["screencapture", "-x", "-D", "1", str(path)], check=True, capture_output=True)
            with Image.open(path) as image:
                return image.convert("RGB")
        except (subprocess.CalledProcessError, OSError) as error:
            raise DesktopError(
                "Screen capture failed. Check Screen Recording access for the launching application "
                "and that the display is available and unlocked. No task result was verified."
            ) from error


def display_scale(image: Image.Image) -> float:
    points_wide = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID()).size.width
    return image.width / points_wide


def point_target(point: tuple[float, float]) -> PointTarget | None:
    """Read the exact AX element under the pointer, without focusing or pressing it."""
    system = AS.AXUIElementCreateSystemWide()
    AS.AXUIElementSetMessagingTimeout(system, AX_MESSAGE_TIMEOUT)
    error, element = AS.AXUIElementCopyElementAtPosition(system, *point, None)
    if error:
        raise RuntimeError("Accessibility hit test failed; check permissions and point again.")
    if element is None:
        return None
    role = str(_ax_attr(element, AS.kAXRoleAttribute) or "")
    protected = role == "AXSecureTextField" or _ax_attr(element, "AXSubrole") == "AXSecureTextField"
    if protected:
        return PointTarget(role, protected=True)  # decided before any content is read
    value = _ax_attr(element, AS.kAXValueAttribute)
    return PointTarget(
        role=role,
        label=_ax_label(element),
        value=value if isinstance(value, str) else "",
        help=str(_ax_attr(element, "AXHelp") or ""),
    )


def point_region(point: tuple[float, float], radius: float) -> tuple[Image.Image, Box]:
    """Capture just a primary-display region; the temporary file is deleted on every path."""
    bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
    box = point_box(point, radius, (0.0, 0.0, float(bounds.size.width), float(bounds.size.height)))
    left, top, right, bottom = box
    region = f"{left},{top},{right - left},{bottom - top}"
    with tempfile.TemporaryDirectory(prefix="glide-point-") as folder:
        path = Path(folder) / "region.png"
        subprocess.run(["screencapture", "-x", f"-R{region}", str(path)], check=True, capture_output=True, timeout=10)
        with Image.open(path) as image:
            return image.convert("RGB"), box


def recognize_text(image: Image.Image) -> list[tuple[str, float, tuple[float, float, float, float]]]:
    """Vision OCR lines as text, confidence, and a box in the image's own pixels."""
    return ocrmac.OCR(image, recognition_level="accurate").recognize(px=True)


def _ax_attr(element, name: str):
    """One attribute, or None. A dead or hostile element raises from the bridge; that is a miss, not a crash."""
    try:
        err, value = AS.AXUIElementCopyAttributeValue(element, name, None)
    except Exception:
        return None
    return value if err == 0 else None


def focused_field() -> Field | None:
    system = AS.AXUIElementCreateSystemWide()
    element = _ax_attr(system, AS.kAXFocusedUIElementAttribute)
    if element is None:
        return None
    x = y = w = h = 0.0
    pos = _ax_attr(element, AS.kAXPositionAttribute)
    size = _ax_attr(element, AS.kAXSizeAttribute)
    if pos is not None and size is not None:
        _, pt = AS.AXValueGetValue(pos, AS.kAXValueCGPointType, None)
        _, sz = AS.AXValueGetValue(size, AS.kAXValueCGSizeType, None)
        x, y, w, h = pt.x, pt.y, sz.width, sz.height
    value = _ax_attr(element, AS.kAXValueAttribute)
    label = _ax_attr(element, AS.kAXTitleAttribute) or _ax_attr(element, AS.kAXDescriptionAttribute) or ""
    return Field(
        role=str(_ax_attr(element, AS.kAXRoleAttribute) or ""),
        label=str(label),
        placeholder=str(_ax_attr(element, AS.kAXPlaceholderValueAttribute) or ""),
        value=value if isinstance(value, str) else "",
        x=x,
        y=y,
        w=w,
        h=h,
        ref=element,
    )


# ------------------------------------------------------------------ acting on an element

# An element accepts these directly, so a press lands on the control the app declared rather than
# on whatever pixel happens to sit at its center. Every one of them is best effort: the element may
# be dead, the app may refuse, and the bridge raises on both. False means "use synthetic input".


def ax_press(ref) -> bool:
    """Send AXPress to an element."""
    check_abort()
    try:
        return AS.AXUIElementPerformAction(ref, AX_PRESS) == 0
    except Exception:
        return False


def ax_focus(ref) -> bool:
    """Give an element the keyboard focus."""
    check_abort()
    try:
        return AS.AXUIElementSetAttributeValue(ref, AS.kAXFocusedAttribute, True) == 0
    except Exception:
        return False


def ax_set_value(ref, text: str) -> bool:
    """Write an element's value. A read-only or unwilling element reports an error."""
    check_abort()
    try:
        return AS.AXUIElementSetAttributeValue(ref, AS.kAXValueAttribute, text) == 0
    except Exception:
        return False


def ax_value(ref) -> str | None:
    """An element's value, when it has a textual one."""
    value = _ax_attr(ref, AS.kAXValueAttribute)
    return value if isinstance(value, str) else None


# ------------------------------------------------------------------ actionable elements

AX_MESSAGE_TIMEOUT = 0.2
AX_VALUE_CHARS = 120


def _ax_children(element) -> list:
    return list(_ax_attr(element, AS.kAXChildrenAttribute) or [])


def _ax_label(element) -> str:
    """AXTitle on AppKit, AXDescription on web and Electron, a short AXValue as a last resort."""
    for name in (AS.kAXTitleAttribute, AS.kAXDescriptionAttribute):
        text = _ax_attr(element, name)
        if isinstance(text, str) and text.strip():
            return " ".join(text.split())
    value = _ax_attr(element, AS.kAXValueAttribute)
    if isinstance(value, str) and 0 < len(value.strip()) <= AX_VALUE_CHARS:
        return " ".join(value.split())
    return ""


def _ax_frame(element) -> Frame | None:
    pos = _ax_attr(element, AS.kAXPositionAttribute)
    size = _ax_attr(element, AS.kAXSizeAttribute)
    if pos is None or size is None:
        return None
    ok_pos, pt = AS.AXValueGetValue(pos, AS.kAXValueCGPointType, None)
    ok_size, sz = AS.AXValueGetValue(size, AS.kAXValueCGSizeType, None)
    if not (ok_pos and ok_size):
        return None
    return float(pt.x), float(pt.y), float(sz.width), float(sz.height)


def _ax_attrs(element) -> AxAttrs:
    return AxAttrs(str(_ax_attr(element, AS.kAXRoleAttribute) or ""), _ax_label(element), _ax_frame(element))


def _ax_actions(element) -> list[str]:
    if _ax_attr(element, AS.kAXEnabledAttribute) is False:
        return ["AXDisabled"]
    try:
        err, names = AS.AXUIElementCopyActionNames(element, None)
    except Exception:
        return []
    return [str(n) for n in names] if err == 0 and names else []


def actionable_elements(
    pid: int, display_w_pt: float, display_h_pt: float, *, focused_only: bool = False
) -> tuple[list[AxNode], list[AxNode], bool]:
    """Labelled controls of one process: the on-screen ones in points, the pressable off-screen ones,
    and whether a cap cut the walk short."""
    app = AS.AXUIElementCreateApplication(pid)
    AS.AXUIElementSetMessagingTimeout(app, AX_MESSAGE_TIMEOUT)
    if focused_only:
        roots = [_ax_attr(app, AS.kAXFocusedWindowAttribute) or app]
        menu = _ax_attr(app, AS.kAXMenuBarAttribute)
        if menu:
            roots.append(menu)

        def visible_menu_attrs(el):
            attrs = _ax_attrs(el)
            if attrs.role == "AXMenu" and attrs.frame and min(attrs.frame[2:]) > 0:
                return attrs._replace(role="AXGroup")
            return attrs

        results = [
            walk_actionable(root, _ax_children, visible_menu_attrs, _ax_actions, display_w_pt, display_h_pt) for root in roots
        ]
        return (
            [n for found, _, _ in results for n in found],
            [n for _, offscreen, _ in results for n in offscreen],
            any(capped for _, _, capped in results),
        )
    return walk_actionable(app, _ax_children, _ax_attrs, _ax_actions, display_w_pt, display_h_pt)


def _browser_jxa(browser, body):
    if browser not in {"Google Chrome", "Brave Browser", "Microsoft Edge", "Chromium"}:
        raise DesktopError("Stable tab identities are unavailable for this browser; configure an approved CDP connection")
    script = f"const app=Application({json.dumps(browser)}); " + body
    try:
        result = subprocess.run(
            ["osascript", "-l", "JavaScript", "-e", script], capture_output=True, text=True, check=True, timeout=5
        )
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, ValueError) as error:
        raise DesktopError("Browser tab observation or operation failed") from error


def execution_tabs(browser: str) -> dict:
    return _browser_jxa(
        browser,
        """
      let tabs={}, active='', ready=true;
      if (app.running()) {
        const windows=app.windows();
        if(windows.length) {active=String(windows[0].activeTab().id());ready=!windows[0].activeTab().loading();}
        for(const w of windows) for(const t of w.tabs()) tabs[String(t.id())]=String(t.url());
      }
      JSON.stringify({tabs,active,ready});
    """,
    )


def execution_tab(browser: str, kind: str, tab_id: str, url: str) -> str:
    from .execution.contracts import safe_url

    check_abort()
    if kind not in {"navigate", "tab_create", "tab_switch", "tab_close"} or (
        kind in {"navigate", "tab_create"} and not safe_url(url)
    ):
        raise DesktopError("Invalid tab operation")
    args = json.dumps({"kind": kind, "id": tab_id, "url": url})
    return str(
        _browser_jxa(
            browser,
            """
      const p=ARGS; let found=null, window=null, index=0;
      if(!app.running()) { if(p.kind!=='navigate' && p.kind!=='tab_create') throw Error('browser absent'); app.launch(); }
      for(const w of app.windows()) { const ts=w.tabs(); for(let i=0;i<ts.length;i++) if(String(ts[i].id())===p.id) {found=ts[i];window=w;index=i;} }
      let result='';
      if(p.kind==='tab_create' || (p.kind==='navigate' && !p.id)) {
        let w=app.windows()[0]; if(!w) {w=app.Window();app.windows.push(w);}
        const t=app.Tab({url:p.url}); w.tabs.push(t); w.activeTabIndex=w.tabs.length;w.index=1;result=String(w.activeTab().id());
      } else {
        if(!found) throw Error('tab absent'); result=p.id;
        if(p.kind==='navigate') {found.url=p.url;window.activeTabIndex=index+1;window.index=1;}
        if(p.kind==='tab_switch') {window.activeTabIndex=index+1;window.index=1;}
        if(p.kind==='tab_close') found.close();
      }
      app.activate(); JSON.stringify(result);
    """.replace("ARGS", args),
        )
    )


def execution_shortcut(key: str, modifiers: tuple[str, ...]) -> None:
    from .execution.contracts import MODIFIERS

    if key not in SHORTCUT_CODES or not set(modifiers) <= MODIFIERS:
        raise DesktopError("Invalid key or shortcut")
    check_abort()
    flags = {
        "command": Quartz.kCGEventFlagMaskCommand,
        "control": Quartz.kCGEventFlagMaskControl,
        "alt": Quartz.kCGEventFlagMaskAlternate,
        "shift": Quartz.kCGEventFlagMaskShift,
    }

    def event(down):
        e = Quartz.CGEventCreateKeyboardEvent(None, SHORTCUT_CODES[key], down)
        Quartz.CGEventSetFlags(e, sum(flags[m] for m in modifiers))
        return e

    _down_then_up(event)


def execution_scrolls(pid: int) -> list[dict]:
    root = AS.AXUIElementCreateApplication(pid)
    AS.AXUIElementSetMessagingTimeout(root, AX_MESSAGE_TIMEOUT)
    root = _ax_attr(root, AS.kAXFocusedWindowAttribute) or root
    pending, result, seen = [root], [], set()
    for _ in range(512):
        if not pending:
            break
        el = pending.pop(0)
        if id(el) in seen:
            continue
        seen.add(id(el))
        role = str(_ax_attr(el, AS.kAXRoleAttribute) or "")
        if role == "AXScrollArea":
            bar = _ax_attr(el, "AXVerticalScrollBar")
            value = _ax_attr(bar, AS.kAXValueAttribute) if bar else None
            frame = _ax_frame(el)
            if isinstance(value, int | float) and frame:
                result.append(
                    {
                        "id": str(frame),
                        "label": _ax_label(el) or "scroll area",
                        "position": float(value),
                        "maximum": 1.0,
                        "ref": bar,
                    }
                )
        pending.extend(_ax_children(el)[:100])
    return result[:32]


def execution_labels(pid: int) -> list[dict]:
    from .control import checkpoint

    root = AS.AXUIElementCreateApplication(pid)
    AS.AXUIElementSetMessagingTimeout(root, AX_MESSAGE_TIMEOUT)
    root = _ax_attr(root, AS.kAXFocusedWindowAttribute) or root
    pending, result = [root], []
    deadline = time.monotonic() + 0.15
    for _ in range(512):
        if not pending or len(result) >= 80 or time.monotonic() >= deadline:
            break
        checkpoint()
        el = pending.pop(0)
        if str(_ax_attr(el, AS.kAXRoleAttribute) or "") == "AXStaticText":
            frame, label = _ax_frame(el), _ax_label(el)
            if frame and min(frame[2:]) > 0 and label:
                result.append({"id": str(frame), "label": label[:300]})
        pending.extend(_ax_children(el)[:100])
    return result


def execution_scroll(ref, direction: str) -> None:
    if direction not in {"up", "down"}:
        raise DesktopError("Invalid scroll direction")
    check_abort()
    value = _ax_attr(ref, AS.kAXValueAttribute)
    if not isinstance(value, int | float):
        raise DesktopError("Scroll position is unavailable")
    target = min(1.0, max(0.0, value + (0.1 if direction == "down" else -0.1)))
    if AS.AXUIElementSetAttributeValue(ref, AS.kAXValueAttribute, target) != 0:
        raise DesktopError("Targeted scroll was refused")
