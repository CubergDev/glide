"""Provider selection and session preflight. Nothing here launches a browser, and no action is taken here.

`make_backend` builds the backend the configuration names (`browser_settings`): native, cdp, obscura or playwright.
A provider that is not installed, not enabled or not reachable fails with a message that says what to do, before any
action. There is no automatic fallback, with one explicit exception: a `[browser] fallback` list in glide.toml. Then,
and only before anything has been done, an unavailable provider is replaced by the next one on the list, and the
switch is a `SwitchEvent`, a recorded event and, under a run's control, a line on its channel, never silent. Once a backend is returned it is never swapped for
another mid-run, so an action whose outcome is unknown cannot be replayed on a different provider. The native desktop
is never a fallback: it takes over the machine, so it has to be selected as the provider.
"""

from ...providers.chain import SwitchEvent
from .. import browser_settings
from ..control import current_control
from ..diagnostics import event
from ..models import DesktopError


def _build(provider, settings, browser, act):
    if provider == "native":
        from .native import NativeBackend

        return NativeBackend(browser)
    connection = settings.connection(provider)
    if provider == "cdp":
        from .dom import BrowserBackend

        return BrowserBackend(connection.endpoint, connection.target)
    if provider == "obscura":
        from .obscura import ObscuraBackend

        return ObscuraBackend(connection.endpoint, connection.target)
    from .playwright_cli import PlaywrightBackend

    return PlaywrightBackend(connection.session, connection.target, allow_actions=act, cli=settings.cli)


def make_backend(browser="", *, act=False, on_switch=None):
    """The backend for the configured provider. `browser` names the native desktop's browser, `act` whether the run
    may take actions (the Playwright CLI cannot inspect passively, so it needs it). `on_switch` hears each fallback."""
    try:
        settings = browser_settings.current()
    except ValueError as error:
        raise DesktopError(str(error)) from error
    if browser == "":
        from ..config import browser as configured_browser

        browser = configured_browser()
    event("browser_provider_selected", provider=settings.provider, source=settings.source, fallback=list(settings.fallback))
    chain = settings.chain
    for position, provider in enumerate(chain):
        backend = _build(provider, settings, browser, act)
        if len(chain) == 1:
            return backend  # no fallback configured: the first observation reports what is wrong
        try:
            backend.check()
        except DesktopError as error:
            backend.close()
            if position == len(chain) - 1:
                raise
            switch = SwitchEvent("browser", provider, chain[position + 1], type(error).__name__, str(error))
            event("browser_provider_switched", provider=provider, to=switch.to_slot, kind=switch.kind)
            if control := current_control():  # said on the run's own channel, so it is never only in memory
                control.event("switch", f"fallback: browser {switch.from_slot} -> {switch.to_slot} ({switch.kind})")
            if on_switch:
                on_switch(switch)
            continue
        return backend
    raise AssertionError("unreachable: the last provider returns or raises")


def prepare(act, *, cancelled=lambda: False, on_switch=None):
    """Check the selected executor before admitting a typed task."""
    if cancelled():
        raise KeyboardInterrupt
    backend = make_backend(act=act, on_switch=on_switch)
    try:
        if backend.transport == "native":
            from ..desktop_access import prepare_desktop

            prepare_desktop(act, cancelled=cancelled)
            return
        backend.inspect()
        if cancelled():
            raise KeyboardInterrupt
    finally:
        backend.close()


def prepare_voice(act, *, cancelled=lambda: False):
    """Prepare voice permissions without requiring a live browser to collect dictation.

    Browser connectivity is checked by the runner's initial observation once a
    command is submitted. An offline browser must not close the recording session.
    """
    if cancelled():
        raise KeyboardInterrupt
    provider = browser_settings.current().provider
    if provider == "native":
        prepare(act, cancelled=cancelled)
    else:
        from .. import diagnostics

        diagnostics.event("browser_check_deferred", provider=provider, stage="task_execution")
    if cancelled():
        raise KeyboardInterrupt
