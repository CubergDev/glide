"""Obscura's headless CDP transport. Attach-only: the user starts its loopback server, Glide never launches it."""

import json

from ..control import checkpoint
from .contracts import UnsupportedCapability, validate
from .dom import BrowserBackend


class ObscuraBackend(BrowserBackend):
    transport = "obscura"
    # Obscura refuses browser-origin WebSockets. Its native clients omit Origin;
    # the loopback host/port checks remain identical to the CDP provider.
    suppress_origin = True

    def select_all(self):
        # Obscura does not implement Chromium's selectAll editor command. Select the
        # focused range before the existing Backspace/insertText pair instead.
        checkpoint()
        self.page.evaluate(
            "(() => {const e=document.activeElement;"
            "if(!e||e.disabled||e.readOnly||e.type==='password')throw Error('unavailable');"
            "if(e.setSelectionRange)e.setSelectionRange(0,e.value.length);"
            "else if(e.isContentEditable){const r=document.createRange();r.selectNodeContents(e);"
            "const s=getSelection();s.removeAllRanges();s.addRange(r)}"
            "else throw Error('unsupported editable selection');})()"
        )

    def execute(self, action, observed):
        validate(action, observed)
        # Obscura accepts many key events but implements only a subset of default
        # editor behavior. Reject unknown shortcuts instead of reporting success.
        if action.kind == "key" and (action.modifiers or action.value not in {"return", "delete", "escape"}):
            raise UnsupportedCapability(["Obscura keyboard shortcut: " + "+".join((*action.modifiers, action.value))])
        if action.kind == "type":
            checkpoint()
            supported = self.page.evaluate(
                "(() => {const e=window.__glideNodes?.refs.get(" + json.dumps(action.target) + ");"
                "return !!e?.isConnected&&['INPUT','TEXTAREA'].includes(e.tagName);})()"
            )
            if not supported:
                raise UnsupportedCapability(["Obscura typing outside input/textarea"])
        return super().execute(action, observed)
