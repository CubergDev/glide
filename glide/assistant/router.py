"""Compatibility shim: the one stop list moved to `glide.routing.stop` (docs/ROUTER.md, decision D9).

The router that lived here is gone (`glide.routing` decides who owns a request) and the answer prompt moved to
`glide.assistant.answer`. These three names stay only because `glide/speech/` still imports them from here; change those
imports to `glide.routing.stop` and delete this file.
"""

from ..routing.stop import is_stop, normalize, stop_phrases

__all__ = ["is_stop", "normalize", "stop_phrases"]
